import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

import open_clip
from ldm.util import default, count_params


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False
    
    def add_embedding(self, tokens, path_to_embeddings):
        """
        Adds new tokens and their corresponding embeddings to the tokenizer and model.

        Args:
            tokens (list): List of new tokens to add (e.g., ["<style-A-tumor>", "<style-B-tumor>"]).
            path_to_embeddings (list): List of file paths to the embeddings for each token.
        """
        from safetensors.torch import load_file

        assert len(tokens) == (1 + len(path_to_embeddings)), "Number of tokens must match number of embedding files."
        
        # for token, path_to_embedding in zip(tokens, path_to_embeddings):
        #     # Load embedding
        #     try:
        #         embedding = load_file(path_to_embedding)[token].to(self.device)
        #     except KeyError:
        #         raise KeyError(f"Token '{token}' not found in embedding file: {path_to_embedding}")
            
        #     # Log tokenizer length before adding the new token
        #     print(f"Length of tokenizer before adding token '{token}': {len(self.tokenizer)}")
            
        #     # Add token to tokenizer
        #     self.tokenizer.add_tokens([token])  # Ensure token is added as a single item
        #     print(f"Length of tokenizer after adding token '{token}': {len(self.tokenizer)}")
            
        #     # Resize model's embedding layer
        #     self.transformer.resize_token_embeddings(len(self.tokenizer))
            
        #     # Assign embedding to the last position (newly added token)
        #     self.transformer.get_input_embeddings().weight.data[-1] = embedding
        #     print(f"Added embedding for token '{token}' from {path_to_embedding}.")
        
        # # Ensure embeddings are frozen (non-trainable)
        # self.transformer.get_input_embeddings().weight.requires_grad = False
        # print("New embeddings are frozen (non-trainable).")

        embedding_0 = load_file(path_to_embeddings[0])[tokens[0]].to(self.device)
        embedding_1 = load_file(path_to_embeddings[0])[tokens[1]].to(self.device)
        embedding_2 = load_file(path_to_embeddings[1])[tokens[0]].to(self.device)
        embedding_3 = load_file(path_to_embeddings[1])[tokens[2]].to(self.device)
        
        # Log tokenizer length before adding the new token
        print(f"Length of tokenizer before adding tokens: {len(self.tokenizer)}")
        
        # Add token to tokenizer
        tokens = ["<A-object>", "<A-style>", "<B-object>", "<B-style>"]
        self.tokenizer.add_tokens(tokens)  # Ensure token is added as a single item
        print(f"Length of tokenizer after adding tokens '{tokens}': {len(self.tokenizer)}")
        
        tokens_index = self.tokenizer.convert_tokens_to_ids(tokens)
        print(f"Index of tokens '{tokens}' in tokenizer: {tokens_index}")
        
        # Resize model's embedding layer
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        
        # Assign embedding to the last position (newly added token)
        # embedding_0 = self.slerp(0.5, embedding_0, embedding_2)
        self.transformer.get_input_embeddings().weight.data[-4] = embedding_0
        print(f"Added embedding for token '{tokens[0]}'.")
        
        self.transformer.get_input_embeddings().weight.data[-3] = embedding_1
        print(f"Added embedding for token '{tokens[1]}'.")
        
        self.transformer.get_input_embeddings().weight.data[-2] = embedding_2
        print(f"Added embedding for token '{tokens[2]}'.")
        
        self.transformer.get_input_embeddings().weight.data[-1] = embedding_3
        print(f"Added embedding for token '{tokens[3]}'.")
        
        # Ensure embeddings are frozen (non-trainable)
        self.transformer.get_input_embeddings().weight.requires_grad = False
        print("New embeddings are frozen (non-trainable).")
    
    def slerp(self, t, v0, v1):
        v0_norm = v0 / torch.norm(v0, dim=-1, keepdim=True)
        v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True)
        dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        return (torch.sin((1 - t) * omega) / sin_omega) * v0 + (torch.sin(t * omega) / sin_omega) * v1

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14",
                 version="/opt/data/private/QiuKunpeng/Diffusion/ControlNet/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin", 
                 device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


