import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


# def load_state_dict(ckpt_path, location='cpu'):
#     _, extension = os.path.splitext(ckpt_path)
#     if extension.lower() == ".safetensors":
#         import safetensors.torch
#         state_dict = safetensors.torch.load_file(ckpt_path, device=location)
#     else:
#         state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
#     state_dict = get_state_dict(state_dict)
#     print(f'Loaded state_dict from [{ckpt_path}]')
#     return state_dict

def load_state_dict(ckpt_path, location='cpu', exclude_buffers=None):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location).to(torch.float16)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))

    if exclude_buffers:
        state_dict = {k: v for k, v in state_dict.items() if not any(buf_name in k for buf_name in exclude_buffers)}

    print(f'Loaded state_dict from [{ckpt_path}]')
    # for name, param in state_dict.items():
    #     print(f"Layer: {name}, Shape: {param.shape}")
    return state_dict



def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

def compare_weights(state_dict, layer1_name, layer2_name):
    if layer1_name not in state_dict:
        print(f"Layer {layer1_name} not found!")
        return False
    if layer2_name not in state_dict:
        print(f"Layer {layer2_name} not found!")
        return False
    weight1 = state_dict[layer1_name]
    weight2 = state_dict[layer2_name]
    
    are_equal = torch.equal(weight1, weight2)
    if are_equal:
        print(f"The weights are identical!")
    else:
        print(f"The weights are different!")
    return are_equal