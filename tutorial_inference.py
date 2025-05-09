import os
import torch
import random
from share import *
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_sample import MyDataset
from cldm.model import create_model, load_state_dict
from safetensors.torch import load_file

pl.seed_everything(0, workers=True)

BATCH_SIZE = 1
CKPT_PATH = "./lightning_logs/version_0/checkpoints/epoch/merged_pytorch_model.pth"
RESULT_DIR = "./generated_results/version_0/"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

learning_rate = 1e-5
logger_freq = 300
sd_locked = False
only_mid_control = False


def get_model():
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(CKPT_PATH, location='cpu'), strict=False)
    
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.to("cuda:8")
    model.eval()
    return model


def log_local(save_dir, images, batch_idx):
    samples_root = os.path.join(save_dir, "images")
    mask_root = os.path.join(save_dir, "masks")

    for k in images:
        for idx, image in enumerate(images[k]):
            if k == "samples_cfg_scale_9.00_mask":
                image = (image + 1.0) / 2.0
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
                path = os.path.join(samples_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(image).save(path)

            if k == "control_mask":
                image = image.permute(1, 2, 0)
                image = image.squeeze(-1).numpy()
                mask = (image * 255).astype(np.uint8)
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
                path = os.path.join(mask_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(mask).convert('1').save(path)


if __name__ == "__main__":
    with torch.cuda.device(8):
        model = get_model()

        dataset = MyDataset()
        dataloader = DataLoader(dataset, num_workers=4, batch_size=1, shuffle=False)
        result_root = RESULT_DIR
        finaldir = os.path.join(result_root)
        os.makedirs(finaldir, exist_ok=True)
        with torch.no_grad():
            with model.ema_scope():
                for idx, batch in enumerate(dataloader):
                    print(idx)
                    images = model.log_images(
                        batch,
                        N=BATCH_SIZE,
                        ddim_steps = 50,
                        ddim_eta = 0.0,
                    )
                    for k in images:
                        if isinstance(images[k], torch.Tensor):
                            images[k] = images[k].detach().cpu()
                            images[k] = torch.clamp(images[k], -1.0, 1.0)

                    log_local(finaldir, images, idx)

