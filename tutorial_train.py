import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from share import *

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict, compare_weights

pl.seed_everything(42, workers=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Configs
resume_path = './stable-diffusion-v1-5/control_sd15.ckpt'
batch_size = 6
logger_freq = 400
learning_rate = 1e-5
sd_locked = False
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=True)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True, drop_last=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(strategy="deepspeed_stage_2", accelerator="gpu", devices=8, callbacks=[logger], deterministic=True, max_steps=3000)
# Train!
trainer.fit(model, dataloader)
