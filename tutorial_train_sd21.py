import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


# Configs
resume_path = './models/stable-diffusion-2-1-base/v2-1_512-nonema-pruned.ckpt'
batch_size = 3
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True, drop_last=True)
logger = ImageLogger(batch_frequency=logger_freq)
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
trainer = pl.Trainer(strategy="ddp_find_unused_parameters_true", accelerator="gpu", devices=8, precision=16, callbacks=[logger], max_steps=4000)
pl.seed_everything(42)

# Train!
trainer.fit(model, dataloader)
