import os
import pytorch_lightning as pl
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torch

from source.privileged_learning.cube_dataset import CubeSlipDataset
from source.privileged_learning.slip_net import SlipNet



def main():
    exp_name = "1004_slip_net"
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    logger = WandbLogger(
        project="force-tool",
        name=exp_name,
        resume='allow',
        settings=wandb.Settings(start_method='thread'),
    )
    trainer = pl.Trainer(
        benchmark=True,
        logger=logger,
        max_epochs=200,
        accelerator='gpu',
    )
    train_ds = CubeSlipDataset()
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    model = SlipNet()
    trainer.fit(model, train_loader)
     

if __name__ == '__main__':
    main()