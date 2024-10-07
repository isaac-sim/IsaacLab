
import copy
import time

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.ops import MLP

class SlipNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MLP(23, [64, 128, 128, 1])
        self.lr = 1e-4
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=self.lr
        )
        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=1),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]
    
    def compute_loss(self, batch, phase="train"):
        x, y = batch["x"], batch["y"]
        y = y.unsqueeze(-1)
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        # compute accuracy
        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat > 0.5).float()
        acc = (y_hat == y).float().mean()
        self.log(f"{phase}_loss", loss)
        self.log(f"{phase}_acc", acc)
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss(batch, "train")
        return loss
