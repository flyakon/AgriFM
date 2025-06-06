
import torch
from mmseg.models.builder import LOSSES
import torch.nn as nn
import numpy as np



@LOSSES.register_module()
class CropCEloss(torch.nn.Module):
    def __init__(self,ignore_index=-1):
        super().__init__()
        self.criterion=torch.nn.CrossEntropyLoss(reduction='none',ignore_index=ignore_index)

    def forward(self,pred,label):
        loss=self.criterion(pred,label)
        return {'crop_ce_loss': loss.mean() if isinstance(loss, torch.Tensor) else np.mean(loss)}
