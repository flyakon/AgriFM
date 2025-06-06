
from typing import Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import MODELS,LOSSES

from mmengine.model import BaseModule,BaseModel
@MODELS.register_module()
class CropFCNHead(BaseModel):
    def __init__(self,embed_dim,num_classes,loss_model):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_classes=num_classes
        self.loss_model=LOSSES.build(loss_model)
        self.head=nn.Sequential(
            nn.Conv2d(self.embed_dim,self.embed_dim//2,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim//2,self.num_classes,kernel_size=1,stride=1,padding=0)
        )

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        logits=self.head(inputs)
        if mode=='loss':
            loss=self.loss_model(logits,data_samples)
            return logits,loss
        else:
            return logits


