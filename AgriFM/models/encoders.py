
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Union
from mmseg.registry import MODELS
from mmengine.model import BaseModule,BaseModel


@MODELS.register_module()
class MultiModalEncoder(BaseModel):
    def __init__(self,encoders_cfg):
        super().__init__()
        self.encoders=nn.ModuleDict()
        for name,cfg in encoders_cfg.items():
            self.encoders[name]=MODELS.build(cfg)
    def forward(self,
                inputs:dict,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        outputs={}
        for name,encoder in self.encoders.items():

            inputs_data=inputs[name]
            outputs[name]=encoder(inputs_data,data_samples,mode)
            if name in inputs.keys():
                inputs.pop(name)
        outputs.update(inputs)
        return outputs