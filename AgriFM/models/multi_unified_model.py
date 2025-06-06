
from typing import Optional, Union, Dict
import torch
from mmseg.models.builder import MODELS
from mmengine.model import BaseModel
from mmengine.runner import load_state_dict,load_checkpoint

@MODELS.register_module()
class MultiUnifiedModel(BaseModel):
    def __init__(self,encoders,head,neck=None,load_from=None):
        super().__init__()
        self.encoders=MODELS.build(encoders)
        if neck is not None:
            self.neck=MODELS.build(neck)
        else:
            self.neck=None
        self.heads=MODELS.build(head)

        if load_from is not None:
            load_checkpoint(self,load_from,strict=False)

    def forward(self,
                inputs:dict,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        outputs=self.encoders(inputs,data_samples,mode)
        if self.neck is not None:
            outputs=self.neck(outputs)
        if mode=='tensor' or mode=='predict':
            outputs=self.heads(outputs,mode=mode)
        else:
            logits,outputs=self.heads(outputs,data_samples,mode)
            self.result_list=logits
        return outputs





