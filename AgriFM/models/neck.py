
import torch
import torch.nn as nn
import os
from mmseg.models.builder import NECKS,MODELS
from mmengine.runner import load_checkpoint


@NECKS.register_module()
class MultiFusionNeck(nn.Module):
    def __init__(self,embed_dim,in_feature_key=('S2',),
                 feature_size=(16,16),out_size=(256,256),
                 in_fusion_key_list=({'S2':512,'HLS':512},
                                     {'S2':256},
                                     {'S2':128,},
                                     )
                 ):
        super(MultiFusionNeck, self).__init__()
        self.embed_dim=embed_dim
        self.fusion_list=nn.ModuleList()
        self.in_feature_key=in_feature_key
        self.feature_size=feature_size
        self.out_size=out_size
        self.in_fusion_key_list=in_fusion_key_list

        if len(in_feature_key)==1:
            self.in_conv=nn.Identity()
        else:
            self.in_conv=nn.Sequential(
                nn.Conv2d(len(in_feature_key)*self.embed_dim,self.embed_dim,3,1,1),
                nn.BatchNorm2d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dim,self.embed_dim,3,1,1),
            )
        embed_dim=self.embed_dim
        pre_embed = embed_dim
        for fusion_keys in in_fusion_key_list:
            in_embed=sum(fusion_keys.values())
            fusion=nn.Sequential(
                nn.Conv2d(in_embed+pre_embed,pre_embed,3,1,1),
                nn.BatchNorm2d(pre_embed),
                nn.ReLU(inplace=True),
                nn.Conv2d(pre_embed,embed_dim,3,1,1),
            )
            self.fusion_list.append(fusion)
            pre_embed=embed_dim


        self.out_conv=nn.Sequential(
            nn.Conv2d(pre_embed,pre_embed,3,1,1),
            nn.BatchNorm2d(pre_embed),
            nn.ReLU(inplace=True),
            nn.Conv2d(pre_embed,pre_embed,3,1,1),
        )

    def forward(self,inputs):
        in_features=[]
        for key in self.in_feature_key:
            features=inputs[key]['encoder_features']
            features=torch.nn.functional.interpolate(features,self.feature_size,mode='bilinear',align_corners=False)
            in_features.append(features)
        in_features=torch.cat(in_features,dim=1)
        in_features=self.in_conv(in_features)

        for i,fusion_keys in enumerate(self.in_fusion_key_list):
            in_features=torch.nn.functional.interpolate(in_features,scale_factor=2,mode='bilinear',align_corners=False)
            in_features_h, in_features_w=in_features.shape[-2:]
            in_features_idx=len(self.in_fusion_key_list)-i-1
            fusion_features=[]
            for key in fusion_keys:
                features=inputs[key]['features_list'][in_features_idx]
                features=torch.nn.functional.interpolate(features,(in_features_h,in_features_w),mode='bilinear',align_corners=False)
                fusion_features.append(features)
            fusion_features=torch.cat(fusion_features,dim=1)
            in_features=torch.cat([in_features,fusion_features],dim=1)
            in_features=self.fusion_list[i](in_features)
        out_features=self.out_conv(in_features)
        out_features=torch.nn.functional.interpolate(out_features,self.out_size,mode='bilinear',align_corners=False)
        return out_features









