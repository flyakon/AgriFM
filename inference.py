# -*- coding: utf-8 -*-
# @Time    : 10/12/2024 6:32 pm
# @Author  : Wenyuan Li
# @File    : inference_vis.py
# @Description :
import os
import numpy as np
import torch
import cv2
from mmseg.registry import MODELS,DATASETS
from AgriFM.utils import path_utils
import argparse
from mmengine.config import Config, DictAction
import torch.utils.data as data_utils
#import load_checkpoint in mmseg
from mmengine.runner import load_state_dict,load_checkpoint
import tqdm
from skimage import io
import copy
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('result_path', help='path to save the inference results')
    args=parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args()
    config_file=args.config
    checkpoint_file=args.checkpoint

    options=["TILED=TRUE","COMPRESS=DEFLATE","NUM_THREADS=4","ZLEVEL=9"]
    cfg=Config.fromfile(config_file)
    model=MODELS.build(cfg.model)
    dataset_cfg=cfg.test_dataloader.dataset
    dataset=DATASETS.build(dataset_cfg)
    dataloader=data_utils.DataLoader(dataset,batch_size=4,shuffle=False,num_workers=4)
    resutl_path=cfg.result_path
    if not os.path.exists(resutl_path):
        os.makedirs(resutl_path)
    #resume from a checkpoint
    load_checkpoint(model,checkpoint_file,strict=True)
    model.eval()
    model.to('cuda')
    for data,label in tqdm.tqdm(dataloader):
        out_data=copy.deepcopy(data)
        file_path = data.pop('file_name')
        data=model.data_preprocessor(data)
        with torch.no_grad():
            logits=model(data,mode='tensor')
        for i in range(len(file_path)):
            cls_pred=torch.argmax(logits[i],dim=0)
            cls_pred=cls_pred.cpu().numpy()
            tile_name=file_path[i]
            pred_mask_path=os.path.join(resutl_path,'%s_pred.png'%tile_name)
            io.imsave(pred_mask_path,cls_pred.astype(np.uint8),check_contrast=False)











