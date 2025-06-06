# -*- coding: utf-8 -*-
# @Time    : 2023/11/23 13:38
# @Author  : Wenyuan Li
# @File    : transform.py
# @Description :
import torch

import numpy as np
import random
import torchvision.transforms.functional as F
import sys
import collections
import copy
from mmseg.registry.registry import TRANSFORMS

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

@TRANSFORMS.register_module()
class MapVerticalFlip():
    """Vertically flip the given PIL Image and bounding boxes randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
        """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict:dict,label=None):
        """
        Args:
            img (PIL Image): Image to be flipped.
            bnd_boxes (ndarray): [Nx4] bndboxes (xmin,ymihn,xmax,ymax)
        Returns:
            PIL Image: Randomly flipped image.
            ndarrsy: [Nx4] Randomly flipped bnndboxes
        """
        if random.random() < self.p:
            for key in data_dict.keys():
                data_dict[key]=F.vflip(data_dict[key])
            if label is not None:
                label=F.vflip(label)
        return data_dict,label



    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

@TRANSFORMS.register_module()
class MapHorizontalFlip():
    """Horizontally flip the given PIL Image and bounding boxes randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
        """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict:dict,label=None):
        """
        Args:
            img (PIL Image): Image to be flipped.
            bnd_boxes (ndarray): [Nx4] bndboxes (xmin,ymihn,xmax,ymax)
        Returns:
            PIL Image: Randomly flipped image.
            ndarrsy: [Nx4] Randomly flipped bnndboxes
        """
        if random.random() < self.p:
            for key in data_dict.keys():
                data_dict[key]=F.hflip(data_dict[key])
            if label is not None:
                label=F.hflip(label)
        return data_dict,label

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
@TRANSFORMS.register_module()
class MapRotate():
    """Rotate the given PIL Image and bounding boxes randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
            angle (int): Rotated angle [0,90,180,270]
        """

    def __init__(self, p=0.5,interpolation=None):
        self.p = p
        self.interpolation_mode=interpolation


    def __call__(self, data_dict:dict,label=None):
        """
        Args:
            img (PIL Image): Image to be flipped.
            bnd_boxes (ndarray): [Nx4] bndboxes (xmin,ymihn,xmax,ymax)
        Returns:
            PIL Image: Randomly flipped image.
            ndarrsy: [Nx4] Randomly flipped bnndboxes
        """
        angles=[0,90,180,270]
        idx=random.randint(0,3)
        angle=angles[idx]
        if random.random() < self.p:
            for key in data_dict.keys():
                if self.interpolation_mode is not None and isinstance(self.interpolation_mode,dict):
                    if key in self.interpolation_mode.keys():
                        interpolation_mode=self.interpolation_mode[key]
                    else:
                        interpolation_mode=F.InterpolationMode.BILINEAR
                else:
                    interpolation_mode=F.InterpolationMode.BILINEAR
                data_dict[key]=F.rotate(data_dict[key],angle,expand=False,
                                            interpolation=interpolation_mode)
            if label is not None:
                label=F.rotate(label,angle,expand=False,
                                            interpolation=F.InterpolationMode.NEAREST)
        return data_dict,label


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

@TRANSFORMS.register_module()
class MapRandomCrop():
    """Horizontally flip the given PIL Image and bounding boxes randomly with a given probability.

        Args:
            p (float): probability of the image being flipped. Default value is 0.5
        """
    def __init__(self,crop_ratio_min=0.8,crop_ratio_max=0.95,base_size=192):
        self.crop_ratio_min=crop_ratio_min
        self.crop_ratio_max=crop_ratio_max
        self.base_size=base_size

    @staticmethod
    def get_params(w, h, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        # w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self,data_dict:dict,label=None):
        self.crop_ratio = random.random()
        base_size=self.base_size
        self.crop_ratio = self.crop_ratio * (self.crop_ratio_max - self.crop_ratio_min) + self.crop_ratio_min
        base_height= int(base_size * self.crop_ratio)
        base_width = int(base_size * self.crop_ratio)
        i, j, h, w = self.get_params(base_size,base_size, (base_width, base_height))
        for key in data_dict.keys():
            img=data_dict[key]
            img_height,img_width=img.shape[-2:]
            crop_i=int(i*img_height/base_size)
            crop_j=int(j*img_width/base_size)
            crop_h=int(h*img_height/base_size)
            crop_w=int(w*img_width/base_size)
            data_dict[key]=F.crop(img, crop_i, crop_j, crop_h, crop_w)
        if label is not None:
            label_height,label_width=label.shape[-2:]
            crop_i=int(i*label_height/base_size)
            crop_j=int(j*label_width/base_size)
            crop_h=int(h*label_height/base_size)
            crop_w=int(w*label_width/base_size)
            label=F.crop(label, crop_i, crop_j, crop_h, crop_w)
        return data_dict,label



@TRANSFORMS.register_module()
class MapCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
    """

    def __init__(self, transforms):
        self.transforms=[]
        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, '
                                    f'but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)}')


    def __call__(self, *args):
        data=args
        for t in self.transforms:
            data= t(*data)
        return data

    def reverse(self,*args):
        data = args
        for t in self.transforms:
            data = t.reverse(*data)
        return data
@TRANSFORMS.register_module()
class MapToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, data_dict:dict,label=None):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        for key in data_dict.keys():
            if not isinstance(data_dict[key],torch.Tensor):
                data_dict[key]=F.to_tensor(data_dict[key])
        if label is not None and not isinstance(label,torch.Tensor):
            label=F.to_tensor(label)
        return data_dict,label

    def __repr__(self):
        return self.__class__.__name__ + '()'

@TRANSFORMS.register_module()
class MapResize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR):
        self.size=size
        self.interpolation = interpolation

    def __call__(self, data_dict:dict,label=None):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """

        for key in data_dict.keys():
            if self.interpolation is not None and isinstance(self.interpolation,dict):
                if key in self.interpolation.keys():
                    interpolation=self.interpolation[key]
                else:
                    interpolation=F.InterpolationMode.BILINEAR
            else:
                interpolation=F.InterpolationMode.BILINEAR
            data_dict[key]=F.resize(data_dict[key], self.size[key], interpolation)
        if label is not None:
            label=F.resize(label, self.size['label'], F.InterpolationMode.NEAREST)
        return data_dict,label


    def __repr__(self):

        return self.__class__.__name__ + '(size={0})'.format(self.size)

@TRANSFORMS.register_module()
class MapNormalize(object):
    def __init__(self,mean,std,norm_keys=None):
        self.mean=mean
        self.std=std
        for key in mean.keys():
            self.mean[key]=np.array(self.mean[key])
            self.mean[key]=np.expand_dims(np.expand_dims(self.mean[key],axis=-1),axis=-1)

        for key in std.keys():
            self.std[key]=np.array(self.std[key])
            self.std[key]=np.expand_dims(np.expand_dims(self.std[key],axis=-1),axis=-1)
        self.norm_keys=norm_keys

    def __call__(self, data_dict:dict,label=None):
        for key in data_dict.keys():
            if self.norm_keys is not None and key not in self.norm_keys:
                continue
            data_dict[key]=data_dict[key].float()
            data_dict[key]=F.normalize(data_dict[key], self.mean[key], self.std[key])
        return data_dict,label




