import torch
import os
from mmseg.registry.registry import MODELS,DATASETS
import numpy as np
import h5py
from AgriFM.datasets.transform import MapCompose
import torch.utils.data as data_utils

@DATASETS.register_module()
class MappingDataset(data_utils.Dataset):
    def __init__(self,data_toot_path, data_list_file,
                 data_pipelines,data_keys=('S2',),label_key='label'):
        '''
        Provide general data loading and preprocessing for mapping dataset,
        with data format h5.
        :param data_toot_path: directory of h5 files
        :param data_list_file: data list file, which contains the list of h5 files.
        :param data_pipelines:
            data pipelines for data preprocessing, which is a list of dicts.
            Each dict should contain the key 'type' and other keys for the specific transform.
            The 'type' should be the name of the transform class.
        :param data_keys:
            keys of the data to be loaded from h5 files, default is ('S2',).
            Support multiple keys, such as ('S2', 'Modis', 'Landsat').
            These keys must be in the h5 files.
            The data will be loaded from the h5 file with these keys.
        :param label_key:
            key of the label to be loaded from h5 files, default is 'label'.
            The label will be loaded from the h5 file with this key.
        '''
        self.data_toot_path = data_toot_path
        self.data_list_file = data_list_file
        self.data_pipelines = data_pipelines
        self.data_keys = data_keys
        self.label_key = label_key
        self.data_list = np.loadtxt(data_list_file, dtype=str).tolist()
        self.data_pipelines = MapCompose(self.data_pipelines)


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        '''
        Load data from h5 file according to the item index.
        :param item:
        :return:
            A dict containing multi-source data.
            The data is a dict with keys as self.data_keys.
            Each value in the dict is a tensor with shape (T,C, H, W),
            where T is the number of time steps (1 for single image),
            C is the number of channels, H is the height, and W is the width.
            where C is the number of channels, H is the height, and W is the width.
            The label is a tensor with key self.label_key.
        '''
        file_name= self.data_list[item]
        data_file= os.path.join(self.data_toot_path, '%s.h5'%file_name)
        data_dict={}
        with h5py.File(data_file, 'r') as f:
            for key in self.data_keys:
                if key in f.keys():
                    data_dict[key] = torch.from_numpy(f[key][:])
                else:
                    raise KeyError(f'{key} not found in {data_file}')
            if self.label_key in f.keys():
                label= torch.from_numpy(f[self.label_key][:])
            else:
                raise KeyError(f'{self.label_key} not found in {data_file}')
        label=torch.unsqueeze(label,dim=0)
        data_dict,label=self.data_pipelines(data_dict,label)
        label= label.squeeze(dim=0)
        data_dict['file_name'] = file_name
        return data_dict, label.long()






