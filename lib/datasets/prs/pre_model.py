import torch.utils.data as data
import os.path
import scipy.io as sio
import torch
import numpy as np
import ipdb

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        self.data_root = kwargs['data_root']
        split = kwargs['split']
        # self.data_root = os.path.join(data_root, split)
        
    def __getitem__(self, index):
        index += 1
        voxel_path = os.path.join(self.data_root, f'{index}_voxel.npy')
        voxel = np.load(voxel_path)
        voxel = torch.from_numpy(voxel).float()
        sample_path = os.path.join(self.data_root, f'{index}_points.npy')
        sample = np.load(sample_path)
        sample = torch.from_numpy(sample).float()
        cp_path = os.path.join(self.data_root, f'{index}_pre.npy')
        cp = np.load(cp_path)
        cp = torch.from_numpy(cp).float()

        input_dict = {'voxel': voxel, 'sample': sample, 'cp': cp.reshape(-1, 3)}

        return input_dict

    def __len__(self):
        return 3000
