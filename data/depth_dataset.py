### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset
import h5py
import numpy as np
import torch

max_depth = np.inf

class DepthDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.data_dir = os.path.join(opt.dataroot, 'train')
        # self.A_paths = sorted(make_dataset(self.dir_A))
        self.data_paths = []
        self.dataset_size = 0
        assert os.path.isdir(self.data_dir), '%s is not a valid directory' % self.data_dir

        for root, _, fnames in sorted(os.walk(self.data_dir)):
            for fname in fnames:
                path = os.path.join(root, fname)
                self.data_paths.append(path)
                self.dataset_size += 1

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        try:
            h5f = h5py.File(data_path, "r")
        except OSError:
            return dict()
        rgb = np.array(h5f['rgb'])
        depth = np.array(h5f['depth'])
        depth = np.dstack((depth, depth, depth))
        if self.opt.sparse:
            rgbd = self.create_sparse_depth(rgb, depth)
            rgbd = np.transpose(rgbd, (2, 0, 1))
            #print("rgbd", rgbd.shape)
            rgbd = torch.tensor(rgbd, dtype=torch.float)
        rgb = torch.tensor(rgb, dtype=torch.float)
        depth = np.transpose(depth, (2, 0, 1))  # chanel first
        depth = torch.tensor(depth, dtype=torch.float)
        input_dict = {'A': rgb, 'B': depth,
                      'A_paths': data_path, 'B_paths': data_path}
        if self.opt.sparse:
            input_dict['A'] = rgbd
        return input_dict

    def __len__(self):
        return len(self.data_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'DepthDataset'



