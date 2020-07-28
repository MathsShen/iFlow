import os
import os.path as osp
import glob

from data import common
import pickle
import numpy as np
import imageio

import torch
#import torch.utils.data as data

import pickle

import numpy as np
import numpy.random as npr

class ToyData(torch.utils.data.Dataset):
    def __init__(self, path, batch_size, shuffle=True):
        # self.args = args
        # self.name = name
        # self.train = train
        # self.split = 'train' if train else 'test'

        # pkl_folder = osp.join(args.dir_data, args.data_train)
        # self.data_folder = osp.join(pkl_folder, "feats")

        # with open(osp.join(pkl_folder, "id2numimgs.pkl"), 'rb') as f:
        #     self.id2numimgs = pickle.load(f)
        # self.num_ids = len(self.id2numimgs)

        #self._scan()

        self.device = torch.device('cuda')
        self.path = path
        data = np.load(path)
        self.data = data
        print('data loaded on {}'.format(self.device))
        self.s = torch.from_numpy(data['s']).to(self.device)
        self.x = torch.from_numpy(data['x']).to(self.device)
        self.u = torch.from_numpy(data['u']).to(self.device)

        self.total_examples = self.x.shape[0]
        self.latent_dim = self.s.shape[1]
        self.aux_dim = self.u.shape[1]
        self.data_dim = self.x.shape[1]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = math.ceil(self.total_examples / self.batch_size)
        self.nps = int(self.total_examples / self.aux_dim)
        #if self.shuffle:
        #    self.idx = np.random.permutation(self.total_examples)
        #else:
        #    self.idx = np.arange(self.total_examples)


    def __getitem__(self, idx):
        # idx indicates where to get

        index_sele = npr.randint(self.total_examples)
        sample = {"x": self.x[index_sele], \
                  "u": self.u[index_sele], \
                  "s": self.s[index_sele]}
        return sample

    def __len__(self):
        return self.len
