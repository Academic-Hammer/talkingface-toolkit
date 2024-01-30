from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from talkingface.data.dataprocess.dfrf_process import DFRF_process
import python_speech_features

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
import json
from talkingface.data.dataset.dataset import Dataset
import imageio

from glob import glob

import os, random, cv2, argparse

# 借助代码提供的零散模块对Dataset进行规范化重组
# 我们首先要明确每次迭代用到的一组数据的内容：
# 是一个视频中提取出来的一帧图片吗，那么getitem需要的id就是这个图片的id号


from talkingface.utils.dfrf_util.load_audface_multiid import load_audface_data
from talkingface.utils.dfrf_util.run_nerf_helpers_deform import *
from talkingface.utils.dfrf_util.run_nerf_deform import *

class dfrfDataset(Dataset):
    def __init__(self, config, datasplit):
        print("dfrfDataset")
        if "train" in datasplit:
            datasplit = "train"
        elif "val" in datasplit:
            datasplit = "val"
        print(datasplit)

        super().__init__(config, datasplit)

        parser = config_parser()
        self.args = parser.parse_args()
        print(self.args.near, self.args.far)
        print(self.args)
        # 返回了所有训练需要的数据
        if self.args.dataset_type == 'audface':
            if self.args.with_test == 1:
                self.all_data = load_audface_data(self.args.datadir, self.args.testskip, self.args.test_file, self.args.aud_file, need_lip=True, need_torso = self.args.need_torso, bc_type=self.args.bc_type)
                images, poses, auds, bc_img, hwfcxy, lip_rects, torso_bcs, _ = self.all_data
                #images = np.zeros(1)
            else:
                self.all_data = load_audface_data(self.args.datadir, self.args.testskip, train_length = self.args.train_length, need_lip=True, need_torso = self.args.need_torso, bc_type=self.args.bc_type)
                images, poses, auds, bc_img, hwfcxy, sample_rects, i_split, id_num, lip_rects, torso_bcs = self.all_data

            #print('Loaded audface', images['0'].shape, hwfcxy, args.datadir)

            if self.args.with_test == 0:
                print('Loaded audface', images['0'].shape, hwfcxy, self.args.datadir)
                #all id has the same split, so this part can be shared
                self.i_train, self.i_val = i_split['0']
            else:
                print('Loaded audface', len(images), hwfcxy, self.args.datadir)
            self.near = self.args.near
            self.far = self.args.far
        else:
            print('Unknown dataset type', self.args.dataset_type, 'exiting')
            return
            # Cast intrinsics to right types

        H, W, focal, cx, cy = hwfcxy
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        hwfcxy = [H, W, focal, cx, cy]

        intrinsic = np.array([[focal, 0., W / 2],
                            [0, focal, H / 2],
                            [0, 0, 1.]])
        self.intrinsic = torch.Tensor(intrinsic).to(device).float()
        print("dfrfDataset")

    def __len__(self):
        if self.split == 'train':
            return len(self.i_train)
        if self.split == 'val':
            return len(self.i_val)

    def __getitem__(self, idx):
        # 这个是训练一次需要的数据列表，顺序是不一样的
        return self.all_data[idx]