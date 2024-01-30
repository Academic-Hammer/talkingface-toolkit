import torch
import os


import sys
import argparse
import math
import os
import torch
import torch.nn as nn

import pickle
class PC_AVS(torch.nn.Module):
    def generate_batch(self):

        os.chdir("./talkingface/model/audio_driven_talkingface/pc_avs")
        torch.cuda.is_available()
        os.system("CUDA_VISIBLE_DEVICES=0 python -u inference.py  --name demo --meta_path_vox './misc/demo.csv' --dataset_mode voxtest --netG modulate --netA resseaudio --netA_sync ressesync --netD multiscale  --netV resnext --netE fan --model av --gpu_ids 0 --clip_len 1 --batchSize 16 --style_dim 2560 --nThreads 4 --input_id_feature --generate_interval 1 --style_feature_loss --use_audio 1 --noise_pose --driving_pose  --gen_video --generate_from_audio_only")
        os.chdir("../../../../")
        
        return self.opt
    @staticmethod
    def modify_commandline_options(parser, is_train):
        pass

    def __init__(self, opt):
        super(PC_AVS, self).__init__()
        self.linear = nn.Linear(10, 5)
        self.opt=opt

    def parameters(self):
        for param in self.children():
            if hasattr(param, 'parameters'):
                for p in param.parameters():
                    yield p


        