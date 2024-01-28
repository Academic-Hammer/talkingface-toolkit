# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

import torchvision.models as models
import functools
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np
#from convolutional_rnn import Conv2dGRU
import torchvision
import torch.nn.init as init
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            conv3d(channel_in, channel_out, 3, 1, 1),
            conv3d(channel_out, channel_out, 3, 1, 1, activation=None)
        )

        self.lrelu = nn.ReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.block(x)

        out += residual
        out = self.lrelu(out)
        return out


def linear(channel_in, channel_out,
           activation=nn.ReLU,
           normalizer=nn.BatchNorm1d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Linear(channel_in, channel_out, bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    # init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def conv2d(channel_in, channel_out,
           ksize=3, stride=1, padding=1,
           activation=nn.ReLU,
           normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Conv2d(channel_in, channel_out,
                           ksize, stride, padding,
                           bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    # init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def conv_transpose2d(channel_in, channel_out,
                     ksize=4, stride=2, padding=1,
                     activation=nn.ReLU,
                     normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.ConvTranspose2d(channel_in, channel_out,
                                    ksize, stride, padding,
                                    bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    # init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def nn_conv2d(channel_in, channel_out,
              ksize=3, stride=1, padding=1,
              scale_factor=2,
              activation=nn.ReLU,
              normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.UpsamplingNearest2d(scale_factor=scale_factor))
    layer.append(nn.Conv2d(channel_in, channel_out,
                           ksize, stride, padding,
                           bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    # init.kaiming_normal(layer[1].weight)

    return nn.Sequential(*layer)


def _apply(layer, activation, normalizer, channel_out=None):
    if normalizer:
        layer.append(normalizer(channel_out))
    if activation:
        layer.append(activation())
    return layer


class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()

        self.emotion_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),

            nn.MaxPool2d((1,3), stride=(1,2)), #[1, 64, 12, 12]
            conv2d(64,128,3,1,1),

            conv2d(128,256,3,1,1),

            nn.MaxPool2d((12,1), stride=(12,1)), #[1, 256, 1, 12]

            conv2d(256,512,3,1,1),

            nn.MaxPool2d((1,2), stride=(1,2)) #[1, 512, 1, 6]

            )
        self.emotion_eocder_fc = nn.Sequential(
            nn.Linear(512 *6,2048),
            nn.ReLU(True),
            nn.Linear(2048,128),
            nn.ReLU(True),

            )
        self.last_fc = nn.Linear(128,8)

    def forward(self, mfcc):
       # mfcc= torch.unsqueeze(mfcc, 1)
        mfcc=torch.transpose(mfcc,2,3)
        feature = self.emotion_eocder(mfcc)
        feature = feature.view(feature.size(0),-1)
        x = self.emotion_eocder_fc(feature)
        re = self.last_fc(x)

        return re

class DisNet(nn.Module):
    def __init__(self):
        super(DisNet, self).__init__()


        self.dis_fc = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,16),
            nn.ReLU(True),
            nn.Linear(16,1),
            nn.ReLU(True)
            )


    def forward(self, feature):

        re = self.dis_fc(feature)

        return re

