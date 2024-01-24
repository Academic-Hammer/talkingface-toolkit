"""This module implements an abstract base class (ABC) 'dataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import torch
import torch.utils.data as data
from abc import ABC, abstractmethod
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, datasplit):

        """
        args: datasplit: str, 'train', 'val' or 'test'(这个参数必须要有, 提前将数据集划分为train, val和test三个部分,
                具体参数形式可以自己定,只要在你的dataset子类中可以获取到数据就可以, 
                对应的配置文件的参数为:train_filelist, val_filelist和test_filelist)
                
        """

        self.config = config
        self.split = datasplit

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser