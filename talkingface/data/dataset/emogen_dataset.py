import os
from os.path import join, isfile, isdir, splitext, basename
import numpy as np
import random 
from glob import glob

import torch
from torch.utils.data import DataLoader, Dataset
import cv2

from hparams import hparams

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
# 假设其他函数如 to_categorical 等已定义

emotion_dict = {'ANG':0, 'DIS':1, 'FEA':2, 'HAP':3, 'NEU':4, 'SAD':5}
intensity_dict = {'XX':0, 'LO':1, 'MD':2, 'HI':3}
emonet_T = 5

class EmotionDataset(Dataset):
    def __init__(self, config, datasplit):
        super().__init__()
        self.config = config
        self.split = datasplit

        path_key = f"{datasplit}_filelist"
        if path_key not in self.config:
            raise ValueError(f"Path for {datasplit} data is not defined in config")

        self.path = self.config[path_key]
        
        self.all_videos = [f for f in os.listdir(self.path) if isdir(join(self.path, f))]

        self.filelist = []
        for filename in self.all_videos:
            labels = splitext(filename)[0].split('_')
            emotion = emotion_dict[labels[2]]
            emotion_intensity = intensity_dict[labels[3]]

            # For validation, only use high intensity
            if datasplit == 'val' and emotion_intensity != 3:
                continue
            
            self.filelist.append((filename, emotion, emotion_intensity))

        print('Num files: ', len(self.filelist))

    # other methods like get_frame_id, get_window, read_window, and prepare_window remain the same

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        while True:
            idx = random.randint(0, len(self.filelist) - 1)
            filename = self.filelist[idx]
            vidname = filename[0]
            emotion = int(filename[1])
            emotion = to_categorical(emotion, num_classes=6)

            img_names = list(glob(join(self.path, vidname, '*.jpg')))

            if len(img_names) <= 3 * emonet_T:
                continue
            img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            if window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None: 
                continue

            x = self.prepare_window(window)
            x = torch.FloatTensor(x)

            data = {
                'input': x,
                'emotion': emotion
            }

            return data

# Example usage
# config = {'train_filelist': 'path/to/train', 'val_filelist': 'path/to/val', ...}
# dataset = EmotionDataset(config, 'train')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
