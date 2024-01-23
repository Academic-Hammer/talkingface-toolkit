from os.path import dirname, join, basename, isfile
import talkingface.data.dataprocess.audio_process as audio
from talkingface.data.dataprocess.hparams_Base import hparams, get_image_list
import torch
import numpy as np
from talkingface.data.dataset.dataset import Dataset

from glob import glob

import os, random, cv2, argparse


class syncNetDataset(Dataset):
    def __init__(self, split, data_root, syncnet_T, syncnet_mel_step_size):
        self.all_videos = get_image_list(data_root, split)
        self.av_offset_shift = 0
        self.syncnet_T = syncnet_T
        self.syncnet_mel_step_size = syncnet_mel_step_size

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):

        start_frame_num = self.get_frame_id(start_frame)

        start_frame_num = start_frame_num + self.av_offset_shift

        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + self.syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def read_window(self, window_fnames, flip_flag=False):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            if flip_flag:
                img = np.flip(img, axis=1).copy()
            window.append(img)

        return window

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):

        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * self.syncnet_T:
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)
            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue

            window = self.read_window(window_fnames, flip_flag=True)

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != self.syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1] // 2:]
            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y