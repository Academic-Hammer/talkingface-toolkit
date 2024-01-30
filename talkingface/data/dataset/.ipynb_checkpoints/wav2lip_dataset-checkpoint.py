from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from talkingface.data.dataprocess.wav2lip_process import Wav2LipAudio
import python_speech_features

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
import librosa
from talkingface.data.dataset.dataset import Dataset

from glob import glob

import os, random, cv2, argparse

class Wav2LipDataset(Dataset):
    def __init__(self, config, datasplit):
        super().__init__(config, datasplit)
        print(self.config['preprocessed_root'])
        self.all_videos = self.get_image_list(self.config['preprocessed_root'], datasplit)
        self.audio_processor = Wav2LipAudio(self.config)

    
    def get_image_list(self, data_root, split_path):
        filelist = []

        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                filelist.append(os.path.join(data_root, line))

        return filelist

    
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + self.config['syncnet_T']):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (self.config['img_size'], self.config['img_size']))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(self.config['fps'])))
        
        end_idx = start_idx + self.config['syncnet_mel_step_size']

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert self.config['syncnet_T'] == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + self.config['syncnet_T']):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != self.config['syncnet_mel_step_size']:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * self.config['syncnet_T']:
                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = self.audio_processor.load_wav(wavpath, self.config['sample_rate'])
                orig_mel = self.audio_processor.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name) 

            if (mel.shape[0] != self.config['syncnet_mel_step_size']):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2]//2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            # breakpoint()
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return {"input_frames":x, "indiv_mels":indiv_mels, "mels":mel, "gt":y}
