import os
import random
from glob import glob
from os.path import basename, dirname, isfile, join

import cv2
import librosa
import numpy as np
import python_speech_features
import torch
import torch.backends.cudnn as cudnn
from imageio import mimread
from skimage import img_as_float32, io, transform
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils import data as data_utils
from tqdm import tqdm

from talkingface.data.dataprocess.wav2lip_process import Wav2LipAudio
from talkingface.data.dataset.dataset import Dataset
from talkingface.utils.augmentation import AllAugmentationTransform
from talkingface.utils.filter1 import OneEuroFilter


class EAMMDataset(Dataset):
    from pathlib import Path
    
    def __init__(self, config, datasplit):
        super().__init__(config, datasplit)
        self.type = config['dataset_name']
        if self.config['train']:
            self._build_dataset()
        else:
            self.videos = np.random.rand(10,256,256,3) #! 伪造数据，以便跑通流程
        
    def _build_dataset(self):
        if self.type == 'Vox':
            self._init_vox()
        elif self.type == 'LRW':
            self._init_lrw()
        elif self.type == 'MEAD':
            self._init_mead()
        
    def _init_vox(self):
        self.root_dir = self.config['dataset_root_dir']
        self.audio_dir = os.path.join(self.root_dir,'MFCC')
        self.image_dir = os.path.join(self.root_dir,'align_img')
        self.pose_dir = os.path.join(self.root_dir,'align_pose')
        
        assert len(os.listdir(self.audio_dir)) == len(os.listdir(self.image_dir)), 'audio and image length not equal'

        #! 作者没有告知这里是什么，如何得到
        self.videos=np.load('/mnt/lustre/share_data/jixinya/VoxCeleb1_Cut/right.npy')
        self.frame_shape = tuple(self.config['dataset_frame_shape'])
        self.pairs_list = None
        self.id_sampling = self.config['dataset_id_sampling']

        if os.path.exists(os.path.join(self.pose_dir, 'train_fo')):
            assert os.path.exists(os.path.join(self.pose_dir, 'test_fo'))
            print("Use predefined train-test split.")
            if self.id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(self.image_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = np.load('/mnt/lustre/share_data/jixinya/VoxCeleb1_Cut/right.npy')# get_list(self.pose_dir, 'train_fo')
    
            self.image_dir = os.path.join(self.image_dir, 'train_fo' if self.datasplit else 'test_fo')
            self.audio_dir = os.path.join(self.audio_dir, 'train' if self.datasplit else 'test')
            self.pose_dir = os.path.join(self.pose_dir, 'train_fo' if self.datasplit else 'test_fo')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=self.config['seed'], test_size=0.2)

        if self.datasplit:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = self.datasplit

        if self.is_train:
            self.transform = AllAugmentationTransform(
                flip_param={
                    'horizontal_flip': self.config['dataset_augmentation_flip_horizontal_flip'],
                    'time_flip': self.config['dataset_augmentation_flip_time_flip'],
                },
                jitter_param={
                    'brightness': self.config['dataset_augmentation_jitter_brightness'],
                    'contrast': self.config['dataset_augmentation_jitter_contrast'],
                    'saturation': self.config['dataset_augmentation_jitter_saturation'],
                    'hue': self.config['dataset_augmentation_jitter_hue'],
                }
            )
        else:
            self.transform = None

    def _init_lrw(self):
        self.root_dir = self.config['dataset_root_dir']
        self.audio_dir = os.path.join(self.root_dir,'MFCC')
        self.image_dir = os.path.join(self.root_dir,'Image')
        self.pose_dir = os.path.join(self.root_dir,'pose')
        assert len(os.listdir(self.audio_dir)) == len(os.listdir(self.image_dir)), 'audio and image length not equal'

        self.frame_shape = tuple(self.config['dataset_frame_shape'])
       
        self.id_sampling = self.config['dataset_id_sampling']

        if os.path.exists(os.path.join(self.pose_dir, 'train_fo')):
            assert os.path.exists(os.path.join(self.pose_dir, 'test_fo'))
            print("Use predefined train-test split.")
            if self.id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(self.image_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos =  np.load('../LRW/list/train_fo.npy')# get_list(self.pose_dir, 'train_fo')
            test_videos=np.load('../LRW/list/test_fo.npy')
           
            self.image_dir = os.path.join(self.image_dir, 'train_fo' if self.datasplit else 'test_fo')
            self.audio_dir = os.path.join(self.audio_dir, 'train' if self.datasplit else 'test')
            self.pose_dir = os.path.join(self.pose_dir, 'train_fo' if self.datasplit else 'test_fo')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=self.config['seed'], test_size=0.2)

        if self.datasplit:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = self.datasplit

        if self.is_train:
            self.transform = AllAugmentationTransform(
                flip_param={
                    'horizontal_flip': self.config['dataset_augmentation_flip_horizontal_flip'],
                    'time_flip': self.config['dataset_augmentation_flip_time_flip'],
                },
                jitter_param={
                    'brightness': self.config['dataset_augmentation_jitter_brightness'],
                    'contrast': self.config['dataset_augmentation_jitter_contrast'],
                    'saturation': self.config['dataset_augmentation_jitter_saturation'],
                    'hue': self.config['dataset_augmentation_jitter_hue'],
                }
            )
        else:
            self.transform = None
    
    def _init_mead(self):
        self.root_dir = self.config['dataset_root_dir']
        self.audio_dir = os.path.join(self.root_dir,'MEAD_MFCC')
        self.image_dir = os.path.join(self.root_dir,'MEAD_fomm_crop')
        self.pose_dir = os.path.join(self.root_dir,'MEAD_fomm_pose_crop')
        self.videos = np.load('/mnt/lustre/share_data/jixinya/MEAD/MEAD_fomm_audio_less_crop.npy')
        self.dict = np.load('/mnt/lustre/share_data/jixinya/MEAD/MEAD_fomm_neu_dic_crop.npy',allow_pickle=True).item()
        self.frame_shape = tuple(self.config['dataset_frame_shape'])
        self.id_sampling = self.config['dataset_id_sampling']
        if os.path.exists(os.path.join(self.root_dir, 'train')):
            assert os.path.exists(os.path.join(self.root_dir, 'test'))
            print("Use predefined train-test split.")
            if self.id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(self.root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(self.root_dir, 'train'))
            test_videos = os.listdir(os.path.join(self.root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if self.datasplit else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=self.config['seed'], test_size=0.2)
        if self.datasplit:
            self.videos = train_videos
        else:
            self.videos = test_videos
        self.is_train = self.datasplit
        if self.is_train:
            self.transform = AllAugmentationTransform(
                crop_mouth_param={
                    'center_x': self.config['dataset_augmentation_crop_mouth_center_x'],
                    'center_y': self.config['dataset_augmentation_crop_mouth_center_y'],
                    'mask_width': self.config['dataset_augmentation_crop_mouth_mask_width'],
                    'mask_height': self.config['dataset_augmentation_crop_mouth_mask_height'],
                },
                rotation_param={
                    'degrees': self.config['dataset_augmentation_rotation_degrees'],
                },
                perspective_param={
                    'pers_num': self.config['dataset_augmentation_perspective_pers_num'],
                    'enlarge_num': self.config['dataset_augmentation_perspective_enlarge_num'],
                },
                flip_param={
                    'horizontal_flip': self.config['dataset_augmentation_flip_horizontal_flip'],
                    'time_flip': self.config['dataset_augmentation_flip_time_flip'],
                },
                jitter_param={
                    'brightness': self.config['dataset_augmentation_jitter_brightness'],
                    'contrast': self.config['dataset_augmentation_jitter_contrast'],
                    'saturation': self.config['dataset_augmentation_jitter_saturation'],
                    'hue': self.config['dataset_augmentation_jitter_hue'],
                }
            )
        else:
            self.transform = None
        
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        if self.type == 'Vox':
            return self._getitem_vox(idx)
        elif self.type == 'LRW':
            return self._getitem_lrw(idx)
        elif self.type == 'MEAD':
            return self._getitem_mead(idx)
        
    def _getitem_vox(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx].split('.')[0]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx].split('.')[0]

            audio_path = os.path.join(self.audio_dir, name+'.npy')
            pose_path = os.path.join(self.pose_dir,name+'.npy')
            path = os.path.join(self.image_dir, name)

        video_name = os.path.basename(path)
        if  os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
            mfcc = np.load(audio_path)
            pose = np.load(pose_path)

            try:
                len(mfcc) > 16
            except:
                print('wrongmfcc len:',audio_path)
            if 16 < len(mfcc) < 24 :
                r = 0
            else:
                r = random.choice([x for x in range(3, len(mfcc)-20)])

            mfccs = []
            poses = []
            video_array = []
            for ind in range(1, 17):
                t_mfcc = mfcc[r+ind][:, 1:]
                mfccs.append(t_mfcc)
                t_pose = pose[r+ind,:-1]
                poses.append(t_pose)
                image = img_as_float32(io.imread(os.path.join(path, str(r + ind)+'.png')))
                video_array.append(image)
            mfccs = np.array(mfccs)
            poses = np.array(poses)
            video_array = np.array(video_array)
            example_image = img_as_float32(io.imread(os.path.join(path, str(r)+'.png')))
        else:
            print('Wrong, data path not an existing file.')

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        driving = np.array(video_array, dtype='float32')
        spatial_size = np.array(driving.shape[1:3][::-1])[np.newaxis]
        driving_pose = np.array(poses, dtype='float32')
        example_image = np.array(example_image, dtype='float32')
        out['example_image'] = example_image.transpose((2, 0, 1))
        out['driving_pose'] = driving_pose
        out['driving'] = driving.transpose((0, 3, 1, 2))
        out['driving_audio'] = np.array(mfccs, dtype='float32')
        return out
    
    def _getitem_lrw(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx].split('.')[0]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx].split('.')[0]
            audio_path = os.path.join(self.audio_dir, name)
            pose_path = os.path.join(self.pose_dir,name)
            path = os.path.join(self.image_dir, name)
        video_name = os.path.basename(path)
        if  os.path.isdir(path):
            # mfcc loading
            r = random.choice([x for x in range(3, 8)])
            example_image = img_as_float32(io.imread(os.path.join(path, str(r)+'.png')))
            mfccs = []
            for ind in range(1, 17):
              #  t_mfcc = mfcc[(r + ind - 3) * 4: (r + ind + 4) * 4, 1:]
                t_mfcc = np.load(os.path.join(audio_path,str(r + ind)+'.npy'),allow_pickle=True)[:, 1:]
                mfccs.append(t_mfcc)
            mfccs = np.array(mfccs)
            poses = []
            video_array = []
            for ind in range(1, 17):
                t_pose = np.load(os.path.join(self.pose_dir,name+'.npy'))[r+ind,:-1]
                poses.append(t_pose)
                image = img_as_float32(io.imread(os.path.join(path, str(r + ind)+'.png')))
                video_array.append(image)
            poses = np.array(poses)
            video_array = np.array(video_array)
        else:
            print('Wrong, data path not an existing file.')
        if self.transform is not None:
            video_array = self.transform(video_array)
        out = {}
        driving = np.array(video_array, dtype='float32')
        spatial_size = np.array(driving.shape[1:3][::-1])[np.newaxis]
        driving_pose = np.array(poses, dtype='float32')
        example_image = np.array(example_image, dtype='float32')
        out['example_image'] = example_image.transpose((2, 0, 1))
        out['driving_pose'] = driving_pose
        out['driving'] = driving.transpose((0, 3, 1, 2))
        out['driving_audio'] = np.array(mfccs, dtype='float32')
        return out
    
    def _getitem_mead(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.image_dir, name)
            video_name = os.path.basename(path)
            id_name = path.split('/')[-2]
            neu_list = self.dict[id_name]
            neu_path = os.path.join(self.image_dir, np.random.choice(neu_list))
            audio_path = os.path.join(self.audio_dir, name+'.npy')
            pose_path = os.path.join(self.pose_dir,name+'.npy')
        if self.is_train and os.path.isdir(path):
            mfcc = np.load(audio_path)
            pose_raw = np.load(pose_path)
            one_euro_filter = OneEuroFilter(mincutoff=0.01, beta=0.7, dcutoff=1.0, freq=100)
            pose = np.zeros((len(pose_raw),7))
            for j in range(len(pose_raw)):
                pose[j]=one_euro_filter.process(pose_raw[j])
            neu_frames = os.listdir(neu_path)
            num_neu_frames = len(neu_frames)
            frame_idx = np.random.choice(num_neu_frames)
            example_image = img_as_float32(io.imread(os.path.join(neu_path, neu_frames[frame_idx])))
            try:
                len(mfcc) > 16
            except:
                print('wrongmfcc len:',audio_path)
            if 16 < len(mfcc) < 24 :
                r = 0
            else:
                r = random.choice([x for x in range(3, len(mfcc)-20)])
            mfccs = []
            poses = []
            video_array = []
            for ind in range(1, 17):
                t_mfcc = mfcc[r+ind][:, 1:]
                mfccs.append(t_mfcc)
                t_pose = pose[r+ind,:-1]
                poses.append(t_pose)
                image = img_as_float32(io.imread(os.path.join(path, str(r + ind)+'.png')))
                video_array.append(image)
            mfccs = np.array(mfccs)
            poses = np.array(poses)
            video_array = np.array(video_array)
        else:
            print('Wrong, data path not an existing file.')
        if self.transform is not None:
            video_array = self.transform(video_array)
        out = {}
        if self.is_train:
            driving = np.array(video_array, dtype='float32')
            driving_pose = np.array(poses, dtype='float32')
            example_image = np.array(example_image, dtype='float32')
            out['example_image'] = example_image.transpose((2, 0, 1))
            out['driving_pose'] = driving_pose
            out['driving'] = driving.transpose((0, 3, 1, 2))
            out['driving_audio'] = np.array(mfccs, dtype='float32')
        return out

