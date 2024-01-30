import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn import init
import functools
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from collections import OrderedDict
from torch.cuda.amp import autocast as autocast
from abc import ABC, abstractmethod
from talkingface.data.dataset.dataset import Dataset
import librosa
import scipy.io as sio
import bisect
from talkingface.model.audio_driven_talkingface.LiveSpeechPortraits.networks import APC_encoder
# from funcs import utils
from pathlib import Path
from skimage.io import imread, imsave
from PIL import Image
import io
import cv2
import h5py
import albumentations as A

class BaseDataset(Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

        To create a subclass, you need to implement the following four functions:
        -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
        -- <__len__>:                       return the size of dataset.
        -- <__getitem__>:                   get a data point.
        -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
        """

    def __init__(self, opt, datasplit):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataset_params['root']

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


class AudioVisualDataset(BaseDataset):
    """ audio-visual dataset. currently, return 2D info and 3D tracking info.

        # for wavenet:
        #           |----receptive_field----|
        #                                 |--output_length--|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                           | | | | | | | | | |

    """

    def __init__(self, opt, datasplit):
        # save the option and dataset root
        BaseDataset.__init__(self, opt, datasplit)
        self.isTrain = self.opt['Train']

        return
        self.state = opt.dataset_type
        self.dataset_name = opt.dataset_names
        self.target_length = opt.time_frame_length
        self.sample_rate = opt.sample_rate
        self.fps = opt.FPS

        self.audioRF_history = opt.audioRF_history
        self.audioRF_future = opt.audioRF_future
        self.compute_mel_online = opt.compute_mel_online
        self.feature_name = opt.feature_name

        self.audio_samples_one_frame = self.sample_rate / self.fps
        self.frame_jump_stride = opt.frame_jump_stride
        self.augment = False
        self.task = opt.task
        self.item_length_audio = int((self.audioRF_history + self.audioRF_future) / self.fps * self.sample_rate)

        if self.task == 'Audio2Feature':
            if opt.feature_decoder == 'WaveNet':
                self.A2L_receptive_field = opt.A2L_receptive_field
                self.A2L_item_length = self.A2L_receptive_field + self.target_length - 1
            elif opt.feature_decoder == 'LSTM':
                self.A2L_receptive_field = 30
                self.A2L_item_length = self.A2L_receptive_field + self.target_length - 1
        elif self.task == 'Audio2Headpose':
            self.A2H_receptive_field = opt.A2H_receptive_field
            self.A2H_item_length = self.A2H_receptive_field + self.target_length - 1
            self.audio_window = opt.audio_windows
            self.half_audio_win = int(self.audio_window / 2)

        self.frame_future = opt.frame_future
        self.predict_length = opt.predict_length
        self.predict_len = int((self.predict_length - 1) / 2)

        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        print('self.device:', self.device)
        if self.task == 'Audio2Feature':
            self.seq_len = opt.sequence_length

        self.total_len = 0
        self.dataset_root = os.path.join(self.root, self.dataset_name)
        if self.state == 'Train':
            self.clip_names = opt.train_dataset_names
        elif self.state == 'Val':
            self.clip_names = opt.validate_dataset_names
        elif self.state == 'Test':
            self.clip_names = opt.test_dataset_names

        self.clip_nums = len(self.clip_names)
        # main info
        self.audio = [''] * self.clip_nums
        self.audio_features = [''] * self.clip_nums
        self.feats = [''] * self.clip_nums
        self.exps = [''] * self.clip_nums
        self.pts3d = [''] * self.clip_nums
        self.rot_angles = [''] * self.clip_nums
        self.trans = [''] * self.clip_nums
        self.headposes = [''] * self.clip_nums
        self.velocity_pose = [''] * self.clip_nums
        self.acceleration_pose = [''] * self.clip_nums
        self.mean_trans = [''] * self.clip_nums
        if self.state == 'Test':
            self.landmarks = [''] * self.clip_nums
        # meta info
        self.start_point = [''] * self.clip_nums
        self.end_point = [''] * self.clip_nums
        self.len = [''] * self.clip_nums
        self.sample_start = []
        self.clip_valid = ['True'] * self.clip_nums
        self.invalid_clip = []

        self.mouth_related_indices = np.concatenate([np.arange(4, 11), np.arange(46, 64)])
        if self.task == 'Audio2Feature':
            if self.opt.only_mouth:
                self.indices = self.mouth_related_indices
            else:
                self.indices = np.arange(73)
        if opt.use_delta_pts:
            self.pts3d_mean = np.load(os.path.join(self.dataset_root, 'mean_pts3d.npy'))

        for i in range(self.clip_nums):
            name = self.clip_names[i]
            clip_root = os.path.join(self.dataset_root, name)
            # audio
            if os.path.exists(os.path.join(clip_root, name + '_denoise.wav')):
                audio_path = os.path.join(clip_root, name + '_denoise.wav')
                print('find denoised wav!')
            else:
                audio_path = os.path.join(clip_root, name + '.wav')
            self.audio[i], _ = librosa.load(audio_path, sr=self.sample_rate)

            if self.opt.audio_encoder == 'APC':
                APC_name = os.path.split(self.opt.APC_model_path)[-1]
                APC_feature_file = name + '_APC_feature_V0324_ckpt_{}.npy'.format(APC_name)
                APC_feature_path = os.path.join(clip_root, APC_feature_file)
                need_deepfeats = False if os.path.exists(APC_feature_path) else True
                if not need_deepfeats:
                    self.audio_features[i] = np.load(APC_feature_path).astype(np.float32)
            else:
                need_deepfeats = False

            # 3D landmarks & headposes
            if self.task == 'Audio2Feature':
                self.start_point[i] = 0
            elif self.task == 'Audio2Headpose':
                self.start_point[i] = 300
            fit_data_path = os.path.join(clip_root, '3d_fit_data.npz')
            fit_data = np.load(fit_data_path)
            if not opt.ispts_norm:
                ori_pts3d = fit_data['pts_3d'].astype(np.float32)
            else:
                ori_pts3d = np.load(os.path.join(clip_root, 'tracked3D_normalized_pts_fix_contour.npy'))
            if opt.use_delta_pts:
                self.pts3d[i] = ori_pts3d - self.pts3d_mean
            else:
                self.pts3d[i] = ori_pts3d
            if opt.feature_dtype == 'pts3d':
                self.feats[i] = self.pts3d[i]
            elif opt.feature_dtype == 'FW':
                track_data_path = os.path.join(clip_root, 'tracking_results.mat')
                self.feats[i] = sio.loadmat(track_data_path)['exps'].astype(np.float32)
            self.rot_angles[i] = fit_data['rot_angles'].astype(np.float32)
            # change -180~180 to 0~360
            if not self.dataset_name == 'Yuxuan':
                rot_change = self.rot_angles[i][:, 0] < 0
                self.rot_angles[i][rot_change, 0] += 360
                self.rot_angles[i][:, 0] -= 180  # change x axis direction
            # use delta translation
            self.mean_trans[i] = fit_data['trans'][:, :, 0].astype(np.float32).mean(axis=0)
            self.trans[i] = fit_data['trans'][:, :, 0].astype(np.float32) - self.mean_trans[i]

            self.headposes[i] = np.concatenate([self.rot_angles[i], self.trans[i]], axis=1)
            self.velocity_pose[i] = np.concatenate(
                [np.zeros(6)[None, :], self.headposes[i][1:] - self.headposes[i][:-1]])
            self.acceleration_pose[i] = np.concatenate(
                [np.zeros(6)[None, :], self.velocity_pose[i][1:] - self.velocity_pose[i][:-1]])

            if self.dataset_name == 'Yuxuan':
                total_frames = self.feats[i].shape[0] - 300 - 130
            else:
                total_frames = self.feats[i].shape[0] - 60

            if need_deepfeats:
                if self.opt.audio_encoder == 'APC':
                    print('dataset {} need to pre-compute APC features ...'.format(name))
                    print('first we compute mel spectram for dataset {} '.format(name))
                    mel80 = utils.compute_mel_one_sequence(self.audio[i])
                    mel_nframe = mel80.shape[0]
                    print('loading pre-trained model: ', self.opt.APC_model_path)
                    APC_model = APC_encoder(self.opt.audiofeature_input_channels,
                                            self.opt.APC_hidden_size,
                                            self.opt.APC_rnn_layers,
                                            self.opt.APC_residual)
                    APC_model.load_state_dict(torch.load(self.opt.APC_model_path, map_location=str(self.device)),
                                              strict=False)
                    #                    APC_model.load_state_dict(torch.load(self.opt.APC_model_path), strict=False)
                    APC_model.cuda()
                    APC_model.eval()
                    with torch.no_grad():
                        length = torch.Tensor([mel_nframe])
                        #                        hidden_reps = torch.zeros([mel_nframe, self.opt.APC_hidden_size]).cuda()
                        mel80_torch = torch.from_numpy(mel80.astype(np.float32)).cuda().unsqueeze(0)
                        hidden_reps = APC_model.forward(mel80_torch, length)[0]  # [mel_nframe, 512]
                        hidden_reps = hidden_reps.cpu().numpy()
                        np.save(APC_feature_path, hidden_reps)
                        self.audio_features[i] = hidden_reps

            valid_frames = total_frames - self.start_point[i]
            self.len[i] = valid_frames - 400
            if i == 0:
                self.sample_start.append(0)
            else:
                self.sample_start.append(self.sample_start[-1] + self.len[i - 1] - 1)
            self.total_len += np.int32(np.floor(self.len[i] / self.frame_jump_stride))

    def __getitem__(self, index):
        # recover real index from compressed one
        index_real = np.int32(index * self.frame_jump_stride)
        # find which audio file and the start frame index
        file_index = bisect.bisect_right(self.sample_start, index_real) - 1
        current_frame = index_real - self.sample_start[file_index] + self.start_point[file_index]
        current_target_length = self.target_length

        if self.task == 'Audio2Feature':
            # start point is current frame
            A2Lsamples = self.audio_features[file_index][current_frame * 2: (current_frame + self.seq_len) * 2]
            target_pts3d = self.feats[file_index][current_frame: current_frame + self.seq_len, self.indices].reshape(
                self.seq_len, -1)

            A2Lsamples = torch.from_numpy(A2Lsamples).float()
            target_pts3d = torch.from_numpy(target_pts3d).float()

            # [item_length, mel_channels, mel_width], or [item_length, APC_hidden_size]
            return A2Lsamples, target_pts3d


        elif self.task == 'Audio2Headpose':
            if self.opt.feature_decoder == 'WaveNet':
                # find the history info start points
                A2H_history_start = current_frame - self.A2H_receptive_field
                A2H_item_length = self.A2H_item_length
                A2H_receptive_field = self.A2H_receptive_field

                if self.half_audio_win == 1:
                    A2Hsamples = self.audio_features[file_index][2 * (A2H_history_start + self.frame_future): 2 * (
                                A2H_history_start + self.frame_future + A2H_item_length)]
                else:
                    A2Hsamples = np.zeros([A2H_item_length, self.audio_window, 512])
                    for i in range(A2H_item_length):
                        A2Hsamples[i] = self.audio_features[file_index][
                                        2 * (A2H_history_start + i) - self.half_audio_win: 2 * (
                                                    A2H_history_start + i) + self.half_audio_win]

                if self.predict_len == 0:
                    target_headpose = self.headposes[file_index][
                                      A2H_history_start + A2H_receptive_field: A2H_history_start + A2H_item_length + 1]
                    history_headpose = self.headposes[file_index][
                                       A2H_history_start: A2H_history_start + A2H_item_length].reshape(A2H_item_length,
                                                                                                       -1)

                    target_velocity = self.velocity_pose[file_index][
                                      A2H_history_start + A2H_receptive_field: A2H_history_start + A2H_item_length + 1]
                    history_velocity = self.velocity_pose[file_index][
                                       A2H_history_start: A2H_history_start + A2H_item_length].reshape(A2H_item_length,
                                                                                                       -1)
                    target_info = torch.from_numpy(
                        np.concatenate([target_headpose, target_velocity], axis=1).reshape(current_target_length,
                                                                                           -1)).float()
                else:
                    history_headpose = self.headposes[file_index][
                                       A2H_history_start: A2H_history_start + A2H_item_length].reshape(A2H_item_length,
                                                                                                       -1)
                    history_velocity = self.velocity_pose[file_index][
                                       A2H_history_start: A2H_history_start + A2H_item_length].reshape(A2H_item_length,
                                                                                                       -1)

                    target_headpose_ = self.headposes[file_index][
                                       A2H_history_start + A2H_receptive_field - self.predict_len: A2H_history_start + A2H_item_length + 1 + self.predict_len + 1]
                    target_headpose = np.zeros([current_target_length, self.predict_length, target_headpose_.shape[1]])
                    for i in range(current_target_length):
                        target_headpose[i] = target_headpose_[i: i + self.predict_length]
                    target_headpose = target_headpose  # .reshape(current_target_length, -1, order='F')

                    target_velocity_ = self.headposes[file_index][
                                       A2H_history_start + A2H_receptive_field - self.predict_len: A2H_history_start + A2H_item_length + 1 + self.predict_len + 1]
                    target_velocity = np.zeros([current_target_length, self.predict_length, target_velocity_.shape[1]])
                    for i in range(current_target_length):
                        target_velocity[i] = target_velocity_[i: i + self.predict_length]
                    target_velocity = target_velocity  # .reshape(current_target_length, -1, order='F')

                    target_info = torch.from_numpy(
                        np.concatenate([target_headpose, target_velocity], axis=2).reshape(current_target_length,
                                                                                           -1)).float()

                A2Hsamples = torch.from_numpy(A2Hsamples).float()

                history_info = torch.from_numpy(np.concatenate([history_headpose, history_velocity], axis=1)).float()

                # [item_length, mel_channels, mel_width], or [item_length, APC_hidden_size]
                return A2Hsamples, history_info, target_info


            elif self.opt.feature_decoder == 'LSTM':
                A2Hsamples = self.audio_features[file_index][
                             current_frame * 2: (current_frame + self.opt.A2H_receptive_field) * 2]

                target_headpose = self.headposes[file_index][
                                  current_frame: current_frame + self.opt.A2H_receptive_field]
                target_velocity = self.velocity_pose[file_index][
                                  current_frame: current_frame + self.opt.A2H_receptive_field]
                target_info = torch.from_numpy(
                    np.concatenate([target_headpose, target_velocity], axis=1).reshape(self.opt.A2H_receptive_field,
                                                                                       -1)).float()

                A2Hsamples = torch.from_numpy(A2Hsamples).float()

                # [item_length, mel_channels, mel_width], or [item_length, APC_hidden_size]
                return A2Hsamples, target_info

    def __len__(self):
        return self.total_len


class FaceDataset(BaseDataset):
    def __init__(self, opt, datasplit):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        BaseDataset.__init__(self, opt, datasplit)


        self.state = 'Train' if self.opt.train else 'Test'
        return
        self.dataset_name = opt.dataset_names[0]
        # default settings
        # currently, we have 8 parts for face parts
        self.part_list = [[list(range(0, 15))],  # contour
                          [[15, 16, 17, 18, 18, 19, 20, 15]],  # right eyebrow
                          [[21, 22, 23, 24, 24, 25, 26, 21]],  # left eyebrow
                          [range(35, 44)],  # nose
                          [[27, 65, 28, 68, 29], [29, 67, 30, 66, 27]],  # right eye
                          [[33, 69, 32, 72, 31], [31, 71, 34, 70, 33]],  # left eye
                          [range(46, 53), [52, 53, 54, 55, 56, 57, 46]],  # mouth
                          [[46, 63, 62, 61, 52], [52, 60, 59, 58, 46]]  # tongue
                          ]
        self.mouth_outer = [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 46]
        self.label_list = [1, 1, 2, 3, 3, 4, 5]  # labeling for different facial parts

        # only load in train mode

        self.dataset_root = os.path.join(self.root, self.dataset_name)
        if self.state == 'Train':
            self.clip_names = opt.train_dataset_names
        elif self.state == 'Val':
            self.clip_names = opt.validate_dataset_names
        elif self.state == 'Test':
            self.clip_names = opt.test_dataset_names

        self.clip_nums = len(self.clip_names)

        # load pts & image info
        self.landmarks2D, self.len, self.sample_len = [''] * self.clip_nums, [''] * self.clip_nums, [
            ''] * self.clip_nums
        self.image_transforms, self.image_pad, self.tgts_paths = [''] * self.clip_nums, [''] * self.clip_nums, [
            ''] * self.clip_nums
        self.shoulders, self.shoulder3D = [''] * self.clip_nums, [''] * self.clip_nums
        self.sample_start = []

        # tracked 3d info & candidates images
        self.pts3d, self.rot, self.trans = [''] * self.clip_nums, [''] * self.clip_nums, [''] * self.clip_nums
        self.full_cand = [''] * self.clip_nums
        self.headposes = [''] * self.clip_nums

        self.total_len = 0
        if self.opt.isTrain:
            for i in range(self.clip_nums):
                name = self.clip_names[i]
                clip_root = os.path.join(self.dataset_root, name)
                # basic image info
                img_file_path = os.path.join(clip_root, name + '.h5')
                img_file = h5py.File(img_file_path, 'r')[name]
                example = np.asarray(Image.open(io.BytesIO(img_file[0])))
                h, w, _ = example.shape

                landmark_path = os.path.join(clip_root, 'tracked2D_normalized_pts_fix_contour.npy')
                self.landmarks2D[i] = np.load(landmark_path).astype(np.float32)
                change_paras = np.load(os.path.join(clip_root, 'change_paras.npz'))
                scale, xc, yc = change_paras['scale'], change_paras['xc'], change_paras['yc']
                x_min, x_max, y_min, y_max = xc - 256, xc + 256, yc - 256, yc + 256
                # if need padding
                x_min, x_max, y_min, y_max, self.image_pad[i] = max(x_min, 0), min(x_max, w), max(y_min, 0), min(y_max,
                                                                                                                 h), None

                if x_min == 0 or x_max == 512 or y_min == 0 or y_max == 512:
                    top, bottom, left, right = abs(yc - 256 - y_min), abs(yc + 256 - y_max), abs(xc - 256 - x_min), abs(
                        xc + 256 - x_max)
                    self.image_pad[i] = [top, bottom, left, right]
                self.image_transforms[i] = A.Compose([
                    A.Resize(np.int32(h * scale), np.int32(w * scale)),
                    A.Crop(x_min, y_min, x_max, y_max)])

                if self.opt.isH5:
                    tgt_file_path = os.path.join(clip_root, name + '.h5')
                    tgt_file = h5py.File(tgt_file_path, 'r')[name]
                    image_length = len(tgt_file)
                else:
                    tgt_paths = list(map(lambda x: str(x), sorted(list(Path(clip_root).glob('*' + self.opt.suffix)),
                                                                  key=lambda x: int(x.stem))))
                    image_length = len(tgt_paths)
                    self.tgts_paths[i] = tgt_paths
                if not self.landmarks2D[i].shape[0] == image_length:
                    raise ValueError('In dataset {} length of landmarks and images are not equal!'.format(name))

                # tracked 3d info
                fit_data_path = os.path.join(clip_root, '3d_fit_data.npz')
                fit_data = np.load(fit_data_path)
                self.pts3d[i] = fit_data['pts_3d'].astype(np.float32)
                self.rot[i] = fit_data['rot_angles'].astype(np.float32)
                self.trans[i] = fit_data['trans'][:, :, 0].astype(np.float32)
                if not self.pts3d[i].shape[0] == image_length:
                    raise ValueError('In dataset {} length of 3d pts and images are not equal!'.format(name))

                    # candidates images

                tmp = []
                for j in range(4):
                    try:
                        output = imread(os.path.join(clip_root, 'candidates', f'normalized_full_{j}.jpg'))
                    except:
                        imgc = imread(os.path.join(clip_root, 'candidates', f'full_{j}.jpg'))
                        output = self.common_dataset_transform(imgc, i)
                        imsave(os.path.join(clip_root, 'candidates', f'normalized_full_{j}.jpg'), output)
                    output = A.pytorch.transforms.ToTensor(normalize={'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)})(
                        image=output)['image']
                    tmp.append(output)
                self.full_cand[i] = torch.cat(tmp)

                # headpose
                fit_data_path = os.path.join(clip_root, '3d_fit_data.npz')
                fit_data = np.load(fit_data_path)
                rot_angles = fit_data['rot_angles'].astype(np.float32)
                # change -180~180 to 0~360
                if not self.dataset_name == 'Yuxuan':
                    rot_change = rot_angles[:, 0] < 0
                    rot_angles[rot_change, 0] += 360
                    rot_angles[:, 0] -= 180  # change x axis direction
                # use delta translation
                mean_trans = fit_data['trans'][:, :, 0].astype(np.float32).mean(axis=0)
                trans = fit_data['trans'][:, :, 0].astype(np.float32) - mean_trans

                self.headposes[i] = np.concatenate([rot_angles, trans], axis=1)

                # shoulders
                shoulder_path = os.path.join(clip_root, 'normalized_shoulder_points.npy')
                self.shoulders[i] = np.load(shoulder_path)
                shoulder3D_path = os.path.join(clip_root, 'shoulder_points3D.npy')
                self.shoulder3D[i] = np.load(shoulder3D_path)

                self.sample_len[i] = np.int32(np.floor((self.landmarks2D[i].shape[0] - 60) / self.opt.frame_jump) + 1)
                self.len[i] = self.landmarks2D[i].shape[0]
                if i == 0:
                    self.sample_start.append(0)
                else:
                    self.sample_start.append(self.sample_start[-1] + self.sample_len[i - 1])  # not minus 1
                self.total_len += self.sample_len[i]

        # test mode
        else:
            # if need padding
            example = imread(os.path.join(self.root, 'example.png'))
            h, w, _ = example.shape
            change_paras = np.load(os.path.join(self.root, 'change_paras.npz'))
            scale, xc, yc = change_paras['scale'], change_paras['xc'], change_paras['yc']
            x_min, x_max, y_min, y_max = xc - 256, xc + 256, yc - 256, yc + 256
            x_min, x_max, y_min, y_max, self.image_pad = max(x_min, 0), min(x_max, w), max(y_min, 0), min(y_max,
                                                                                                          h), None

            if x_min == 0 or x_max == 512 or y_min == 0 or y_max == 512:
                top, bottom, left, right = abs(yc - 256 - y_min), abs(yc + 256 - y_max), abs(xc - 256 - x_min), abs(
                    xc + 256 - x_max)
                self.image_pad = [top, bottom, left, right]

    def __getitem__(self, ind):
        dataset_index = bisect.bisect_right(self.sample_start, ind) - 1
        data_index = (ind - self.sample_start[dataset_index]) * self.opt.frame_jump + np.random.randint(
            self.opt.frame_jump)

        target_ind = data_index + 1  # history_ind, current_ind
        landmarks = self.landmarks2D[dataset_index][target_ind]  # [73, 2]
        shoulders = self.shoulders[dataset_index][target_ind].copy()

        dataset_name = self.clip_names[dataset_index]
        clip_root = os.path.join(self.dataset_root, dataset_name)
        if self.opt.isH5:
            tgt_file_path = os.path.join(clip_root, dataset_name + '.h5')
            tgt_file = h5py.File(tgt_file_path, 'r')[dataset_name]
            tgt_image = np.asarray(Image.open(io.BytesIO(tgt_file[target_ind])))

            # do transform
            tgt_image = self.common_dataset_transform(tgt_image, dataset_index, None)
        else:
            pass

        h, w, _ = tgt_image.shape

        ### transformations & online data augmentations on images and landmarks
        self.get_crop_coords(landmarks, (w, h), dataset_name,
                             random_trans_scale=0)  # 30.5 µs ± 348 ns  random translation

        transform_tgt = self.get_transform(dataset_name, True, n_img=1, n_keypoint=1, flip=False)
        transformed_tgt = transform_tgt(image=tgt_image, keypoints=landmarks)

        tgt_image, points = transformed_tgt['image'], np.array(transformed_tgt['keypoints']).astype(np.float32)

        feature_map = self.get_feature_image(points, (self.opt.loadSize, self.opt.loadSize), shoulders,
                                             self.image_pad[dataset_index])[np.newaxis, :].astype(np.float32) / 255.
        feature_map = torch.from_numpy(feature_map)

        ## facial weight mask
        weight_mask = self.generate_facial_weight_mask(points, h, w)[None, :]

        cand_image = self.full_cand[dataset_index]

        return_list = {'feature_map': feature_map, 'cand_image': cand_image, 'tgt_image': tgt_image,
                       'weight_mask': weight_mask}

        return return_list

    def common_dataset_transform(self, input, i):
        output = self.image_transforms[i](image=input)['image']
        if self.image_pad[i] is not None:
            top, bottom, left, right = self.image_pad[i]
            output = cv2.copyMakeBorder(output, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return output

    def generate_facial_weight_mask(self, points, h=512, w=512):
        mouth_mask = np.zeros([512, 512, 1])
        points = points[self.mouth_outer]
        points = np.int32(points)
        mouth_mask = cv2.fillPoly(mouth_mask, [points], (255, 0, 0))
        #        plt.imshow(mouth_mask[:,:,0])
        mouth_mask = cv2.dilate(mouth_mask, np.ones((45, 45))) / 255

        return mouth_mask.astype(np.float32)

    def get_transform(self, dataset_name, keypoints=False, n_img=1, n_keypoint=1, normalize=True, flip=False):
        min_x = getattr(self, 'min_x_' + str(dataset_name))
        max_x = getattr(self, 'max_x_' + str(dataset_name))
        min_y = getattr(self, 'min_y_' + str(dataset_name))
        max_y = getattr(self, 'max_y_' + str(dataset_name))

        additional_flag = False
        additional_targets_dict = {}
        if n_img > 1:
            additional_flag = True
            image_str = ['image' + str(i) for i in range(0, n_img)]
            for i in range(n_img):
                additional_targets_dict[image_str[i]] = 'image'
        if n_keypoint > 1:
            additional_flag = True
            keypoint_str = ['keypoint' + str(i) for i in range(0, n_keypoint)]
            for i in range(n_keypoint):
                additional_targets_dict[keypoint_str[i]] = 'keypoints'

        transform = A.Compose([
            A.Crop(x_min=min_x, x_max=max_x, y_min=min_y, y_max=max_y),
            A.Resize(self.opt.loadSize, self.opt.loadSize),
            A.HorizontalFlip(p=flip),
            A.pytorch.transforms.ToTensor(
                normalize={'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)} if normalize == True else None)],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False) if keypoints == True else None,
            additional_targets=additional_targets_dict if additional_flag == True else None
        )
        return transform

    def get_data_test_mode(self, landmarks, shoulder, pad=None):
        ''' get transformed data
        '''

        feature_map = torch.from_numpy(
            self.get_feature_image(landmarks, (self.opt.loadSize, self.opt.loadSize), shoulder, pad)[np.newaxis,
            :].astype(np.float32) / 255.)

        return feature_map

    def get_feature_image(self, landmarks, size, shoulders=None, image_pad=None):
        # draw edges
        im_edges = self.draw_face_feature_maps(landmarks, size)
        if shoulders is not None:
            if image_pad is not None:
                top, bottom, left, right = image_pad
                delta_y = top - bottom
                delta_x = right - left
                shoulders[:, 0] += delta_x
                shoulders[:, 1] += delta_y
            im_edges = self.draw_shoulder_points(im_edges, shoulders)

        return im_edges

    def draw_shoulder_points(self, img, shoulder_points):
        num = int(shoulder_points.shape[0] / 2)
        for i in range(2):
            for j in range(num - 1):
                pt1 = [int(flt) for flt in shoulder_points[i * num + j]]
                pt2 = [int(flt) for flt in shoulder_points[i * num + j + 1]]
                img = cv2.line(img, tuple(pt1), tuple(pt2), 255, 2)  # BGR

        return img

    def draw_face_feature_maps(self, keypoints, size=(512, 512)):
        w, h = size
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8)  # edge map for all edges
        for edge_list in self.part_list:
            for edge in edge_list:
                for i in range(len(edge) - 1):
                    pt1 = [int(flt) for flt in keypoints[edge[i]]]
                    pt2 = [int(flt) for flt in keypoints[edge[i + 1]]]
                    im_edges = cv2.line(im_edges, tuple(pt1), tuple(pt2), 255, 2)

        return im_edges

    def get_crop_coords(self, keypoints, size, dataset_name, random_trans_scale=50):
        # cut a rought region for fine cutting
        # here x towards right and y towards down, origin is left-up corner
        w_ori, h_ori = size
        min_y, max_y = keypoints[:, 1].min(), keypoints[:, 1].max()
        min_x, max_x = keypoints[:, 0].min(), keypoints[:, 0].max()
        xc = (min_x + max_x) // 2
        yc = (min_y * 3 + max_y) // 4
        h = w = min((max_x - min_x) * 2, w_ori, h_ori)

        if self.opt.isTrain:
            # do online augment on landmarks & images
            # 1. random translation: move 10%
            x_bias, y_bias = np.random.uniform(-random_trans_scale, random_trans_scale, size=(2,))
            xc, yc = xc + x_bias, yc + y_bias

        # modify the center x, center y to valid position
        xc = min(max(0, xc - w // 2) + w, w_ori) - w // 2
        yc = min(max(0, yc - h // 2) + h, h_ori) - h // 2

        min_x, max_x = xc - w // 2, xc + w // 2
        min_y, max_y = yc - h // 2, yc + h // 2

        setattr(self, 'min_x_' + str(dataset_name), int(min_x))
        setattr(self, 'max_x_' + str(dataset_name), int(max_x))
        setattr(self, 'min_y_' + str(dataset_name), int(min_y))
        setattr(self, 'max_y_' + str(dataset_name), int(max_y))

    def crop(self, img, dataset_name):
        min_x = getattr(self, 'min_x_' + str(dataset_name))
        max_x = getattr(self, 'max_x_' + str(dataset_name))
        min_y = getattr(self, 'min_y_' + str(dataset_name))
        max_y = getattr(self, 'max_y_' + str(dataset_name))
        if isinstance(img, np.ndarray):
            return img[min_y:max_y, min_x:max_x]
        else:
            return img.crop((min_x, min_y, max_x, max_y))

    def __len__(self):
        if self.opt.train:
            return self.total_len
        else:
            return 1

    def name(self):
        return 'FaceDataset'


class live_speech_portraitsDataset(Dataset):

    def __init__(self, config, datasplit):

        self.opt = config
        self.split = datasplit

    @abstractmethod
    def __getitem__(self, index):

        # recover real index from compressed one
        index_real = np.int32(index * self.frame_jump_stride)
        # find which audio file and the start frame index
        file_index = bisect.bisect_right(self.sample_start, index_real) - 1
        current_frame = index_real - self.sample_start[file_index] + self.start_point[file_index]
        current_target_length = self.target_length

        if self.task == 'Audio2Feature':
            # start point is current frame
            A2Lsamples = self.audio_features[file_index][current_frame * 2: (current_frame + self.seq_len) * 2]
            target_pts3d = self.feats[file_index][current_frame: current_frame + self.seq_len, self.indices].reshape(
                self.seq_len, -1)

            A2Lsamples = torch.from_numpy(A2Lsamples).float()
            target_pts3d = torch.from_numpy(target_pts3d).float()

            # [item_length, mel_channels, mel_width], or [item_length, APC_hidden_size]
            return A2Lsamples, target_pts3d


        elif self.task == 'Audio2Headpose':
            if self.opt.feature_decoder == 'WaveNet':
                # find the history info start points
                A2H_history_start = current_frame - self.A2H_receptive_field
                A2H_item_length = self.A2H_item_length
                A2H_receptive_field = self.A2H_receptive_field

                if self.half_audio_win == 1:
                    A2Hsamples = self.audio_features[file_index][2 * (A2H_history_start + self.frame_future): 2 * (
                                A2H_history_start + self.frame_future + A2H_item_length)]
                else:
                    A2Hsamples = np.zeros([A2H_item_length, self.audio_window, 512])
                    for i in range(A2H_item_length):
                        A2Hsamples[i] = self.audio_features[file_index][
                                        2 * (A2H_history_start + i) - self.half_audio_win: 2 * (
                                                    A2H_history_start + i) + self.half_audio_win]

                if self.predict_len == 0:
                    target_headpose = self.headposes[file_index][
                                      A2H_history_start + A2H_receptive_field: A2H_history_start + A2H_item_length + 1]
                    history_headpose = self.headposes[file_index][
                                       A2H_history_start: A2H_history_start + A2H_item_length].reshape(A2H_item_length,
                                                                                                       -1)

                    target_velocity = self.velocity_pose[file_index][
                                      A2H_history_start + A2H_receptive_field: A2H_history_start + A2H_item_length + 1]
                    history_velocity = self.velocity_pose[file_index][
                                       A2H_history_start: A2H_history_start + A2H_item_length].reshape(A2H_item_length,
                                                                                                       -1)
                    target_info = torch.from_numpy(
                        np.concatenate([target_headpose, target_velocity], axis=1).reshape(current_target_length,
                                                                                           -1)).float()
                else:
                    history_headpose = self.headposes[file_index][
                                       A2H_history_start: A2H_history_start + A2H_item_length].reshape(A2H_item_length,
                                                                                                       -1)
                    history_velocity = self.velocity_pose[file_index][
                                       A2H_history_start: A2H_history_start + A2H_item_length].reshape(A2H_item_length,
                                                                                                       -1)

                    target_headpose_ = self.headposes[file_index][
                                       A2H_history_start + A2H_receptive_field - self.predict_len: A2H_history_start + A2H_item_length + 1 + self.predict_len + 1]
                    target_headpose = np.zeros([current_target_length, self.predict_length, target_headpose_.shape[1]])
                    for i in range(current_target_length):
                        target_headpose[i] = target_headpose_[i: i + self.predict_length]
                    target_headpose = target_headpose  # .reshape(current_target_length, -1, order='F')

                    target_velocity_ = self.headposes[file_index][
                                       A2H_history_start + A2H_receptive_field - self.predict_len: A2H_history_start + A2H_item_length + 1 + self.predict_len + 1]
                    target_velocity = np.zeros([current_target_length, self.predict_length, target_velocity_.shape[1]])
                    for i in range(current_target_length):
                        target_velocity[i] = target_velocity_[i: i + self.predict_length]
                    target_velocity = target_velocity  # .reshape(current_target_length, -1, order='F')

                    target_info = torch.from_numpy(
                        np.concatenate([target_headpose, target_velocity], axis=2).reshape(current_target_length,
                                                                                           -1)).float()

                A2Hsamples = torch.from_numpy(A2Hsamples).float()

                history_info = torch.from_numpy(np.concatenate([history_headpose, history_velocity], axis=1)).float()

                # [item_length, mel_channels, mel_width], or [item_length, APC_hidden_size]
                return A2Hsamples, history_info, target_info


            elif self.opt.feature_decoder == 'LSTM':
                A2Hsamples = self.audio_features[file_index][
                             current_frame * 2: (current_frame + self.opt.A2H_receptive_field) * 2]

                target_headpose = self.headposes[file_index][
                                  current_frame: current_frame + self.opt.A2H_receptive_field]
                target_velocity = self.velocity_pose[file_index][
                                  current_frame: current_frame + self.opt.A2H_receptive_field]
                target_info = torch.from_numpy(
                    np.concatenate([target_headpose, target_velocity], axis=1).reshape(self.opt.A2H_receptive_field,
                                                                                       -1)).float()

                A2Hsamples = torch.from_numpy(A2Hsamples).float()

                # [item_length, mel_channels, mel_width], or [item_length, APC_hidden_size]
                return A2Hsamples, target_info

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.opt.train:
            return self.total_len
        else:
            return 1

    def modify_commandline_options(parser, is_train):
        return parser
