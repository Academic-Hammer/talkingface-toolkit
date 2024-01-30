import os
import math
import numpy as np
from models.config import AudioConfig
import shutil
import cv2
import glob
import random
import torch
import util.util as util
import torch.utils.data as data
class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def to_Tensor(self, img):
        if img.ndim == 3:
            wrapped_img = img.transpose(2, 0, 1) / 255.0
        elif img.ndim == 4:
            wrapped_img = img.transpose(0, 3, 1, 2) / 255.0
        else:
            wrapped_img = img / 255.0
        wrapped_img = torch.from_numpy(wrapped_img).float()

        return wrapped_img * 2 - 1

    def face_augmentation(self, img, crop_size):
        img = self._color_transfer(img)
        img = self._reshape(img, crop_size)
        img = self._blur_and_sharp(img)
        return img

    def _blur_and_sharp(self, img):
        blur = np.random.randint(0, 2)
        img2 = img.copy()
        output = []
        for i in range(len(img2)):
            if blur:
                ksize = np.random.choice([3, 5, 7, 9])
                output.append(cv2.medianBlur(img2[i], ksize))
            else:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                output.append(cv2.filter2D(img2[i], -1, kernel))
        output = np.stack(output)
        return output

    def _color_transfer(self, img):

        transfer_c = np.random.uniform(0.3, 1.6)

        start_channel = np.random.randint(0, 2)
        end_channel = np.random.randint(start_channel + 1, 4)

        img2 = img.copy()

        img2[:, :, :, start_channel:end_channel] = np.minimum(np.maximum(img[:, :, :, start_channel:end_channel] * transfer_c, np.zeros(img[:, :, :, start_channel:end_channel].shape)),
                                 np.ones(img[:, :, :, start_channel:end_channel].shape) * 255)
        return img2

    def perspective_transform(self, img, crop_size=224, pers_size=10, enlarge_size=-10):
        h, w, c = img.shape
        dst = np.array([
          [-enlarge_size, -enlarge_size],
          [-enlarge_size + pers_size, w + enlarge_size],
          [h + enlarge_size, -enlarge_size],
          [h + enlarge_size - pers_size, w + enlarge_size],], dtype=np.float32)
        src = np.array([[-enlarge_size, -enlarge_size], [-enlarge_size, w + enlarge_size],
                        [h + enlarge_size, -enlarge_size], [h + enlarge_size, w + enlarge_size]]).astype(np.float32())
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (crop_size, crop_size), borderMode=cv2.BORDER_REPLICATE)
        return warped, M

    def _reshape(self, img, crop_size):
        reshape = np.random.randint(0, 2)
        reshape_size = np.random.randint(15, 25)
        extra_padding_size = np.random.randint(0, reshape_size // 2)
        pers_size = np.random.randint(20, 30) * pow(-1, np.random.randint(2))

        enlarge_size = np.random.randint(20, 40) * pow(-1, np.random.randint(2))
        shape = img[0].shape
        img2 = img.copy()
        output = []
        for i in range(len(img2)):
            if reshape:
                im = cv2.resize(img2[i], (shape[0] - reshape_size*2, shape[1] + reshape_size*2))
                im = cv2.copyMakeBorder(im, 0, 0, reshape_size + extra_padding_size, reshape_size + extra_padding_size, cv2.cv2.BORDER_REFLECT)
                im = im[reshape_size - extra_padding_size:shape[0] + reshape_size + extra_padding_size, :, :]
                im, _ = self.perspective_transform(im, crop_size=crop_size, pers_size=pers_size, enlarge_size=enlarge_size)
                output.append(im)
            else:
                im = cv2.resize(img2[i], (shape[0] + reshape_size*2, shape[1] - reshape_size*2))
                im = cv2.copyMakeBorder(im, reshape_size + extra_padding_size, reshape_size + extra_padding_size, 0, 0, cv2.cv2.BORDER_REFLECT)
                im = im[:, reshape_size - extra_padding_size:shape[0] + reshape_size + extra_padding_size, :]
                im, _ = self.perspective_transform(im, crop_size=crop_size, pers_size=pers_size, enlarge_size=enlarge_size)
                output.append(im)
        output = np.stack(output)
        return output
class VOXTestDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def cv2_loader(self, img_str):
        img_array = np.frombuffer(img_str, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    def load_img(self, image_path, M=None, crop=True, crop_len=16):
        img = cv2.imread(image_path)

        if img is None:
            raise Exception('None Image')

        if M is not None:
            img = cv2.warpAffine(img, M, (self.opt.crop_size, self.opt.crop_size), borderMode=cv2.BORDER_REPLICATE)

        if crop:
            img = img[:self.opt.crop_size - crop_len*2, crop_len:self.opt.crop_size - crop_len]
            if self.opt.target_crop_len > 0:
                img = img[self.opt.target_crop_len:self.opt.crop_size - self.opt.target_crop_len, self.opt.target_crop_len:self.opt.crop_size - self.opt.target_crop_len]
            img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def fill_list(self, tmp_list):
        length = len(tmp_list)
        if length % self.opt.batchSize != 0:
            end = math.ceil(length / self.opt.batchSize) * self.opt.batchSize
            tmp_list = tmp_list + tmp_list[-1 * (end - length) :]
        return tmp_list

    def frame2audio_indexs(self, frame_inds):
        start_frame_ind = frame_inds - self.audio.num_frames_per_clip // 2

        start_audio_inds = start_frame_ind * self.audio.num_bins_per_frame
        return start_audio_inds

    def __init__(self, opt):
        self.opt = opt
        self.path_label = opt.path_label
        self.clip_len = opt.clip_len
        self.frame_interval = opt.frame_interval
        self.num_clips = opt.num_clips
        self.frame_rate = opt.frame_rate
        self.num_inputs = opt.num_inputs
        self.filename_tmpl = opt.filename_tmpl

        self.mouth_num_frames = None
        self.mouth_frame_path = None
        self.pose_num_frames = None

        self.audio = AudioConfig.AudioConfig(num_frames_per_clip=opt.num_frames_per_clip, hop_size=opt.hop_size)
        self.num_audio_bins = self.audio.num_frames_per_clip * self.audio.num_bins_per_frame


        assert len(opt.path_label.split()) == 8, opt.path_label
        id_path, ref_num, \
        pose_frame_path, pose_num_frames, \
        audio_path, mouth_frame_path, mouth_num_frames, spectrogram_path = opt.path_label.split()


        id_idx, mouth_idx = id_path.split('/')[-1], audio_path.split('/')[-1].split('.')[0]
        if not os.path.isdir(pose_frame_path):
            pose_frame_path = id_path
            pose_num_frames = 1

        pose_idx = pose_frame_path.split('/')[-1]
        id_idx, pose_idx, mouth_idx = str(id_idx), str(pose_idx), str(mouth_idx)

        self.processed_file_savepath = os.path.join('results', 'id_' + id_idx + '_pose_' + pose_idx +
                                   '_audio_' + os.path.basename(audio_path)[:-4])
        if not os.path.exists(self.processed_file_savepath): os.makedirs(self.processed_file_savepath)


        if not os.path.isfile(spectrogram_path):
            wav = self.audio.read_audio(audio_path)
            self.spectrogram = self.audio.audio_to_spectrogram(wav)

        else:
            self.spectrogram = np.load(spectrogram_path)

        if os.path.isdir(mouth_frame_path):
            self.mouth_frame_path = mouth_frame_path
            self.mouth_num_frames = mouth_num_frames

        self.pose_num_frames = int(pose_num_frames)

        self.target_frame_inds = np.arange(2, len(self.spectrogram) // self.audio.num_bins_per_frame - 2)
        self.audio_inds = self.frame2audio_indexs(self.target_frame_inds)

        self.dataset_size = len(self.target_frame_inds)

        id_img_paths = glob.glob(os.path.join(id_path, '*.jpg')) + glob.glob(os.path.join(id_path, '*.png'))
        random.shuffle(id_img_paths)
        opt.num_inputs = min(len(id_img_paths), opt.num_inputs)
        id_img_tensors = []

        for i, image_path in enumerate(id_img_paths):
            id_img_tensor = self.to_Tensor(self.load_img(image_path))
            id_img_tensors += [id_img_tensor]
            shutil.copyfile(image_path, os.path.join(self.processed_file_savepath, 'ref_id_{}.jpg'.format(i)))
            if i == (opt.num_inputs - 1):
                break
        self.id_img_tensor = torch.stack(id_img_tensors)
        self.pose_frame_path = pose_frame_path
        self.audio_path = audio_path
        self.id_path = id_path
        self.mouth_frame_path = mouth_frame_path
        self.initialized = False


    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1)[-10:])[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2)[-10:])[0]
        return filename1_without_ext == filename2_without_ext

    def load_one_frame(self, frame_ind, video_path, M=None, crop=True):
        filepath = os.path.join(video_path, self.filename_tmpl.format(frame_ind))
        img = self.load_img(filepath, M=M, crop=crop)
        img = self.to_Tensor(img)
        return img

    def load_spectrogram(self, audio_ind):
        mel_shape = self.spectrogram.shape

        if (audio_ind + self.num_audio_bins) <= mel_shape[0] and audio_ind >= 0:
            spectrogram = np.array(self.spectrogram[audio_ind:audio_ind + self.num_audio_bins, :]).astype('float32')
        else:
            print('(audio_ind {} + opt.num_audio_bins {}) > mel_shape[0] {} '.format(audio_ind, self.num_audio_bins,
                                                                                     mel_shape[0]))
            if audio_ind > 0:
                spectrogram = np.array(self.spectrogram[audio_ind:audio_ind + self.num_audio_bins, :]).astype('float32')
            else:
                spectrogram = np.zeros((self.num_audio_bins, mel_shape[1])).astype(np.float16).astype(np.float32)

        spectrogram = torch.from_numpy(spectrogram)
        spectrogram = spectrogram.unsqueeze(0)

        spectrogram = spectrogram.transpose(-2, -1)
        return spectrogram

    def __getitem__(self, index):

        img_index = self.target_frame_inds[index]
        mel_index = self.audio_inds[index]

        pose_index = util.calc_loop_idx(img_index, self.pose_num_frames)

        pose_frame = self.load_one_frame(pose_index, self.pose_frame_path)

        if os.path.isdir(self.mouth_frame_path):
            mouth_frame = self.load_one_frame(img_index, self.mouth_frame_path)
        else:
            mouth_frame = torch.zeros_like(pose_frame)

        spectrograms = self.load_spectrogram(mel_index)

        input_dict = {
                      'input': self.id_img_tensor,
                      'target': mouth_frame,
                      'driving_pose_frames': pose_frame,
                      'augmented': pose_frame,
                      'label': torch.zeros(1),
                      }
        if self.opt.use_audio:
            input_dict['spectrograms'] = spectrograms

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_processed_file_savepath(self):
        return self.processed_file_savepath
