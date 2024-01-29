import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_
from tqdm import tqdm
from os import listdir, path
import numpy as np
import os, subprocess
from glob import glob
import cv2
from talkingface.model.layers import Conv2d, Conv2dTranspose, nonorm_Conv2d
from talkingface.model.abstract_talkingface import AbstractTalkingFace
from talkingface.data.dataprocess.wav2lip_process import Wav2LipPreprocessForInference, Wav2LipAudio
from talkingface.utils import ensure_dir
from talkingface.model.audio_driven_talkingface.eamm_modules.generator import OcclusionAwareGenerator
from talkingface.model.audio_driven_talkingface.eamm_modules.discriminator import MultiScaleDiscriminator
from talkingface.model.audio_driven_talkingface.eamm_modules.keypoint_detector import KPDetector, Audio_Feature, KPDetector_a
from talkingface.model.audio_driven_talkingface.eamm_modules.util import AT_net, Emotion_k

class EAMM(AbstractTalkingFace):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print("init EAMM module")
        self._build_model()
    
    def forward(self):
        pass
    
    def predict(self):
        pass

    def calculate_loss(self, interaction, valid=False):
        pass

    def generate_batch(self):
        pass

    def _build_model(self):
        self.generator = OcclusionAwareGenerator(
            num_kp=self.config['model_common_num_kp'],
            num_channels=self.config['model_common_num_channels'],
            estimate_jacobian=self.config['model_common_estimate_jacobian'],
            block_expansion=self.config['model_generator_block_expansion'],
            max_features=self.config['model_generator_max_features'],
            num_down_blocks=self.config['model_generator_num_down_blocks'],
            num_bottleneck_blocks=self.config['model_generator_num_bottleneck_blocks'],
            estimate_occlusion_map=self.config['model_generator_estimate_occlusion_map'],
            dense_motion_params={
                'block_expansion': self.config['model_generator_dense_motion_block_expansion'],
                'max_features': self.config['model_generator_dense_motion_max_features'],
                'num_blocks': self.config['model_generator_dense_motion_num_blocks'],
                'scale_factor': self.config['model_generator_dense_motion_scale_factor'],
            },
        )
        
        if self.config['use_gpu'] and torch.cuda.is_available():
            self.generator.to(self.config['device_ids'][0])
        else:
            self.generator.cpu()

        if self.config['verbose']:
            print(self.generator)

        self.discriminator = MultiScaleDiscriminator(
            num_kp=self.config['model_common_num_kp'],
            num_channels=self.config['model_common_num_channels'],
            estimate_jacobian=self.config['model_common_estimate_jacobian'],
            scales=self.config['model_discriminator_scales'],
            block_expansion=self.config['model_discriminator_block_expansion'],
            max_features=self.config['model_discriminator_max_features'],
            num_blocks=self.config['model_discriminator_num_blocks'],
            sn=self.config['model_discriminator_sn'],
        )
        if self.config['use_gpu'] and torch.cuda.is_available():
            self.discriminator.to(self.config['device_ids'][0])
        else:
            self.discriminator.cpu()

        if self.config['verbose']:
            print(self.discriminator)

        self.kp_detector = KPDetector(
            num_kp=self.config['model_common_num_kp'],
            num_channels=self.config['model_common_num_channels'],
            estimate_jacobian=self.config['model_common_estimate_jacobian'],
            temperature=self.config['model_kp_detector_temperature'],
            block_expansion=self.config['model_kp_detector_block_expansion'],
            max_features=self.config['model_kp_detector_max_features'],
            scale_factor=self.config['model_kp_detector_scale_factor'],
            num_blocks=self.config['model_kp_detector_num_blocks'],
        )

        self.kp_detector_a = KPDetector_a(
            num_kp=self.config['model_audio_num_kp'],
            num_channels=self.config['model_audio_num_channels'],
            num_channels_a=self.config['model_audio_num_channels_a'],
            estimate_jacobian=self.config['model_common_estimate_jacobian'],
            temperature=self.config['model_kp_detector_temperature'],
            block_expansion=self.config['model_kp_detector_block_expansion'],
            max_features=self.config['model_kp_detector_max_features'],
            scale_factor=self.config['model_kp_detector_scale_factor'],
            num_blocks=self.config['model_kp_detector_num_blocks'],
        )

        if self.config['use_gpu'] and torch.cuda.is_available():
            self.kp_detector.to(self.config['device_ids'][0])
            self.kp_detector_a.to(self.config['device_ids'][0])
        else:
            self.kp_detector.cpu()
            self.kp_detector_a.cpu()

        self.audio_feature = AT_net()
        self.emo_feature = Emotion_k(
            block_expansion=32,
            num_channels=3,
            max_features=1024,
            num_blocks=5,
            scale_factor=0.25,
            num_classes=8,
        )

        if self.config['use_gpu'] and torch.cuda.is_available():
            self.audio_feature.to(self.config['device_ids'][0])
            self.emo_feature.to(self.config['device_ids'][0])
        else:
            self.audio_feature.cpu()
            self.emo_feature.cpu()

        if self.config['verbose']:
            print(self.kp_detector)
            print(self.kp_detector_a)
            print(self.audio_feature)
            print(self.emo_feature)
    
