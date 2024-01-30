import os
from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from skimage.io import imread
from torch.optim import lr_scheduler
from torch.nn import init
import functools
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import librosa
from tqdm import tqdm
from cog import BasePredictor, Input, Path
from talkingface.utils import set_color
from logging import getLogger
from collections import OrderedDict
from torch.cuda.amp import autocast as autocast
import albumentations
from abc import ABC, abstractmethod
from .LiveSpeechPortraits import networks
from talkingface.model.abstract_talkingface import AbstractTalkingFace
import talkingface.model.audio_driven_talkingface.LiveSpeechPortraits.audio2feature as audio2feature
import talkingface.model.audio_driven_talkingface.LiveSpeechPortraits.audio2headpose as audio2headpose
import talkingface.model.audio_driven_talkingface.LiveSpeechPortraits.feature2face_G as feature2face_G
import talkingface.model.audio_driven_talkingface.LiveSpeechPortraits.feature2face_D as feature2face_D
from talkingface.model.audio_driven_talkingface.LiveSpeechPortraits.losses import GMMLogLoss, Sample_GMM, GANLoss, MaskedL1Loss, VGGLoss
from talkingface.model.audio_driven_talkingface.LiveSpeechPortraits import create_model
from talkingface.data.dataset.LiveSpeechPortraits import create_dataset
from talkingface.utils.live_speech_portraits.options.test_audio2feature_options import TestOptions as FeatureOptions
from talkingface.utils.live_speech_portraits.options.test_audio2headpose_options import TestOptions as HeadposeOptions
from talkingface.utils.live_speech_portraits.options.test_feature2face_options import TestOptions as RenderOptions
from talkingface.utils.live_speech_portraits import utils
import talkingface.utils.live_speech_portraits.util.util as util
from talkingface.utils.live_speech_portraits.util.visualizer import Visualizer
from talkingface.model.audio_driven_talkingface.LiveSpeechPortraits.networks import APC_encoder

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        # get device name: CPU or GPU
        # if self.gpu_ids == '-1':
        #     self.device = torch.device('cpu')
        #     self.gpu_ids = opt.gpu_ids == []
        # else:
        #     self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if len(self.gpu_ids) > 0 else torch.device('cpu')

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        # torch speed up training
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.load_epoch)
        self.print_networks(opt.verbose)

    def train(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train(mode=True)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch, train_info=None):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s.pkl' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)
                torch.save(net.state_dict(), save_path)
        if train_info is not None:
            epoch, epoch_iter = train_info
            iter_path = os.path.join(self.save_dir, 'iter.txt')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        for name in self.model_names:
            if isinstance(name, str):
                if epoch[-3:] == 'pkl':
                    load_path = epoch
                else:
                    load_filename = '%s_%s.pkl' % (epoch, name)
                    load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                #                if isinstance(net, torch.nn.DataParallel):
                #                    net = net.module
                if os.path.exists(load_path):
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if self.device == torch.device('cpu'):
                        for key in list(state_dict.keys()):
                            state_dict[key[7:]] = state_dict.pop(key)
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                    print('loading the model from %s' % load_path)
                    net.load_state_dict(state_dict, strict=False)
                else:
                    print('No model weight file:', load_path, 'initialize model without pre-trained weights.')
                    if self.isTrain == False:
                        raise ValueError(
                            'We are now in inference process, no pre-trained model found! Check the model checkpoint!')

    #                if isinstance(net, torch.nn.DataParallel):
    #                    net = net.module

    # if you are using PyTorch newer than 0.4 (e.g., built from
    # GitHub source), you can remove str() on self.device

    #                state_dict = torch.load(load_path, map_location=str(self.device))
    #                if hasattr(state_dict, '_metadata'):
    #                    del state_dict._metadata
    #
    #                # patch InstanceNorm checkpoints prior to 0.4
    #                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
    #                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    #                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class Audio2HeadposeModel(BaseModel):
    def __init__(self, opt):
        """Initialize the Audio2Headpose class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # define networks
        self.model_names = ['Audio2Headpose']
        if opt.feature_decoder == 'WaveNet':
            self.Audio2Headpose = networks.init_net(audio2headpose.Audio2Headpose(opt), init_type='normal',
                                                    init_gain=0.02, gpu_ids=opt.gpu_ids)
        elif opt.feature_decoder == 'LSTM':
            self.Audio2Headpose = networks.init_net(audio2headpose.Audio2Headpose_LSTM(opt), init_type='normal',
                                                    init_gain=0.02, gpu_ids=opt.gpu_ids)

        # define only during training time
        if self.isTrain:
            # losses
            self.criterion_GMM = GMMLogLoss(opt.A2H_GMM_ncenter, opt.A2H_GMM_ndim, opt.A2H_GMM_sigma_min).to(
                self.device)
            self.criterion_L2 = nn.MSELoss().cuda()
            # optimizer
            self.optimizer = torch.optim.Adam([{'params': self.Audio2Headpose.parameters(),
                                                'initial_lr': opt.lr}], lr=opt.lr, betas=(0.9, 0.99))

            self.optimizers.append(self.optimizer)

            if opt.continue_train:
                self.resume_training()

    def resume_training(self):
        opt = self.opt
        ### if continue training, recover previous states
        print('Resuming from epoch %s ' % (opt.load_epoch))
        # change epoch count & update schedule settings
        opt.epoch_count = int(opt.load_epoch)
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        # print lerning rate
        lr = self.optimizers[0].param_groups[0]['lr']
        print('update learning rate: {} -> {}'.format(opt.lr, lr))

    def set_input(self, data, data_info=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        if self.opt.feature_decoder == 'WaveNet':
            self.headpose_audio_feats, self.history_headpose, self.target_headpose = data
            self.headpose_audio_feats = self.headpose_audio_feats.to(self.device)
            self.history_headpose = self.history_headpose.to(self.device)
            self.target_headpose = self.target_headpose.to(self.device)
        elif self.opt.feature_decoder == 'LSTM':
            self.headpose_audio_feats, self.target_headpose = data
            self.headpose_audio_feats = self.headpose_audio_feats.to(self.device)
            self.target_headpose = self.target_headpose.to(self.device)

    def forward(self):
        '''
        Args:
            history_landmarks: [b, T, ndim]
            audio_features: [b, 1, nfeas, nwins]
        Returns:
            preds: [b, T, output_channels]
        '''

        if self.opt.audio_windows == 2:
            bs, item_len, ndim = self.headpose_audio_feats.shape
            self.headpose_audio_feats = self.headpose_audio_feats.reshape(bs, -1, ndim * 2)
        else:
            bs, item_len, _, ndim = self.headpose_audio_feats.shape
        if self.opt.feature_decoder == 'WaveNet':
            self.preds_headpose = self.Audio2Headpose.forward(self.history_headpose, self.headpose_audio_feats)
        elif self.opt.feature_decoder == 'LSTM':
            self.preds_headpose = self.Audio2Headpose.forward(self.headpose_audio_feats)

    def calculate_loss(self):
        """ calculate loss in detail, only forward pass included"""
        if self.opt.loss == 'GMM':
            self.loss_GMM = self.criterion_GMM(self.preds_headpose, self.target_headpose)
            self.loss = self.loss_GMM
        elif self.opt.loss == 'L2':
            self.loss_L2 = self.criterion_L2(self.preds_headpose, self.target_headpose)
            self.loss = self.loss_L2

        if not self.opt.smooth_loss == 0:
            mu_gen = Sample_GMM(self.preds_headpose,
                                self.Audio2Headpose.module.WaveNet.ncenter,
                                self.Audio2Headpose.module.WaveNet.ndim,
                                sigma_scale=0)
            self.smooth_loss = (mu_gen[:, 2:] + self.target_headpose[:, :-2] - 2 * self.target_headpose[:, 1:-1]).mean(
                dim=2).abs().mean()
            self.loss += self.smooth_loss * self.opt.smooth_loss

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.calculate_loss()
        self.loss.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.optimizer.zero_grad()  # clear optimizer parameters grad
        self.forward()  # forward pass
        self.backward()  # calculate loss and gradients
        self.optimizer.step()  # update gradients

    def validate(self):
        """ validate process """
        with torch.no_grad():
            self.forward()
            self.calculate_loss()

    def generate_sequences(self, audio_feats, pre_headpose, fill_zero=True, sigma_scale=0.0, opt=[]):
        ''' generate landmark sequences given audio and a initialized landmark.
        Note that the audio input should have the same sample rate as the training.
        Args:
            audio_sequences: [n,], in numpy
            init_landmarks: [npts, 2], in numpy
            sample_rate: audio sample rate, should be same as training process.
            method(str): optional, how to generate the sequence, indeed it is the
                loss function during training process. Options are 'L2' or 'GMM'.
        Reutrns:
            landmark_sequences: [T, npts, 2] predition landmark sequences
        '''

        frame_future = opt.frame_future
        audio_feats = audio_feats.reshape(-1, 512 * 2)
        nframe = audio_feats.shape[0] - frame_future
        pred_headpose = np.zeros([nframe, opt.A2H_GMM_ndim])

        if opt.feature_decoder == 'WaveNet':
            # fill zero or not
            if fill_zero == True:
                # headpose
                audio_feats_insert = np.repeat(audio_feats[0], opt.A2H_receptive_field - 1)
                audio_feats_insert = audio_feats_insert.reshape(-1, opt.A2H_receptive_field - 1).T
                audio_feats = np.concatenate([audio_feats_insert, audio_feats])
                # history headpose
                history_headpose = np.repeat(pre_headpose, opt.A2H_receptive_field)
                history_headpose = history_headpose.reshape(-1, opt.A2H_receptive_field).T
                history_headpose = torch.from_numpy(history_headpose).unsqueeze(0).float().to(self.device)
                infer_start = 0
            else:
                return None

                # evaluate mode
            self.Audio2Headpose.eval()

            with torch.no_grad():
                for i in tqdm(range(infer_start, nframe), desc='generating headpose'):
                    history_start = i - infer_start
                    input_audio_feats = audio_feats[
                                        history_start + frame_future: history_start + frame_future + opt.A2H_receptive_field]
                    input_audio_feats = torch.from_numpy(input_audio_feats).unsqueeze(0).float().to(self.device)

                    if self.opt.feature_decoder == 'WaveNet':
                        preds = self.Audio2Headpose.forward(history_headpose, input_audio_feats)
                    elif self.opt.feature_decoder == 'LSTM':
                        preds = self.Audio2Headpose.forward(input_audio_feats)

                    if opt.loss == 'GMM':
                        pred_data = Sample_GMM(preds, opt.A2H_GMM_ncenter, opt.A2H_GMM_ndim, sigma_scale=sigma_scale)
                    elif opt.loss == 'L2':
                        pred_data = preds

                    # get predictions
                    pred_headpose[i] = pred_data[0, 0].cpu().detach().numpy()
                    history_headpose = torch.cat((history_headpose[:, 1:, :], pred_data.to(self.device)),
                                                 dim=1)  # add in time-axis

            return pred_headpose

        elif opt.feature_decoder == 'LSTM':
            self.Audio2Headpose.eval()
            with torch.no_grad():
                input = torch.from_numpy(audio_feats).unsqueeze(0).float().to(self.device)
                preds = self.Audio2Headpose.forward(input)
                if opt.loss == 'GMM':
                    pred_data = Sample_GMM(preds, opt.A2H_GMM_ncenter, opt.A2H_GMM_ndim, sigma_scale=sigma_scale)
                elif opt.loss == 'L2':
                    pred_data = preds
                # get predictions
                pred_headpose = pred_data[0].cpu().detach().numpy()

            return pred_headpose


class Feature2FaceModel(BaseModel):
    def __init__(self, opt):
        """Initialize the Feature2Face class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.Tensor = torch.cuda.FloatTensor
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # define networks
        self.model_names = ['Feature2Face_G']
        self.Feature2Face_G = networks.init_net(feature2face_G.Feature2Face_G(opt), init_type='normal', init_gain=0.02,
                                                gpu_ids=opt.gpu_ids)
        if self.isTrain:
            if not opt.no_discriminator:
                self.model_names += ['Feature2Face_D']
                from . import feature2face_D
                self.Feature2Face_D = networks.init_net(feature2face_D.Feature2Face_D(opt), init_type='normal',
                                                        init_gain=0.02, gpu_ids=opt.gpu_ids)

        # define only during training time
        if self.isTrain:
            # define losses names
            self.loss_names_G = ['L1', 'VGG', 'Style', 'loss_G_GAN', 'loss_G_FM']
            # criterion
            self.criterionMaskL1 = MaskedL1Loss().cuda()
            self.criterionL1 = nn.L1Loss().cuda()
            self.criterionVGG = VGGLoss.cuda()
            self.criterionFlow = nn.L1Loss().cuda()

            # initialize optimizer G
            if opt.TTUR:
                beta1, beta2 = 0, 0.9
                lr = opt.lr / 2
            else:
                beta1, beta2 = opt.beta1, 0.999
                lr = opt.lr
            self.optimizer_G = torch.optim.Adam([{'params': self.Feature2Face_G.module.parameters(),
                                                  'initial_lr': lr}],
                                                lr=lr,
                                                betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_G)

            # fp16 training
            if opt.fp16:
                self.scaler = torch.cuda.amp.GradScaler()

            # discriminator setting
            if not opt.no_discriminator:
                self.criterionGAN = GANLoss(opt.gan_mode, tensor=self.Tensor)
                self.loss_names_D = ['D_real', 'D_fake']
                # initialize optimizer D
                if opt.TTUR:
                    beta1, beta2 = 0, 0.9
                    lr = opt.lr * 2
                else:
                    beta1, beta2 = opt.beta1, 0.999
                    lr = opt.lr
                self.optimizer_D = torch.optim.Adam([{'params': self.Feature2Face_D.module.netD.parameters(),
                                                      'initial_lr': lr}],
                                                    lr=lr,
                                                    betas=(beta1, beta2))
                self.optimizers.append(self.optimizer_D)

    def init_paras(self, dataset):
        opt = self.opt
        iter_path = os.path.join(self.save_dir, 'iter.txt')
        start_epoch, epoch_iter = 1, 0
        ### if continue training, recover previous states
        if opt.continue_train:
            if os.path.exists(iter_path):
                start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
                print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
                # change epoch count & update schedule settings
                opt.epoch_count = start_epoch
                self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
                # print lerning rate
                lr = self.optimizers[0].param_groups[0]['lr']
                print('update learning rate: {} -> {}'.format(opt.lr, lr))
            else:
                print('not found training log, hence training from epoch 1')
            # change training sequence length
        #            if start_epoch > opt.nepochs_step:
        #                dataset.dataset.update_training_batch((start_epoch-1)//opt.nepochs_step)

        total_steps = (start_epoch - 1) * len(dataset) + epoch_iter
        total_steps = total_steps // opt.print_freq * opt.print_freq

        return start_epoch, opt.print_freq, total_steps, epoch_iter

    def set_input(self, data, data_info=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        self.feature_map, self.cand_image, self.tgt_image, self.facial_mask = \
            data['feature_map'], data['cand_image'], data['tgt_image'], data['weight_mask']
        self.feature_map = self.feature_map.to(self.device)
        self.cand_image = self.cand_image.to(self.device)
        self.tgt_image = self.tgt_image.to(self.device)

    #        self.facial_mask = self.facial_mask.to(self.device)

    def forward(self):
        ''' forward pass for feature2Face
        '''
        self.input_feature_maps = torch.cat([self.feature_map, self.cand_image], dim=1)
        self.fake_pred = self.Feature2Face_G(self.input_feature_maps)

    def backward_G(self):
        """Calculate GAN and other loss for the generator"""
        # GAN loss
        real_AB = torch.cat((self.input_feature_maps, self.tgt_image), dim=1)
        fake_AB = torch.cat((self.input_feature_maps, self.fake_pred), dim=1)
        pred_real = self.Feature2Face_D(real_AB)
        pred_fake = self.Feature2Face_D(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # L1, vgg, style loss
        loss_l1 = self.criterionL1(self.fake_pred, self.tgt_image) * self.opt.lambda_L1
        #        loss_maskL1 = self.criterionMaskL1(self.fake_pred, self.tgt_image, self.facial_mask * self.opt.lambda_mask)
        loss_vgg, loss_style = self.criterionVGG(self.fake_pred, self.tgt_image, style=True)
        loss_vgg = torch.mean(loss_vgg) * self.opt.lambda_feat
        loss_style = torch.mean(loss_style) * self.opt.lambda_feat
        # feature matching loss
        loss_FM = self.compute_FeatureMatching_loss(pred_fake, pred_real)

        # combine loss and calculate gradients

        if not self.opt.fp16:
            self.loss_G = loss_G_GAN + loss_l1 + loss_vgg + loss_style + loss_FM  # + loss_maskL1
            self.loss_G.backward()
        else:
            with autocast():
                self.loss_G = loss_G_GAN + loss_l1 + loss_vgg + loss_style + loss_FM  # + loss_maskL1
            self.scaler.scale(self.loss_G).backward()

        self.loss_dict = {**self.loss_dict,
                          **dict(zip(self.loss_names_G, [loss_l1, loss_vgg, loss_style, loss_G_GAN, loss_FM]))}

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # GAN loss
        real_AB = torch.cat((self.input_feature_maps, self.tgt_image), dim=1)
        fake_AB = torch.cat((self.input_feature_maps, self.fake_pred), dim=1)
        pred_real = self.Feature2Face_D(real_AB)
        pred_fake = self.Feature2Face_D(fake_AB.detach())
        with autocast():
            loss_D_real = self.criterionGAN(pred_real, True) * 2
            loss_D_fake = self.criterionGAN(pred_fake, False)

        self.loss_D = (loss_D_fake + loss_D_real) * 0.5

        self.loss_dict = dict(zip(self.loss_names_D, [loss_D_real, loss_D_fake]))

        if not self.opt.fp16:
            self.loss_D.backward()
        else:
            self.scaler.scale(self.loss_D).backward()

    def compute_FeatureMatching_loss(self, pred_fake, pred_real):
        # GAN feature matching loss
        loss_FM = torch.zeros(1).cuda()
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        D_weights = 1.0 / self.opt.num_D
        for i in range(min(len(pred_fake), self.opt.num_D)):
            for j in range(len(pred_fake[i])):
                loss_FM += D_weights * feat_weights * \
                           self.criterionL1(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        return loss_FM

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        # only train single image generation
        ## forward
        self.forward()
        # update D
        self.set_requires_grad(self.Feature2Face_D, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        if not self.opt.fp16:
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights
        else:
            with autocast():
                self.backward_D()
            self.scaler.step(self.optimizer_D)

        # update G
        self.set_requires_grad(self.Feature2Face_D, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        if not self.opt.fp16:
            self.backward_G()  # calculate graidents for G
            self.optimizer_G.step()  # udpate G's weights
        else:
            with autocast():
                self.backward_G()
            self.scaler.step(self.optimizer_G)
            self.scaler.update()

    def inference(self, feature_map, cand_image):
        """ inference process """
        with torch.no_grad():
            if cand_image == None:
                input_feature_maps = feature_map
            else:
                input_feature_maps = torch.cat([feature_map, cand_image], dim=1)
            if not self.opt.fp16:
                fake_pred = self.Feature2Face_G(input_feature_maps)
            else:
                with autocast():
                    fake_pred = self.Feature2Face_G(input_feature_maps)
        return fake_pred

class Audio2FeatureModel(BaseModel):
    def __init__(self, opt):
        """Initialize the Audio2Feature class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # define networks
        self.model_names = ['Audio2Feature']
        self.Audio2Feature = networks.init_net(audio2feature.Audio2Feature(opt), init_type='normal', init_gain=0.02,
                                               gpu_ids=opt.gpu_ids)

        # define only during training time
        if self.isTrain:
            # losses
            self.featureL2loss = torch.nn.MSELoss().to(self.device)
            # optimizer
            self.optimizer = torch.optim.Adam([{'params': self.Audio2Feature.parameters(),
                                                'initial_lr': opt.lr}], lr=opt.lr, betas=(0.9, 0.99))

            self.optimizers.append(self.optimizer)

            if opt.continue_train:
                self.resume_training()

    def resume_training(self):
        opt = self.opt
        ### if continue training, recover previous states
        print('Resuming from epoch %s ' % (opt.load_epoch))
        # change epoch count & update schedule settings
        opt.epoch_count = int(opt.load_epoch)
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        # print lerning rate
        lr = self.optimizers[0].param_groups[0]['lr']
        print('update learning rate: {} -> {}'.format(opt.lr, lr))

    def set_input(self, data, data_info=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """

        self.audio_feats, self.target_info = data
        #        b, item_length, mel_channels, width = self.audio_feats.shape
        self.audio_feats = self.audio_feats.to(self.device)
        self.target_info = self.target_info.to(self.device)

        # gaussian noise

    #        if self.opt.gaussian_noise:
    #            self.audio_feats = self.opt.gaussian_noise_scale * torch.randn(self.audio_feats.shape).cuda()
    #            self.target_info += self.opt.gaussian_noise_scale * torch.randn(self.target_info.shape).cuda()

    def forward(self):
        '''
        Args:
            history_landmarks: [b, T, ndim]
            audio_features: [b, 1, nfeas, nwins]
        Returns:
            preds: [b, T, output_channels]
        '''
        self.preds = self.Audio2Feature.forward(self.audio_feats)

    def calculate_loss(self):
        """ calculate loss in detail, only forward pass included"""
        if self.opt.loss == 'GMM':
            b, T, _ = self.target_info.shape
            self.loss_GMM = self.criterion_GMM(self.preds, self.target_info)
            self.loss = self.loss_GMM

        elif self.opt.loss == 'L2':
            frame_future = self.opt.frame_future
            if not frame_future == 0:
                self.loss = self.featureL2loss(self.preds[:, frame_future:], self.target_info[:, :-frame_future]) * 1000
            else:
                self.loss = self.featureL2loss(self.preds, self.target_info) * 1000

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.calculate_loss()
        self.loss.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.optimizer.zero_grad()  # clear optimizer parameters grad
        self.forward()  # forward pass
        self.backward()  # calculate loss and gradients
        self.optimizer.step()  # update gradients

    def validate(self):
        """ validate process """
        with torch.no_grad():
            self.forward()
            self.calculate_loss()

    def generate_sequences(self, audio_feats, sample_rate=16000, fps=60, fill_zero=True, opt=[]):
        ''' generate landmark sequences given audio and a initialized landmark.
        Note that the audio input should have the same sample rate as the training.
        Args:
            audio_sequences: [n,], in numpy
            init_landmarks: [npts, 2], in numpy
            sample_rate: audio sample rate, should be same as training process.
            method(str): optional, how to generate the sequence, indeed it is the
                loss function during training process. Options are 'L2' or 'GMM'.
        Reutrns:
            landmark_sequences: [T, npts, 2] predition landmark sequences
        '''

        frame_future = opt.frame_future
        nframe = int(audio_feats.shape[0] / 2)

        if not frame_future == 0:
            audio_feats_insert = np.repeat(audio_feats[-1], 2 * (frame_future)).reshape(-1, 2 * (frame_future)).T
            audio_feats = np.concatenate([audio_feats, audio_feats_insert])

        # evaluate mode
        self.Audio2Feature.eval()

        with torch.no_grad():
            input = torch.from_numpy(audio_feats).unsqueeze(0).float().to(self.device)
            preds = self.Audio2Feature.forward(input)

            # drop first frame future results
        if not frame_future == 0:
            preds = preds[0, frame_future:].cpu().detach().numpy()
        else:
            preds = preds[0, :].cpu().detach().numpy()

        assert preds.shape[0] == nframe

        return preds


class live_speech_portraits(AbstractTalkingFace):

    def __init__(self, opt):
        self.logger = getLogger()
        super(live_speech_portraits, self).__init__()
        self.opt = opt
        return

    def calculate_loss(self, interaction):
        """ calculate loss in detail, only forward pass included"""
        if self.opt.loss == 'GMM':
            self.loss_GMM = self.criterion_GMM(self.preds_headpose, self.target_headpose)
            self.loss = self.loss_GMM
        elif self.opt.loss == 'L2':
            self.loss_L2 = self.criterion_L2(self.preds_headpose, self.target_headpose)
            self.loss = self.loss_L2

        if not self.opt.smooth_loss == 0:
            mu_gen = Sample_GMM(self.preds_headpose,
                                self.Audio2Headpose.module.WaveNet.ncenter,
                                self.Audio2Headpose.module.WaveNet.ndim,
                                sigma_scale=0)
            self.smooth_loss = (mu_gen[:, 2:] + self.target_headpose[:, :-2] - 2 * self.target_headpose[:, 1:-1]).mean(
                dim=2).abs().mean()
            self.loss += self.smooth_loss * self.opt.smooth_loss

    def predict(self, driving_audio: Path = Input(description='driving audio, if the file is more than 20 seconds, only the first 20 seconds will be processed for video generation'),
                talking_head: str = Input(description="choose a talking head", choices=['May', 'Obama1', 'Obama2', 'Nadella', 'McStay'], default='May')
                ) -> Path:
        ############################### I/O Settings ##############################
        device = self.config['device']
        optID = self.config['dataset_params']['root'].split('/')[-1]
        driving_audio = self.config['driving_audio_path']
        data_root = self.config['dataset_params']['root']

        # create the results folder
        audio_name = driving_audio.split('/')[-1].split('.')[-2]
        save_root = join('./results/', optID, audio_name)
        if not os.path.exists(save_root):
            os.makedirs(save_root)


        ############################ Hyper Parameters #############################
        h, w, sr, FPS = 512, 512, 16000, 60
        mouth_indices = np.concatenate([np.arange(4, 11), np.arange(46, 64)])
        eye_brow_indices = [27, 65, 28, 68, 29, 67, 30, 66, 31, 72, 32, 69, 33, 70, 34, 71]
        eye_brow_indices = np.array(eye_brow_indices, np.int32)


        ############################ Pre-defined Data #############################
        mean_pts3d = np.load(join(data_root, 'mean_pts3d.npy'))
        fit_data = np.load(self.config['dataset_params']['fit_data_path'])
        pts3d = np.load(self.config['dataset_params']['pts3d_path']) - mean_pts3d
        trans = fit_data['trans'][:, :, 0].astype(np.float32)
        mean_translation = trans.mean(axis=0)
        candidate_eye_brow = pts3d[10:, eye_brow_indices]
        std_mean_pts3d = np.load(self.config['dataset_params']['pts3d_path']).mean(axis=0)

        # candidates images
        img_candidates = []
        for j in range(4):
            output = imread(join(data_root, 'candidates', f'normalized_full_{j}.jpg'))
            output = albumentations.pytorch.transforms.ToTensor(normalize={'mean': (0.5, 0.5, 0.5),
                                                              'std': (0.5, 0.5, 0.5)})(image=output)['image']
            img_candidates.append(output)
        img_candidates = torch.cat(img_candidates).unsqueeze(0).to(device)

        # shoulders
        shoulders = np.load(join(data_root, 'normalized_shoulder_points.npy'))
        shoulder3D = np.load(join(data_root, 'shoulder_points3D.npy'))[1]
        ref_trans = trans[1]

        # camera matrix, we always use training set intrinsic parameters.
        camera = utils.camera()
        camera_intrinsic = np.load(join(data_root, 'camera_intrinsic.npy')).astype(np.float32)
        APC_feat_database = np.load(join(data_root, 'APC_feature_base.npy'))

        # load reconstruction data
        scale = sio.loadmat(join(data_root, 'id_scale.mat'))['scale'][0, 0]
        # Audio2Mel_torch = audio_funcs.Audio2Mel(n_fft=512, hop_length=int(16000/120), win_length=int(16000/60), sampling_rate=16000,
            #                                         n_mel_channels=80, mel_fmin=90, mel_fmax=7600.0).to(device)



        ########################### Experiment Settings ###########################
        #### user config
        use_LLE = self.config['model_params']['APC']['use_LLE']
        Knear = self.config['model_params']['APC']['Knear']
        LLE_percent = self.config['model_params']['APC']['LLE_percent']
        headpose_sigma = self.config['model_params']['Headpose']['sigma']
        Feat_smooth_sigma = self.config['model_params']['Audio2Mouth']['smooth']
        Head_smooth_sigma = self.config['model_params']['Headpose']['smooth']
        Feat_center_smooth_sigma, Head_center_smooth_sigma = 0, 0
        AMP_method = self.config['model_params']['Audio2Mouth']['AMP'][0]
        Feat_AMPs = self.config['model_params']['Audio2Mouth']['AMP'][1:]
        rot_AMP, trans_AMP = self.config['model_params']['Headpose']['AMP']
        shoulder_AMP = self.config['model_params']['Headpose']['shoulder_AMP']
        save_feature_maps = self.config['model_params']['Image2Image']['save_input']

        #### common settings
        Featopt = FeatureOptions().parse()
        Headopt = HeadposeOptions().parse()
        Renderopt = RenderOptions().parse()
        Featopt.load_epoch = self.config['model_params']['Audio2Mouth']['ckp_path']
        Headopt.load_epoch = self.config['model_params']['Headpose']['ckp_path']
        Renderopt.dataroot = self.config['dataset_params']['root']
        Renderopt.load_epoch = self.config['model_params']['Image2Image']['ckp_path']
        Renderopt.size = self.config['model_params']['Image2Image']['size']
        ## GPU or CPU
        if device == 'cpu':
            Featopt.gpu_ids = Headopt.gpu_ids = Renderopt.gpu_ids = []



        ############################# Load Models #################################
        print('---------- Loading Model: APC-------------')
        APC_model = APC_encoder(self.config['model_params']['APC']['mel_dim'],
                                self.config['model_params']['APC']['hidden_size'],
                                self.config['model_params']['APC']['num_layers'],
                                self.config['model_params']['APC']['residual'])
        APC_model.load_state_dict(torch.load(self.config['model_params']['APC']['ckp_path']), strict=False)
        if device == 'cuda':
            APC_model.cuda()
        APC_model.eval()
        print('---------- Loading Model: {} -------------'.format(Featopt.task))
        Audio2Feature = create_model(Featopt)
        Audio2Feature.setup(Featopt)
        Audio2Feature.eval()
        print('---------- Loading Model: {} -------------'.format(Headopt.task))
        Audio2Headpose = create_model(Headopt)
        Audio2Headpose.setup(Headopt)
        Audio2Headpose.eval()
        if Headopt.feature_decoder == 'WaveNet':
            if device == 'cuda':
                Headopt.A2H_receptive_field = Audio2Headpose.Audio2Headpose.module.WaveNet.receptive_field
            else:
                Headopt.A2H_receptive_field = Audio2Headpose.Audio2Headpose.WaveNet.receptive_field
        print('---------- Loading Model: {} -------------'.format(Renderopt.task))
        facedataset = create_dataset(Renderopt)
        Feature2Face = create_model(Renderopt)
        Feature2Face.setup(Renderopt)
        Feature2Face.eval()
        visualizer = Visualizer(Renderopt)


        ############################## Inference ##################################
        print('Processing audio: {} ...'.format(audio_name))
        # read audio
        audio, _ = librosa.load(driving_audio, sr=sr)
        total_frames = np.int32(audio.shape[0] / sr * FPS)


        #### 1. compute APC features
        print('1. Computing APC features...')
        mel80 = utils.compute_mel_one_sequence(audio, device=device)
        mel_nframe = mel80.shape[0]
        with torch.no_grad():
            length = torch.Tensor([mel_nframe])
            mel80_torch = torch.from_numpy(mel80.astype(np.float32)).to(device).unsqueeze(0)
            hidden_reps = APC_model.forward(mel80_torch, length)[0]  # [mel_nframe, 512]
            hidden_reps = hidden_reps.cpu().numpy()
        audio_feats = hidden_reps


        #### 2. manifold projection
        if use_LLE:
            print('2. Manifold projection...')
            ind = utils.KNN_with_torch(audio_feats, APC_feat_database, K=Knear)
            weights, feat_fuse = utils.compute_LLE_projection_all_frame(audio_feats, APC_feat_database, ind,
                                                                        audio_feats.shape[0])
            audio_feats = audio_feats * (1 - LLE_percent) + feat_fuse * LLE_percent



        #### 3. Audio2Mouth
        print('3. Audio2Mouth inference...')
        pred_Feat = Audio2Feature.generate_sequences(audio_feats, sr, FPS, fill_zero=True, opt=Featopt)




        #### 4. Audio2Headpose
        print('4. Headpose inference...')
        # set history headposes as zero
        pre_headpose = np.zeros(Headopt.A2H_wavenet_input_channels, np.float32)
        pred_Head = Audio2Headpose.generate_sequences(audio_feats, pre_headpose, fill_zero=True, sigma_scale=0.3,
                                                      opt=Headopt)



        #### 5. Post-Processing
        print('5. Post-processing...')
        nframe = min(pred_Feat.shape[0], pred_Head.shape[0])
        pred_pts3d = np.zeros([nframe, 73, 3])
        pred_pts3d[:, mouth_indices] = pred_Feat.reshape(-1, 25, 3)[:nframe]



        ## mouth
        pred_pts3d = utils.landmark_smooth_3d(pred_pts3d, Feat_smooth_sigma, area='only_mouth')
        pred_pts3d = utils.mouth_pts_AMP(pred_pts3d, True, AMP_method, Feat_AMPs)
        pred_pts3d = pred_pts3d + mean_pts3d
        pred_pts3d = utils.solve_intersect_mouth(pred_pts3d)  # solve intersect lips if exist



        ## headpose
        pred_Head[:, 0:3] *= rot_AMP
        pred_Head[:, 3:6] *= trans_AMP
        pred_headpose = utils.headpose_smooth(pred_Head[:, :6], Head_smooth_sigma).astype(np.float32)
        pred_headpose[:, 3:] += mean_translation
        pred_headpose[:, 0] += 180



        ## compute projected landmarks
        pred_landmarks = np.zeros([nframe, 73, 2], dtype=np.float32)
        final_pts3d = np.zeros([nframe, 73, 3], dtype=np.float32)
        final_pts3d[:] = std_mean_pts3d.copy()
        final_pts3d[:, 46:64] = pred_pts3d[:nframe, 46:64]
        for k in tqdm(range(nframe)):
            ind = k % candidate_eye_brow.shape[0]
            final_pts3d[k, eye_brow_indices] = candidate_eye_brow[ind] + mean_pts3d[eye_brow_indices]
            pred_landmarks[k], _, _ = utils.project_landmarks(camera_intrinsic, camera.relative_rotation,
                                                              camera.relative_translation, scale,
                                                              pred_headpose[k], final_pts3d[k])


        ## Upper Body Motion
        pred_shoulders = np.zeros([nframe, 18, 2], dtype=np.float32)
        pred_shoulders3D = np.zeros([nframe, 18, 3], dtype=np.float32)
        for k in range(nframe):
            diff_trans = pred_headpose[k][3:] - ref_trans
            pred_shoulders3D[k] = shoulder3D + diff_trans * shoulder_AMP
            # project
            project = camera_intrinsic.dot(pred_shoulders3D[k].T)
            project[:2, :] /= project[2, :]  # divide z
            pred_shoulders[k] = project[:2, :].T



        #### 6. Image2Image translation & Save resuls
        print('6. Image2Image translation & Saving results...')
        for ind in tqdm(range(0, nframe), desc='Image2Image translation inference'):
            # feature_map: [input_nc, h, w]
            current_pred_feature_map = facedataset.dataset.get_data_test_mode(pred_landmarks[ind],
                                                                              pred_shoulders[ind],
                                                                              facedataset.dataset.image_pad)
            input_feature_maps = current_pred_feature_map.unsqueeze(0).to(device)
            pred_fake = Feature2Face.inference(input_feature_maps, img_candidates)
            # save results
            visual_list = [('pred', util.tensor2im(pred_fake[0]))]
            if save_feature_maps:
                visual_list += [('input', np.uint8(current_pred_feature_map[0].cpu().numpy() * 255))]
            visuals = OrderedDict(visual_list)
            visualizer.save_images(save_root, visuals, str(ind + 1))

        ## make videos
        # generate corresponding audio, reused for all results
        tmp_audio_path = join(save_root, 'tmp.wav')
        tmp_audio_clip = audio[: np.int32(nframe * sr / FPS)]
        librosa.output.write_wav(tmp_audio_path, tmp_audio_clip, sr)

    def generate_batch(self):
        return

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + set_color("\nTrainable parameters", "blue")
            + f": {params}"
        )
