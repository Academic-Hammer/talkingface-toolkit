import os
import torch
from collections import OrderedDict
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn import functional as F
import numpy as np
from torchvision import models
import pdb


class PHADTF_GAN():

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        PHADTF_GAN.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.isTrain:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'mem']
            if self.opt.attention:
                self.loss_names += ['G_Att', 'G_Att_smooth', 'G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.opt.Nw > 1:
            self.visual_names = ['real_A_0', 'real_A_1', 'real_A_2', 'fake_B', 'real_B']
        if self.opt.attention:
            self.visual_names += ['fake_B_img', 'fake_B_mask_vis']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'mem']
        else:  # during test time, only load G and mem
            self.model_names = ['G', 'mem']
        # define networks (both generator and discriminator)
        self.netG = define_G(opt.input_nc*opt.Nw, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, feat_dim=opt.iden_feat_dim)
        
        self.netmem = Memory_Network(mem_size = opt.mem_size, color_feat_dim = opt.iden_feat_dim, spatial_feat_dim = opt.spatial_feat_dim, top_k = opt.top_k, alpha = opt.alpha, gpu_ids = self.gpu_ids).to(self.device)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = define_D(opt.input_nc*opt.Nw + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_mem = torch.optim.Adam(self.netmem.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_mem)
        self.replace = 0
        self.update = 0

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unetac_adain_256', dataset_mode='aligned_feature_multi', direction='AtoB',Nw=3)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_mask', type=float, default=0.1, help='lambda mask loss')
            parser.add_argument('--lambda_mask_smooth', type=float, default=1e-5, help='lambda mask smooth loss')
        else:
            parser.add_argument('--test_use_gt', type=int, default=0, help='use gt feature in test')
        parser.add_argument('--attention', type=int, default=1, help='whether to use attention mechanism')
        parser.add_argument('--do_saturate_mask', action="store_true", default=False, help='do use mask_fake for mask_cyc')
        # for memory net
        parser.add_argument("--iden_feat_dim", type = int, default = 512)
        parser.add_argument("--spatial_feat_dim", type = int, default = 512)
        parser.add_argument("--mem_size", type = int, default = 30000)#982=819*1.2
        parser.add_argument("--alpha", type = float, default = 0.3)
        parser.add_argument("--top_k", type = int, default = 256)
        parser.add_argument("--iden_thres", type = float, default = 0.98)#0.5)
        parser.add_argument("--iden_feat_dir", type = str, default = 'arcface/iden_feat/')


        return parser

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device) # channel is input_nc * Nw
        self.real_B = input['B' if AtoB else 'A'].to(self.device) # channel is output_nc
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # for memory net
        if self.isTrain or self.opt.test_use_gt:
            self.real_B_feat = input['B_feat' if AtoB else 'A_feat'].to(self.device)
        self.resnet_input = input['resnet_input'].to(self.device)
        self.idx = input['index'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.attention:
            if self.isTrain or self.opt.test_use_gt:
                self.fake_B_img, self.fake_B_mask = self.netG(self.real_A, self.real_B_feat)
            else:
                query = self.netmem(self.resnet_input)
                #pdb.set_trace()
                top1_feature, top1_index, topk_index = self.netmem.topk_feature(query, 1)
                top1_feature = top1_feature[:, 0, :]
                self.fake_B_img, self.fake_B_mask = self.netG(self.real_A, top1_feature)
            self.fake_B_mask = self._do_if_necessary_saturate_mask(self.fake_B_mask, saturate=self.opt.do_saturate_mask)
            self.fake_B = self.fake_B_mask * self.real_A[:,-self.opt.input_nc:] + (1 - self.fake_B_mask) * self.fake_B_img
            #print(torch.min(self.fake_B_mask), torch.max(self.fake_B_mask))
            self.fake_B_mask_vis = self.fake_B_mask * 2 - 1
        else:
            if self.isTrain or self.opt.test_use_gt:
                self.fake_B = self.netG(self.real_A, self.real_B_feat)
            else:
                query = self.netmem(self.resnet_input)
                top1_feature, _, _ = self.netmem.topk_feature(query, 1)
                top1_feature = top1_feature[:, 0, :]
                self.fake_B = self.netG(self.real_A, top1_feature)
        if self.opt.Nw > 1:
            self.real_A_0 = self.real_A[:,:self.opt.input_nc]
            self.real_A_1 = self.real_A[:,self.opt.input_nc:2*self.opt.input_nc]
            self.real_A_2 = self.real_A[:,-self.opt.input_nc:]

    def update_mem(self):
        with torch.no_grad():
            resnet_feature = self.netmem(self.resnet_input)
            replace = self.netmem.memory_update(resnet_feature, self.real_B_feat, self.opt.iden_thres, self.idx)
            if replace:
                self.replace += 1
            else:
                self.update += 1

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # Loss for attention mask
        if self.opt.attention:
            # the attention mask can easily saturate to 1, which makes that generator has no effect
            self.loss_G_Att = torch.mean(self.fake_B_mask) * self.opt.lambda_mask
            # to enforce smooth spatial color transformation
            self.loss_G_Att_smooth = self._compute_loss_smooth(self.fake_B_mask) * self.opt.lambda_mask_smooth
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if self.opt.attention:
            self.loss_G += self.loss_G_Att + self.loss_G_Att_smooth
        self.loss_G.backward()

    def optimize_parameters(self):
        # update mem
        self.optimizer_mem.zero_grad()
        self.backward_mem()
        self.optimizer_mem.step()
        self.update_mem()

        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

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
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if name != 'mem':
                    if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                        torch.save(net.module.cpu().state_dict(), save_path)
                        net.cuda(self.gpu_ids[0])
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                else:
                    torch.save({'mem_model' : net.cpu().state_dict(),
                       'mem_key' : net.spatial_key.cpu(),
                       'mem_value' : net.color_value.cpu(),
                       'mem_age' : net.age.cpu(),
                       'mem_index' : net.top_index.cpu()}, save_path)
                    net.cuda(self.gpu_ids[0])

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
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                if not os.path.exists(load_path):
                    load_path =  os.path.join(self.opt.checkpoints_dir, self.opt.name.split('/')[0],load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                if name != 'mem':
                    net.load_state_dict(state_dict)
                else:
                    net.load_state_dict(state_dict['mem_model'])
                    net.spatial_key = state_dict['mem_key']
                    net.color_value = state_dict['mem_value']
                    net.age = state_dict['mem_age']
                    net.top_index = state_dict['mem_index']
                    #print(net.spatial_key)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
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

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m

##################################################Basic network compotents##################################################################
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        #print(classname, hasattr(m, 'weight'),)
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], feat_dim=512):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    
    norm_layer = get_norm_layer(norm_type=norm)
    net = unet_generator_ac_adain(input_nc, output_nc, ngf, feat_dim)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetSkipConnectionACBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionACBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)

        # assume outermost:
        upconv = nn.ConvTranspose2d(inner_nc * 2, inner_nc,
                                    kernel_size=4, stride=2,
                                    padding=1)
        upnorm = norm_layer(inner_nc)
        down = [downconv]
        up = [uprelu, upconv, upnorm, uprelu]
        model = down + [submodule] + up

        self.model = nn.Sequential(*model)

        layers = []
        layers.append(nn.Conv2d(inner_nc, 3, kernel_size=7, stride=1, padding=3))
        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(inner_nc, 1, kernel_size=7, stride=1, padding=3))
        layers.append(nn.Sigmoid())
        self.attention_reg = nn.Sequential(*layers)

    def forward(self, x):
        features = self.model(x)
        return self.img_reg(features), self.attention_reg(features)

################################################################################################################
#                                          Unet with AdaIN
################################################################################################################
class unet_generator_ac_adain(nn.Module):
    
    def __init__(self, input_nc, output_nc, ngf, feat_dim = 512):
        super(unet_generator_ac_adain, self).__init__()
        
        self.e1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.e2 = unet_encoder_block(ngf, ngf * 2)
        self.e3 = unet_encoder_block(ngf * 2, ngf * 4)
        self.e4 = unet_encoder_block(ngf * 4, ngf * 8)
        self.e5 = unet_encoder_block(ngf * 8, ngf * 8)
        self.e6 = unet_encoder_block(ngf * 8, ngf * 8)
        self.e7 = unet_encoder_block(ngf * 8, ngf * 8)
        self.e8 = unet_encoder_block(ngf * 8, ngf * 8, norm = None)

        self.d1 = unet_decoder_block(ngf * 8, ngf * 8)
        self.d2 = unet_decoder_block(ngf * 8 * 2, ngf * 8)
        self.d3 = unet_decoder_block(ngf * 8 * 2, ngf * 8)
        self.d4 = unet_decoder_block(ngf * 8 * 2, ngf * 8, drop_out = None)
        self.d5 = unet_decoder_block(ngf * 8 * 2, ngf * 4, drop_out = None)
        self.d6 = unet_decoder_block(ngf * 4 * 2, ngf * 2, drop_out = None)
        self.d7 = unet_decoder_block(ngf * 2 * 2, ngf, drop_out = None)
        self.d8 = unet_decoder_ac_block(ngf * 2, output_nc, norm = None, drop_out = None)
        
        self.layers = [self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7, self.e8,
                 self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7, self.d8]
        
        self.mlp = MLP(feat_dim, self.get_num_adain_params(self.layers), self.get_num_adain_params(self.layers), 3)

    
    def forward(self, x, feat):
        
        ### AdaIn params
        adain_params = self.mlp(feat)
        self.assign_adain_params(adain_params, self.layers)
        
        ### Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        
        ### Decoder
        d1_ = self.d1(e8)
        d1 = torch.cat([d1_, e7], dim = 1)
        
        d2_ = self.d2(d1)
        d2 = torch.cat([d2_, e6], dim = 1)
        
        d3_ = self.d3(d2)
        d3 = torch.cat([d3_, e5], dim = 1)
        
        d4_ = self.d4(d3)
        d4 = torch.cat([d4_, e4], dim = 1)
        
        d5_ = self.d5(d4)
        d5 = torch.cat([d5_, e3], dim = 1)
        
        d6_ = self.d6(d5)
        d6 = torch.cat([d6_, e2], dim = 1)
        
        d7_ = self.d7(d6)
        d7 = torch.cat([d7_, e1], dim = 1)
        
        color, attention = self.d8(d7)
        
        return color, attention
    
    def get_num_adain_params(self, _module):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for model in _module:
            for m in model.modules():
                if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                    num_adain_params += 2*m.num_features
        return num_adain_params
    
    def assign_adain_params(self, adain_params, _module):
        # assign the adain_params to the AdaIN layers in model
        for model in _module:
            for m in model.modules():
                if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                    mean = adain_params[:, :m.num_features]
                    std = adain_params[:, m.num_features:2*m.num_features]
                    m.bias = mean.contiguous().view(-1)
                    m.weight = std.contiguous().view(-1)
                    if adain_params.size(1) > 2*m.num_features:
                        adain_params = adain_params[:, 2*m.num_features:]

    def forward(self, x, feat):
        
        ### AdaIn params
        adain_params = self.mlp(feat)
        self.assign_adain_params(adain_params, self.layers)
        
        ### Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        
        ### Decoder
        d1_ = self.d1(e8)
        d1 = torch.cat([d1_, e7], dim = 1)
        
        d2_ = self.d2(d1)
        d2 = torch.cat([d2_, e6], dim = 1)
        
        d3_ = self.d3(d2)
        d3 = torch.cat([d3_, e5], dim = 1)
        
        d4_ = self.d4(d3)
        d4 = torch.cat([d4_, e4], dim = 1)
        
        d5_ = self.d5(d4)
        d5 = torch.cat([d5_, e3], dim = 1)
        
        d6_ = self.d6(d5)
        d6 = torch.cat([d6_, e2], dim = 1)
        
        d7_ = self.d7(d6)
        d7 = torch.cat([d7_, e1], dim = 1)
        
        d8 = self.d8(d7)
        
        output = self.tanh(d8)
        
        return output
    
    def get_num_adain_params(self, _module):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for model in _module:
            for m in model.modules():
                if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                    num_adain_params += 2*m.num_features
        return num_adain_params
    
    def assign_adain_params(self, adain_params, _module):
        # assign the adain_params to the AdaIN layers in model
        for model in _module:
            for m in model.modules():
                if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                    mean = adain_params[:, :m.num_features]
                    std = adain_params[:, m.num_features:2*m.num_features]
                    m.bias = mean.contiguous().view(-1)
                    m.weight = std.contiguous().view(-1)
                    if adain_params.size(1) > 2*m.num_features:
                        adain_params = adain_params[:, 2*m.num_features:]
        
class unet_encoder_block(nn.Module):
    
    def __init__(self, input_nc, output_nc, ks = 4, stride = 2, padding = 1, norm = 'adain', act = nn.LeakyReLU(inplace = True, negative_slope = 0.2)):
        super(unet_encoder_block, self).__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, ks, stride, padding)
        m = [act, self.conv]
        
        if norm == 'adain':
            m.append(AdaptiveInstanceNorm2d(output_nc))
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        return self.body(x)
    
class unet_decoder_block(nn.Module):
    
    def __init__(self, input_nc, output_nc, ks = 4, stride = 2, padding = 1, norm = 'adain', act = nn.ReLU(inplace = True), drop_out = 0.5):
        super(unet_decoder_block, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_nc, output_nc, ks, stride, padding)
        m = [act, self.deconv]
        
        if norm == 'adain':
            m.append(AdaptiveInstanceNorm2d(output_nc))
            
        if drop_out is not None:
            m.append(nn.Dropout(drop_out))
            
        self.body = nn.Sequential(*m)
    
    def forward(self, x):
        return self.body(x)

class unet_decoder_ac_block(nn.Module):
    
    def __init__(self, input_nc, output_nc, ks = 4, stride = 2, padding = 1, norm = 'adain', act = nn.ReLU(inplace = True), drop_out = 0.5):
        super(unet_decoder_ac_block, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_nc, int(input_nc/2), ks, stride, padding)
        m = [act, self.deconv, AdaptiveInstanceNorm2d(int(input_nc/2)), act]

        self.body = nn.Sequential(*m)

        layers = []
        layers.append(nn.Conv2d(int(input_nc/2), output_nc, kernel_size=7, stride=1, padding=3))
        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(int(input_nc/2), 1, kernel_size=7, stride=1, padding=3))
        layers.append(nn.Sigmoid())
        self.attention_reg = nn.Sequential(*layers)
    
    def forward(self, x):
        features = self.body(x)
        return self.img_reg(features), self.attention_reg(features)
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, act = nn.ReLU(inplace = True)):

        super(MLP, self).__init__()
        self.model = []
        
        self.model.append(nn.Linear(input_dim, dim))
        self.model.append(act)
        
        for i in range(n_blk - 2):
            self.model.append(nn.Linear(dim, dim))
            self.model.append(act)
            
        self.model.append(nn.Linear(dim, output_dim))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = nn.functional.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
################################################################################################################



class UnetSkipConnectionRefineBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionRefineBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        refine = nn.Conv2d(outer_nc, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1, bias=use_bias)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm, uprelu, refine, upnorm, uprelu, refine, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            if use_dropout:
                up = [uprelu, upconv, upnorm, nn.Dropout(0.5), uprelu, refine, upnorm, nn.Dropout(0.5), uprelu, refine, upnorm, nn.Dropout(0.5)]
            else:
                up = [uprelu, upconv, upnorm, uprelu, refine, upnorm, uprelu, refine, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x):
        d = self.down(x)
        if not self.innermost:
            d = self.submodule(d)
        u = self.up(d)
        if self.outermost:
            return u
        else:
            return torch.cat([x, u], 1)

class UnetSkipConnectionRefineAddinputBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, dadd_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionRefineAddinputBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        refine = nn.Conv2d(outer_nc, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1, bias=use_bias)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2 + dadd_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc + dadd_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm, uprelu, refine, upnorm, uprelu, refine, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2 + dadd_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            if use_dropout:
                up = [uprelu, upconv, upnorm, nn.Dropout(0.5), uprelu, refine, upnorm, nn.Dropout(0.5), uprelu, refine, upnorm, nn.Dropout(0.5)]
            else:
                up = [uprelu, upconv, upnorm, uprelu, refine, upnorm, uprelu, refine, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, im):
        d = self.down(x)
        #print x.shape, d.shape, type(self)
        if not self.innermost:
            if isinstance(self.submodule,UnetSkipConnectionRefineBlock):
                d = self.submodule(d)
            else:
                im2 = nn.Upsample(scale_factor=[0.5,0.5],mode='bilinear')(im)
                #print im2.shape, im.shape
                d = self.submodule(d, im2)
            #print d.shape
        u = self.up(torch.cat([im, d], 1))
        if self.outermost:
            return u
        else:
            return torch.cat([x, u], 1)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)



##########################################Memory_Network##################################################################################
class ResNet18(nn.Module):
    def __init__(self, pre_trained = True, require_grad = False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained = True)       
        
        self.body = [layers for layers in self.model.children()]
        self.body.pop(-1)
        
        self.body = nn.Sequential(*self.body)
        
        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False
                
    def forward(self, x):
        x = self.body(x)
        x = x.view(-1, 512)
        return x


class Memory_Network(nn.Module):
    
    def __init__(self, mem_size, color_feat_dim = 512, spatial_feat_dim = 512, top_k = 256, alpha = 0.1, age_noise = 4.0, gpu_ids = []):
        
        super(Memory_Network, self).__init__()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.ResNet18 = ResNet18().to(self.device)
        self.ResNet18 = self.ResNet18.eval()
        self.mem_size = mem_size
        self.color_feat_dim = color_feat_dim
        self.spatial_feat_dim = spatial_feat_dim
        self.alpha = alpha
        self.age_noise = age_noise
        self.top_k = top_k
        
        ## Each color_value is probability distribution
        self.color_value = F.normalize(random_uniform((self.mem_size, self.color_feat_dim), 0, 0.01), p = 1, dim=1).to(self.device)
        
        self.spatial_key = F.normalize(random_uniform((self.mem_size, self.spatial_feat_dim), -0.01, 0.01), dim=1).to(self.device)
        self.age = torch.zeros(self.mem_size).to(self.device)
        
        self.top_index = torch.zeros(self.mem_size).to(self.device)
        self.top_index = self.top_index - 1.0
        
        self.color_value.requires_grad = False
        self.spatial_key.requires_grad = False
        
        self.Linear = nn.Linear(512, spatial_feat_dim)
        self.body = [self.ResNet18, self.Linear]
        self.body = nn.Sequential(*self.body)
        self.body = self.body.to(self.device)
        
    def forward(self, x):
        q = self.body(x)
        q = F.normalize(q, dim = 1)
        return q
    
    def unsupervised_loss(self, query, color_feat, color_thres):
        
        bs = query.size()[0]
        cosine_score = torch.matmul(query, torch.t(self.spatial_key))
        
        top_k_score, top_k_index = torch.topk(cosine_score, self.top_k, 1)
        
        ### For unsupervised training
        color_value_expand = torch.unsqueeze(torch.t(self.color_value), 0)
        color_value_expand = torch.cat([color_value_expand[:,:,idx] for idx in top_k_index], dim = 0)
        
        color_feat_expand = torch.unsqueeze(color_feat, 2)
        color_feat_expand = torch.cat([color_feat_expand for _ in range(self.top_k)], dim = 2)
        
        #color_similarity = self.KL_divergence(color_value_expand, color_feat_expand, 1)
        color_similarity = torch.sum(torch.mul(color_value_expand, color_feat_expand),dim=1)
        
        #loss_mask = color_similarity < color_thres
        loss_mask = color_similarity > color_thres
        loss_mask = loss_mask.float()
        
        pos_score, pos_index = torch.topk(torch.mul(top_k_score, loss_mask), 1, dim = 1)
        neg_score, neg_index = torch.topk(torch.mul(top_k_score, 1 - loss_mask), 1, dim = 1)
        
        loss = self._unsupervised_loss(pos_score, neg_score)
        
        return loss
    
    
    def memory_update(self, query, color_feat, color_thres, top_index):
        
        cosine_score = torch.matmul(query, torch.t(self.spatial_key))
        top1_score, top1_index = torch.topk(cosine_score, 1, dim = 1)
        top1_index = top1_index[:, 0]
        top1_feature = self.spatial_key[top1_index]
        top1_color_value = self.color_value[top1_index]
        
        #color_similarity1 = self.KL_divergence(top1_color_value, color_feat, 1)
        color_similarity = torch.sum(torch.mul(top1_color_value, color_feat),dim=1)
            
        #memory_mask = color_similarity < color_thres
        memory_mask = color_similarity > color_thres
        self.age = self.age + 1.0
        
        ## Case 1 update
        case_index = top1_index[memory_mask]
        self.spatial_key[case_index] = F.normalize(self.spatial_key[case_index] + query[memory_mask], dim = 1)
        self.age[case_index] = 0.0
        #if torch.sum(memory_mask).cpu().numpy()==1:
        #    print(top_index,'update',self.top_index[case_index],color_similarity)
        
        ## Case 2 replace
        memory_mask = 1.0 - memory_mask
        case_index = top1_index[memory_mask]
        
        random_noise = random_uniform((self.mem_size, 1), -self.age_noise, self.age_noise)[:, 0]
        random_noise = random_noise.to(self.device)
        age_with_noise = self.age + random_noise
        old_values, old_index = torch.topk(age_with_noise, len(case_index), dim=0)
        
        self.spatial_key[old_index] = query[memory_mask]
        self.color_value[old_index] = color_feat[memory_mask]
        #if torch.sum(memory_mask).cpu().numpy()==1:
        #    print(top_index[memory_mask],'replace',self.top_index[old_index],color_similarity)
        #pdb.set_trace()
        self.top_index[old_index] = top_index[memory_mask]
        self.age[old_index] = 0.0
        
        return torch.sum(memory_mask).cpu().numpy()==1 # for batch size 1, return number of replace
    
    def topk_feature(self, query, top_k = 1):
        _bs = query.size()[0]
        cosine_score = torch.matmul(query, torch.t(self.spatial_key))
        topk_score, topk_index = torch.topk(cosine_score, top_k, dim = 1)
        
        topk_feat = torch.cat([torch.unsqueeze(self.color_value[topk_index[i], :], dim = 0) for i in range(_bs)], dim = 0)
        topk_idx = torch.cat([torch.unsqueeze(self.top_index[topk_index[i]], dim = 0) for i in range(_bs)], dim = 0)
        
        return topk_feat, topk_idx, topk_index

    def get_feature(self, k, _bs):
        feat = torch.cat([torch.unsqueeze(self.color_value[k, :], dim = 0) for i in range(_bs)], dim = 0)
        return feat, self.top_index[k]

    
    def KL_divergence(self, a, b, dim, eps = 1e-8):
        
        b = b + eps
        log_val = torch.log10(torch.div(a, b))
        kl_div = torch.mul(a, log_val)
        kl_div = torch.sum(kl_div, dim = dim)
        
        return kl_div
        
        
    def _unsupervised_loss(self, pos_score, neg_score):
        
        hinge = torch.clamp(neg_score - pos_score + self.alpha, min = 0.0)
        loss = torch.mean(hinge)
        
        return loss
        

def random_uniform(shape, low, high):
    x = torch.rand(*shape)
    result = (high - low) * x + low
    
    return result