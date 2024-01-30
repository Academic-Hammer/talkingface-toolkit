import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_
from tqdm import tqdm, trange
from os import listdir, path
import numpy as np
import os, subprocess
import imageio
from glob import glob
import cv2

from talkingface.model.layers import Conv2d, Conv2dTranspose, nonorm_Conv2d
from talkingface.model.abstract_talkingface import AbstractTalkingFace
from talkingface.utils import ensure_dir
from talkingface.utils.dfrf_util.load_audface_multiid import load_audface_data
from talkingface.utils.dfrf_util.run_nerf_helpers_deform import *
from talkingface.utils.dfrf_util.run_nerf_deform import *


from natsort import natsorted


class dfrf(AbstractTalkingFace):
    def __init__(self, config):
        super(dfrf, self).__init__()
        self.config = config
            # Create nerf model
        parser = config_parser()
        self.args = parser.parse_args()
        if self.args.dataset_type == 'audface':
            if self.args.with_test == 1:
                self.images, self.poses, self.auds, self.bc_img, self.hwfcxy, self.lip_rects, self.torso_bcs, _ = \
                    load_audface_data(self.args.datadir, self.args.testskip, self.args.test_file, self.args.aud_file, need_lip=True, need_torso = self.args.need_torso, bc_type=self.args.bc_type)
                #images = np.zeros(1)
            else:
                self.images, self.poses, self.auds, self.bc_img, self.hwfcxy, self.sample_rects, self.i_split, self.id_num, self.lip_rects, self.torso_bcs = load_audface_data(
                    self.args.datadir, self.args.testskip, train_length = self.args.train_length, need_lip=True, need_torso = self.args.need_torso, bc_type=self.args.bc_type)

            #print('Loaded audface', images['0'].shape, hwfcxy, args.datadir)
            if self.args.with_test == 0:
                print('Loaded audface', self.images['0'].shape, self.hwfcxy, self.args.datadir)
                #all id has the same split, so this part can be shared
                self.i_train, self.i_val = self.i_split['0']
            else:
                print('Loaded audface', len(self.images), self.hwfcxy, self.args.datadir)
            near = self.args.near
            far = self.args.far
        else:
            print('Unknown dataset type', self.args.dataset_type, 'exiting')
            return

        # Cast intrinsics to right types
        self.H, self.W, self.focal, self.cx, self.cy = self.hwfcxy
        self.H, self.W = int(self.H), int(self.W)
        self.hwf = [self.H, self.W, self.focal]
        self.hwfcxy = [self.H, self.W, self.focal, self.cx, self.cy]

        self.intrinsic = np.array([[self.focal, 0., self.W / 2],
                            [0, self.focal, self.H / 2],
                            [0, 0, 1.]])
        self.intrinsic = torch.Tensor(self.intrinsic).to(device).float()

        self.render_kwargs_train, self.render_kwargs_test, self.start, self.grad_vars, self.optimizer, self.learned_codes, \
        self.AudNet_state, self.optimizer_aud_state, self.AudAttNet_state, self.optimizer_audatt_state, self.models = create_nerf(self.args)
        self.global_step = self.start

        self.AudNet = AudioNet(self.args.dim_aud, self.args.win_size).to(device)
        self.AudAttNet = AudioAttNet().to(device)
        self.optimizer_Aud = torch.optim.Adam(
            params=list(self.AudNet.parameters()), lr=self.args.lrate, betas=(0.9, 0.999))
        self.optimizer_AudAtt = torch.optim.Adam(
            params=list(self.AudAttNet.parameters()), lr=self.args.lrate, betas=(0.9, 0.999))

        if self.AudNet_state is not None:
            self.AudNet.load_state_dict(self.AudNet_state, strict=False)
        if self.optimizer_aud_state is not None:
            self.optimizer_Aud.load_state_dict(self.optimizer_aud_state)
        if self.AudAttNet_state is not None:
            self.AudAttNet.load_state_dict(self.AudAttNet_state, strict=False)
        if self.optimizer_audatt_state is not None:
            self.optimizer_AudAtt.load_state_dict(self.optimizer_audatt_state)
        bds_dict = {
            'near': near,
            'far': far,
        }

        self.render_kwargs_train.update(bds_dict)
        self.render_kwargs_test.update(bds_dict)
        self.N_iters = self.args.N_iters + 1

        print('Begin')
        print('TRAIN views are', self.i_train)
        print('VAL views are', self.i_val)
        self.start = self.start + 1
        self.basedir = self.args.basedir
        self.expname = self.args.expname
        self.expname_finetune = self.args.expname_finetune

    def calculate_loss(self, rgb, target_s, loss_translation, extras):
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss + self.args.L2loss_weight * loss_translation
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        return loss, psnr, psnr0

    def forward(self, i):
        if self.args.render_only or i == -1:
            print('RENDER ONLY')
            with torch.no_grad():
                images_refer, poses_refer, auds_refer, bc_img_refer, _ , lip_rects_refer, _, _ = \
                    load_audface_data(self.args.datadir, self.args.testskip, 'transforms_train.json', self.args.aud_file, need_lip=True, need_torso=False, bc_type=self.args.bc_type)

                images_refer = torch.cat([torch.Tensor(imageio.imread(images_refer[i])).cpu().unsqueeze(0) for i in
                                    range(len(images_refer))], 0).float()/255.0
                poses_refer = torch.Tensor(poses_refer).float().cpu()
                # Default is smoother render_poses path
                #the data loader return these: images, poses, auds, bc_img, hwfcxy
                bc_img = torch.Tensor(self.bc_img).to(device).float() / 255.0
                poses = torch.Tensor(self.poses).to(device).float()
                auds = torch.Tensor(self.auds).to(device).float()
                testsavedir = os.path.join(self.basedir, self.expname_finetune, 'renderonly_{}_{:06d}'.format(
                    'test' if self.args.render_test else 'path', self.start))
                os.makedirs(testsavedir, exist_ok=True)

                print('test poses shape', poses.shape)
                #select reference images for the test set
                if self.args.refer_from_train:
                    perm = [50,100,150,200]
                    perm = perm[0:self.args.num_reference_images]
                    attention_images = images_refer[perm].to(device)
                    attention_poses = poses_refer[perm, :3, :4].to(device)
                else:
                    perm = np.random.randint(images_refer.shape[0]-1, size=4).tolist()
                    attention_images_ = np.array(self.images)[perm]
                    attention_images = torch.cat([torch.Tensor(imageio.imread(i)).unsqueeze(0) for i in
                                            attention_images_], 0).float() / 255.0
                    attention_poses = poses[perm, :3, :4].to(device)

                auds_val = []
                if self.start < self.args.nosmo_iters:
                    auds_val = self.AudNet(auds)
                else:
                    print('Load the smooth audio for rendering!')
                    for i in range(poses.shape[0]):
                        smo_half_win = int(self.args.smo_size / 2)
                        left_i = i - smo_half_win
                        right_i = i + smo_half_win
                        pad_left, pad_right = 0, 0
                        if left_i < 0:
                            pad_left = -left_i
                            left_i = 0
                        if right_i > poses.shape[0]:
                            pad_right = right_i - poses.shape[0]
                            right_i = poses.shape[0]
                        auds_win = auds[left_i:right_i]
                        if pad_left > 0:
                            auds_win = torch.cat(
                                (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                        if pad_right > 0:
                            auds_win = torch.cat(
                                (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
                        auds_win = self.AudNet(auds_win)
                        #aud = auds_win[smo_half_win]
                        aud_smo = self.AudAttNet(auds_win)
                        auds_val.append(aud_smo)
                    auds_val = torch.stack(auds_val, 0)

                with torch.no_grad():
                    rgbs, disp, last_weight = render_path(self.args, self.torso_bcs, self.poses, auds_val, self.bc_img, self.hwfcxy, attention_poses,attention_images,
                                self.intrinsic, self.args.chunk, self.render_kwargs_test, gt_imgs=None, savedir=testsavedir, lip_rect=lip_rects_refer[perm])

                np.save(os.path.join(testsavedir, 'last_weight.npy'), last_weight)
                print('Done rendering', testsavedir)
                imageio.mimwrite(os.path.join(
                    testsavedir, 'video.mp4'), to8b(rgbs), fps=25, quality=8)
                return
        # Random from one image
        #1.Select a id for training
        select_id = np.random.choice(self.id_num)
        #bc_img_ = torch.Tensor(bc_img[str(select_id)]).to(device).float()/255.0
        poses_ = torch.Tensor(self.poses[str(select_id)]).to(device).float()
        auds_ = torch.Tensor(self.auds[str(select_id)]).to(device).float()
        i_train, i_val = self.i_split[str(select_id)]
        img_i = np.random.choice(i_train)
        image_path = self.images[str(select_id)][img_i]

        bc_img_ = torch.as_tensor(imageio.imread(self.torso_bcs[str(select_id)][img_i])).to(device).float()/255.0

        target = torch.as_tensor(imageio.imread(image_path)).to(device).float()/255.0
        pose = poses_[img_i, :3, :4]
        rect = self.sample_rects[str(select_id)][img_i]
        aud = auds_[img_i]

        #select the attention pose and image
        if self.args.select_nearest:
            current_poses = self.poses[str(select_id)][:, :3, :4]
            current_images = self.images[str(select_id)]  # top size was set at 4 for reflective ones
            current_images = torch.cat([torch.as_tensor(imageio.imread(current_images[i])).unsqueeze(0) for i in range(current_images.shape[0])], 0)
            current_images = current_images.float() / 255.0
            attention_poses, attention_images = get_similar_k(pose, current_poses, current_images, top_size=None, k = 20)
        else:
            i_train_left = np.delete(i_train, np.where(np.array(i_train) == img_i))
            perm = np.random.permutation(i_train_left)[:self.args.num_reference_images]#selete num_reference_images images from the training set as reference
            attention_images = self.images[str(select_id)][perm]
            attention_images = torch.cat([torch.as_tensor(imageio.imread(attention_images[i])).unsqueeze(0) for i in range(self.args.num_reference_images)],0)
            attention_images = attention_images.float()/255.0
            attention_poses = self.poses[str(select_id)][perm, :3, :4]
            lip_rect = torch.Tensor(self.lip_rects[str(select_id)][perm])

        attention_poses = torch.Tensor(attention_poses).to(device).float()
        if self.global_step >= self.args.nosmo_iters:
            smo_half_win = int(self.args.smo_size / 2)
            left_i = img_i - smo_half_win
            right_i = img_i + smo_half_win
            pad_left, pad_right = 0, 0
            if left_i < 0:
                pad_left = -left_i
                left_i = 0
            if right_i > i_train.shape[0]:
                pad_right = right_i-i_train.shape[0]
                right_i = i_train.shape[0]
            auds_win = auds_[left_i:right_i]
            if pad_left > 0:
                auds_win = torch.cat(
                    (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
            if pad_right > 0:
                auds_win = torch.cat(
                    (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
            auds_win = self.AudNet(auds_win)
            aud = auds_win[smo_half_win]
            aud_smo = self.AudAttNet(auds_win)
        else:
            aud = self.AudNet(aud.unsqueeze(0))

        N_rand = self.args.N_rand
        if N_rand is not None:
            rays_o, rays_d = get_rays(
                self.H, self.W, self.focal, torch.Tensor(pose), self.cx, self.cy)  # (H, W, 3), (H, W, 3)

            if i < self.args.precrop_iters:
                dH = int(self.H//2 * self.args.precrop_frac)
                dW = int(self.W//2 * self.args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(self.H//2 - dH, self.H//2 + dH - 1, 2*dH),
                        torch.linspace(self.W//2 - dW, self.W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == self.start:
                    print(
                        f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {self.args.precrop_iters}")
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(
                    0, self.H-1, self.H), torch.linspace(0, self.W-1, self.W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
            if self.args.sample_rate > 0:
                rect_inds = (coords[:, 0] >= rect[0]) & (
                    coords[:, 0] <= rect[0] + rect[2]) & (
                        coords[:, 1] >= rect[1]) & (
                            coords[:, 1] <= rect[1] + rect[3])
                coords_rect = coords[rect_inds]
                coords_norect = coords[~rect_inds]
                rect_num = int(N_rand*self.args.sample_rate)
                norect_num = N_rand - rect_num
                select_inds_rect = np.random.choice(
                    coords_rect.shape[0], size=[rect_num], replace=False)  # (N_rand,)
                # (N_rand, 2)
                select_coords_rect = coords_rect[select_inds_rect].long()
                select_inds_norect = np.random.choice(
                    coords_norect.shape[0], size=[norect_num], replace=False)  # (N_rand,)
                # (N_rand, 2)
                select_coords_norect = coords_norect[select_inds_norect].long(
                )
                select_coords = torch.cat(
                    (select_coords_rect, select_coords_norect), dim=0)
            else:
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()

            rays_o = rays_o[select_coords[:, 0],
                            select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0],
                            select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
            bc_rgb = bc_img_[select_coords[:, 0],
                            select_coords[:, 1]]

        #####  Core optimization loop  #####
        if self.global_step >= self.args.nosmo_iters:
            rgb, disp, acc, _, extras, loss_translation = render_dynamic_face_new(self.H, self.W, self.focal, self.cx, self.cy, chunk=self.args.chunk, rays=batch_rays,
                                                            aud_para=aud_smo, bc_rgb=bc_rgb,
                                                            verbose=i < 10, retraw=True, attention_images = attention_images,
                                                            attention_poses = attention_poses, intrinsic = self.intrinsic, render_pose=pose,lip_rect = lip_rect, **self.render_kwargs_train)
        else:
            rgb, disp, acc, _, extras, loss_translation = render_dynamic_face_new(self.H, self.W, self.focal, self.cx, self.cy, chunk=self.args.chunk, rays=batch_rays,
                                                            aud_para=aud, bc_rgb=bc_rgb,
                                                            verbose=i < 10, retraw=True, attention_images = attention_images,
                                                            attention_poses = attention_poses, intrinsic = self.intrinsic, render_pose=pose,lip_rect = lip_rect, **self.render_kwargs_train)

        self.optimizer.zero_grad()
        self.optimizer_Aud.zero_grad()
        self.optimizer_AudAtt.zero_grad()

        loss, psnr, psnr0 = self.calculate_loss(rgb, target_s, loss_translation, extras)

        loss.backward()

        self.optimizer.step()
        self.optimizer_Aud.step()
        if self.global_step >= self.args.nosmo_iters:
            self.optimizer_AudAtt.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = self.args.lrate_decay * 1500
        new_lrate = self.args.lrate * (decay_rate ** (self.global_step / decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lrate

        for param_group in self.optimizer_Aud.param_groups:
            param_group['lr'] = new_lrate

        for param_group in self.optimizer_AudAtt.param_groups:
            param_group['lr'] = new_lrate*5
        ################################

        if i % self.args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        self.global_step += 1
        return rgb 

    def predict(self):
        return self.forward(-1)

    def generate_batch(self):
        file_dict = {'generated_video': [], 'real_video': []}
        return file_dict

