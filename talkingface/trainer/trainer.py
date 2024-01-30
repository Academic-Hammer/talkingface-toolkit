import os

from logging import getLogger
from time import time
import torch.nn.functional as F
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import torch.cuda.amp as amp
from torch import nn
from pathlib import Path

from talkingface.utils import(
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger
)
from talkingface.evaluator import Evaluator

from talkingface.data.dataset.dfrf_dataset import dfrfDataset
from talkingface.utils.dfrf_util.load_audface_multiid import load_audface_data
from talkingface.utils.dfrf_util.run_nerf_helpers_deform import *
from talkingface.utils.dfrf_util.run_nerf_deform import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model
    
    def fit(self, train_data):
        r"""Train the model based on the train data."""
        raise NotImplementedError("Method [next] should be implemented.")

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data."""

        raise NotImplementedError("Method [next] should be implemented.")
    

class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in talkingface systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """
    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        # self.enable_amp = config["enable_amp"]
        # self.enable_scaler = torch.cuda.is_available() and config["enable_scaler"]

        # config for train 
        self.learner = config["learner"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.stopping_step = config["stopping_step"]
        self.test_batch_size = config["eval_batch_size"]
        self.gpu_available = torch.cuda.is_available() and config["use_gpu"]
        self.device = config["device"]
        self.checkpoint_dir = config["checkpoint_dir"]
        ensure_dir(self.checkpoint_dir)
        saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config["weight_decay"]
        self.start_epoch = 0
        self.cur_step = 0
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.evaluator = Evaluator(config)

        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None

    def _build_optimizer(self, **kwargs):
        params = kwargs.pop("params", self.model.parameters())
        learner = kwargs.pop("learner", self.learner)
        learning_rate = kwargs.pop("learning_rate", self.learning_rate)
        weight_decay = kwargs.pop("weight_decay", self.weight_decay)
        if (self.config["reg_weight"] and weight_decay and weight_decay * self.config["reg_weight"] > 0):
            self.logger.warning(
                "The parameters [weight_decay] and [reg_weight] are specified simultaneously, "
                "which may lead to double regularization."
            )

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == "sparse_adam":
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning(
                    "Sparse Adam cannot argument received argument [{weight_decay}]"
                )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            the averaged loss of this epoch
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss_dict = {}
        step = 0
        iter_data = (
            tqdm(
            train_data,
            total=len(train_data),
            ncols=None,
            )
            if show_progress
            else train_data
        )

        for batch_idx, interaction in enumerate(iter_data):
            self.optimizer.zero_grad()
            step += 1
            losses_dict = loss_func(interaction)
            loss = losses_dict["loss"]

            for key, value in losses_dict.items():
                if key in total_loss_dict:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] += value
                    # 如果键已经在总和字典中，累加当前值
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] += value.item()
                else:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] = value
                    # 否则，将当前值添加到字典中
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] = value.item()
            iter_data.set_description(set_color(f"train {epoch_idx} {losses_dict}", "pink"))

            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
        average_loss_dict = {}
        for key, value in total_loss_dict.items():
            average_loss_dict[key] = value/step

        return average_loss_dict

        

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data. Different from the evaluate, this is use for training.

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            loss
        """
        print('Valid for {} steps'.format(self.eval_steps))
        self.model.eval()
        total_loss_dict = {}
        iter_data = (
            tqdm(valid_data,
                total=len(valid_data),
                ncols=None,
            )
            if show_progress
            else valid_data
        )
        step = 0
        for batch_idx, batched_data in enumerate(iter_data):
            step += 1
            batched_data.to(self.device)    
            losses_dict = self.model.calculate_loss(batched_data, valid=True)
            for key, value in losses_dict.items():
                if key in total_loss_dict:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] += value
                    # 如果键已经在总和字典中，累加当前值
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] += value.item()
                else:
                    if not torch.is_tensor(value):
                        total_loss_dict[key] = value
                    # 否则，将当前值添加到字典中
                    else:
                        losses_dict[key] = value.item()
                        total_loss_dict[key] = value.item()
            iter_data.set_description(set_color(f"Valid {losses_dict}", "pink"))
        average_loss_dict = {}
        for key, value in total_loss_dict.items():
            average_loss_dict[key] = value/step

        return average_loss_dict
                


    
    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        saved_model_file = kwargs.pop("saved_model_file", self.saved_model_file)
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.model.state_dict(),
            "other_parameter": self.model.other_parameter(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file, pickle_protocol=4)
        if verbose:
            self.logger.info(
                set_color("Saving current", "blue") + f": {saved_model_file}"
            )

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.cur_step = checkpoint["cur_step"]
        # self.best_valid_score = checkpoint["best_valid_score"]

        # load architecture params from checkpoint
        if checkpoint["config"]["model"].lower() != self.config["model"].lower():
            self.logger.warning(
                "Architecture configuration given in config file is different from that of checkpoint. "
                "This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.load_other_parameter(checkpoint.get("other_parameter"))

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        message_output = "Checkpoint loaded. Resume training from epoch {}".format(
            self.start_epoch
        )
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config["loss_decimal_place"] or 4
        train_loss_output = (
            set_color(f"epoch {epoch_idx} training", "green")
            + " ["
            + set_color("time", "blue")
            + f": {e_time - s_time:.2f}s, "
        )
        # 遍历字典，格式化并添加每个损失项
        loss_items = [
            set_color(f"{key}", "blue") + f": {value:.{des}f}"
            for key, value in losses.items()
        ]
        # 将所有损失项连接成一个字符串，并与前面的输出拼接
        train_loss_output += ", ".join(loss_items)
        return train_loss_output + "]"

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            "learner": self.config["learner"],
            "learning_rate": self.config["learning_rate"],
            "train_batch_size": self.config["train_batch_size"],
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values()
            for parameter in parameters
        }.union({"model", "dataset", "config_files", "device"})
        # other model-specific hparam
        hparam_dict.update(
            {
                para: val
                for para, val in self.config.final_config_dict.items()
                if para not in unrecorded_parameter
            }
        )
        for k in hparam_dict:
            if hparam_dict[k] is not None and not isinstance(
                hparam_dict[k], (bool, str, float, int)
            ):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(
            hparam_dict, {"hparam/best_valid_result": best_valid_result}
        )

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                            If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
            best result
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        if not (self.config['resume_checkpoint_path'] == None ) and self.config['resume']:
            self.resume_checkpoint(self.config['resume_checkpoint_path'])
        
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss)
            
            if verbose:
                self.logger.info(train_loss_output)
            # self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_loss = self._valid_epoch(valid_data=valid_data, show_progress=show_progress)

                (self.best_valid_score, self.cur_step, stop_flag,update_flag,) = early_stopping(
                    valid_loss['loss'],
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()

                valid_loss_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_loss)
                )
                if verbose:
                    self.logger.info(valid_loss_output)

                
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_loss['loss']

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break
    @torch.no_grad()
    def evaluate(self, load_best_model=True, model_file=None):
        """
        Evaluate the model based on the test data.

        args: load_best_model: bool, whether to load the best model in the training process.
                model_file: str, the model file you want to evaluate.

        """
        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)
        self.model.eval()

        datadict = self.model.generate_batch()
        eval_result = self.evaluator.evaluate(datadict)
        self.logger.info(eval_result)

class dfrfTrainer(Trainer):
    def __init__(self, config, model):
        super(dfrfTrainer, self).__init__(config, model)
    
    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parser = config_parser()
        args = parser.parse_args()
        print(args.near, args.far)
        print(args)
        # Load data

        if args.dataset_type == 'audface':
            if args.with_test == 1:
                images, poses, auds, bc_img, hwfcxy, lip_rects, torso_bcs, _ = \
                    load_audface_data(args.datadir, args.testskip, args.test_file, args.aud_file, need_lip=True, need_torso = args.need_torso, bc_type=args.bc_type)
                #images = np.zeros(1)
            else:
                images, poses, auds, bc_img, hwfcxy, sample_rects, i_split, id_num, lip_rects, torso_bcs = load_audface_data(
                    args.datadir, args.testskip, train_length = args.train_length, need_lip=True, need_torso = args.need_torso, bc_type=args.bc_type)

            #print('Loaded audface', images['0'].shape, hwfcxy, args.datadir)
            if args.with_test == 0:
                print('Loaded audface', images['0'].shape, hwfcxy, args.datadir)
                #all id has the same split, so this part can be shared
                i_train, i_val = i_split['0']
            else:
                print('Loaded audface', len(images), hwfcxy, args.datadir)
            near = args.near
            far = args.far
        else:
            print('Unknown dataset type', args.dataset_type, 'exiting')
            return

        # Cast intrinsics to right types
        H, W, focal, cx, cy = hwfcxy
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        hwfcxy = [H, W, focal, cx, cy]

        intrinsic = np.array([[focal, 0., W / 2],
                            [0, focal, H / 2],
                            [0, 0, 1.]])
        intrinsic = torch.Tensor(intrinsic).to(device).float()

        # if args.render_test:
        #     render_poses = np.array(poses[i_test])

        # Create log dir and copy the config file
        basedir = args.basedir
        expname = args.expname
        expname_finetune = args.expname_finetune
        os.makedirs(os.path.join(basedir, expname_finetune), exist_ok=True)
        f = os.path.join(basedir, expname_finetune, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(basedir, expname_finetune, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())

        # Create nerf model
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, learned_codes, \
        AudNet_state, optimizer_aud_state, AudAttNet_state, optimizer_audatt_state, models = create_nerf(args)

        global_step = start

        AudNet = AudioNet(args.dim_aud, args.win_size).to(device)
        AudAttNet = AudioAttNet().to(device)
        optimizer_Aud = torch.optim.Adam(
            params=list(AudNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))
        optimizer_AudAtt = torch.optim.Adam(
            params=list(AudAttNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))

        if AudNet_state is not None:
            AudNet.load_state_dict(AudNet_state, strict=False)
        if optimizer_aud_state is not None:
            optimizer_Aud.load_state_dict(optimizer_aud_state)
        if AudAttNet_state is not None:
            AudAttNet.load_state_dict(AudAttNet_state, strict=False)
        if optimizer_audatt_state is not None:
            optimizer_AudAtt.load_state_dict(optimizer_audatt_state)
        bds_dict = {
            'near': near,
            'far': far,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        if args.render_only:
            print('RENDER ONLY')
            with torch.no_grad():
                images_refer, poses_refer, auds_refer, bc_img_refer, _ , lip_rects_refer, _, _ = \
                    load_audface_data(args.datadir, args.testskip, 'transforms_train.json', args.aud_file, need_lip=True, need_torso=False, bc_type=args.bc_type)

                images_refer = torch.cat([torch.Tensor(imageio.imread(images_refer[i])).cpu().unsqueeze(0) for i in
                                    range(len(images_refer))], 0).float()/255.0
                poses_refer = torch.Tensor(poses_refer).float().cpu()
                # Default is smoother render_poses path
                #the data loader return these: images, poses, auds, bc_img, hwfcxy
                bc_img = torch.Tensor(bc_img).to(device).float() / 255.0
                poses = torch.Tensor(poses).to(device).float()
                auds = torch.Tensor(auds).to(device).float()
                testsavedir = os.path.join(basedir, expname_finetune, 'renderonly_{}_{:06d}'.format(
                    'test' if args.render_test else 'path', start))
                os.makedirs(testsavedir, exist_ok=True)

                print('test poses shape', poses.shape)
                #select reference images for the test set
                if args.refer_from_train:
                    perm = [50,100,150,200]
                    perm = perm[0:args.num_reference_images]
                    attention_images = images_refer[perm].to(device)
                    attention_poses = poses_refer[perm, :3, :4].to(device)
                else:
                    perm = np.random.randint(images_refer.shape[0]-1, size=4).tolist()
                    attention_images_ = np.array(images)[perm]
                    attention_images = torch.cat([torch.Tensor(imageio.imread(i)).unsqueeze(0) for i in
                                            attention_images_], 0).float() / 255.0
                    attention_poses = poses[perm, :3, :4].to(device)

                auds_val = []
                if start < args.nosmo_iters:
                    auds_val = AudNet(auds)
                else:
                    print('Load the smooth audio for rendering!')
                    for i in range(poses.shape[0]):
                        smo_half_win = int(args.smo_size / 2)
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
                        auds_win = AudNet(auds_win)
                        #aud = auds_win[smo_half_win]
                        aud_smo = AudAttNet(auds_win)
                        auds_val.append(aud_smo)
                    auds_val = torch.stack(auds_val, 0)

                with torch.no_grad():
                    rgbs, disp, last_weight = render_path(args, torso_bcs, poses, auds_val, bc_img, hwfcxy, attention_poses,attention_images,
                                intrinsic, args.chunk, render_kwargs_test, gt_imgs=None, savedir=testsavedir, lip_rect=lip_rects_refer[perm])

                np.save(os.path.join(testsavedir, 'last_weight.npy'), last_weight)
                print('Done rendering', testsavedir)
                imageio.mimwrite(os.path.join(
                    testsavedir, 'video.mp4'), to8b(rgbs), fps=25, quality=8)
                return


        # Prepare raybatch tensor if batching random rays
        N_rand = args.N_rand
        print('N_rand', N_rand, 'no_batching',
            args.no_batching, 'sample_rate', args.sample_rate)
        use_batching = not args.no_batching

        if use_batching:
            # For random ray batching
            print('get rays')
            rays = np.stack([get_rays_np(H, W, focal, p, cx, cy)
                            for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
            print('done, concats')
            # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.concatenate([rays, images[:, None]], 1)
            # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
            rays_rgb = np.stack([rays_rgb[i]
                                for i in i_train], 0)  # train images only
            # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
            rays_rgb = rays_rgb.astype(np.float32)
            print('shuffle rays')
            np.random.shuffle(rays_rgb)

            print('done')
            i_batch = 0

        if use_batching:
            rays_rgb = torch.Tensor(rays_rgb).to(device)

        N_iters = args.N_iters + 1
        print('Begin')
        print('TRAIN views are', i_train)
        print('VAL views are', i_val)

        start = start + 1
        for i in trange(start, N_iters):
            time0 = time.time()
            # Sample random ray batch
            if use_batching:
                print("use_batching")
                # Random over all images
                batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
                batch = torch.transpose(batch, 0, 1)
                batch_rays, target_s = batch[:2], batch[2]

                i_batch += N_rand
                if i_batch >= rays_rgb.shape[0]:
                    print("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_rgb.shape[0])
                    rays_rgb = rays_rgb[rand_idx]
                    i_batch = 0

            else:
                # Random from one image
                #1.Select a id for training
                id_num = 1
                select_id = np.random.choice(id_num)
                #bc_img_ = torch.Tensor(bc_img[str(select_id)]).to(device).float()/255.0
                poses_ = torch.Tensor(poses[str(select_id)]).to(device).float()
                auds_ = torch.Tensor(auds[str(select_id)]).to(device).float()
                i_train, i_val = i_split[str(select_id)]
                img_i = np.random.choice(i_train)
                image_path = images[str(select_id)][img_i]

                bc_img_ = torch.as_tensor(imageio.imread(torso_bcs[str(select_id)][img_i])).to(device).float()/255.0

                target = torch.as_tensor(imageio.imread(image_path)).to(device).float()/255.0
                pose = poses_[img_i, :3, :4]
                rect = sample_rects[str(select_id)][img_i]
                aud = auds_[img_i]

                #select the attention pose and image
                if args.select_nearest:
                    current_poses = poses[str(select_id)][:, :3, :4]
                    current_images = images[str(select_id)]  # top size was set at 4 for reflective ones
                    current_images = torch.cat([torch.as_tensor(imageio.imread(current_images[i])).unsqueeze(0) for i in range(current_images.shape[0])], 0)
                    current_images = current_images.float() / 255.0
                    attention_poses, attention_images = get_similar_k(pose, current_poses, current_images, top_size=None, k = 20)
                else:
                    i_train_left = np.delete(i_train, np.where(np.array(i_train) == img_i))
                    perm = np.random.permutation(i_train_left)[:args.num_reference_images]#selete num_reference_images images from the training set as reference
                    attention_images = images[str(select_id)][perm]
                    attention_images = torch.cat([torch.as_tensor(imageio.imread(attention_images[i])).unsqueeze(0) for i in range(args.num_reference_images)],0)
                    attention_images = attention_images.float()/255.0
                    attention_poses = poses[str(select_id)][perm, :3, :4]
                    lip_rect = torch.Tensor(lip_rects[str(select_id)][perm])

                attention_poses = torch.Tensor(attention_poses).to(device).float()
                if global_step >= args.nosmo_iters:
                    smo_half_win = int(args.smo_size / 2)
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
                    auds_win = AudNet(auds_win)
                    aud = auds_win[smo_half_win]
                    aud_smo = AudAttNet(auds_win)
                else:
                    aud = AudNet(aud.unsqueeze(0))
                if N_rand is not None:
                    rays_o, rays_d = get_rays(
                        H, W, focal, torch.Tensor(pose), cx, cy)  # (H, W, 3), (H, W, 3)

                    if i < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
                        if i == start:
                            print(
                                f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(
                            0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                    if args.sample_rate > 0:
                        rect_inds = (coords[:, 0] >= rect[0]) & (
                            coords[:, 0] <= rect[0] + rect[2]) & (
                                coords[:, 1] >= rect[1]) & (
                                    coords[:, 1] <= rect[1] + rect[3])
                        coords_rect = coords[rect_inds]
                        coords_norect = coords[~rect_inds]
                        rect_num = int(N_rand*args.sample_rate)
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
            if global_step >= args.nosmo_iters:
                rgb, disp, acc, _, extras, loss_translation = render_dynamic_face_new(H, W, focal, cx, cy, chunk=args.chunk, rays=batch_rays,
                                                                aud_para=aud_smo, bc_rgb=bc_rgb,
                                                                verbose=i < 10, retraw=True, attention_images = attention_images,
                                                                attention_poses = attention_poses, intrinsic = intrinsic, render_pose=pose,lip_rect = lip_rect, **render_kwargs_train)
            else:
                rgb, disp, acc, _, extras, loss_translation = render_dynamic_face_new(H, W, focal, cx, cy, chunk=args.chunk, rays=batch_rays,
                                                                aud_para=aud, bc_rgb=bc_rgb,
                                                                verbose=i < 10, retraw=True, attention_images = attention_images,
                                                                attention_poses = attention_poses, intrinsic = intrinsic, render_pose=pose,lip_rect = lip_rect, **render_kwargs_train)

            optimizer.zero_grad()
            optimizer_Aud.zero_grad()
            optimizer_AudAtt.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss + args.L2loss_weight * loss_translation
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            loss.backward()
            optimizer.step()
            optimizer_Aud.step()
            if global_step >= args.nosmo_iters:
                optimizer_AudAtt.step()
            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1500
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            for param_group in optimizer_Aud.param_groups:
                param_group['lr'] = new_lrate

            for param_group in optimizer_AudAtt.param_groups:
                param_group['lr'] = new_lrate*5
            ################################
            dt = time.time()-time0

            # Rest is logging
            if i % args.i_weights == 0:
                path = os.path.join(basedir, expname_finetune, '{:06d}_head.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'network_audnet_state_dict': AudNet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_aud_state_dict': optimizer_Aud.state_dict(),
                    'network_audattnet_state_dict': AudAttNet.state_dict(),
                    'optimizer_audatt_state_dict': optimizer_AudAtt.state_dict(),
                    'unet_state_dict': render_kwargs_train['feature_extractor'].state_dict(),
                    'attention_state_dict': models['attention_model'].state_dict(),
                    'position_warp_state_dict':render_kwargs_train['position_warp_model'].state_dict(),
                }, path)
                print('Saved checkpoints at', path)

            
            if i % args.i_print == 0:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

            global_step += 1
