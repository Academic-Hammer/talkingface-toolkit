import os

from logging import getLogger
from time import time
import dlib, json, subprocess
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
from talkingface.data.dataprocess.wav2lip_process import Wav2LipAudio
from talkingface.evaluator import Evaluator


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



class Wav2LipTrainer(Trainer):
    def __init__(self, config, model):
        super(Wav2LipTrainer, self).__init__(config, model)

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

        
    
    def _valid_epoch(self, valid_data, loss_func=None, show_progress=False):
        print('Valid'.format(self.eval_step))
        self.model.eval()
        total_loss_dict = {}
        iter_data = (
            tqdm(valid_data,
                total=len(valid_data),
                ncols=None,
                desc=set_color("Valid", "pink")
            )
            if show_progress
            else valid_data
        )
        step = 0
        for batch_idx, batched_data in enumerate(iter_data):
            step += 1
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
        average_loss_dict = {}
        for key, value in total_loss_dict.items():
            average_loss_dict[key] = value/step
        if losses_dict["sync_loss"] < .75:
            self.model.config["syncnet_wt"] = 0.01
        return average_loss_dict


class EAMMTrainer(Trainer):
    import matplotlib

    matplotlib.use("Agg")
    import yaml
    from argparse import ArgumentParser
    import skimage
    import imageio
    import skimage.transform as st
    from talkingface.utils.filter1 import OneEuroFilter
    import torch.utils

    from torch.autograd import Variable
    from talkingface.utils.augmentation import AllAugmentationTransform
    
    from talkingface.model.audio_driven_talkingface.eamm_modules.generator import OcclusionAwareGenerator
    from talkingface.model.audio_driven_talkingface.eamm_modules.keypoint_detector import KPDetector, KPDetector_a
    from talkingface.model.audio_driven_talkingface.eamm_modules.util import AT_net, Emotion_k, Emotion_map, AT_net2

    from scipy.spatial import ConvexHull

    import python_speech_features
    import cv2
    import librosa
    from skimage import transform as tf
    import itertools
    from talkingface.utils.eamm_logger import Logger
    from torch.optim.lr_scheduler import MultiStepLR
    from talkingface.model.audio_driven_talkingface.eamm_modules.model import DiscriminatorFullModel, TrainPart1Model, TrainPart2Model

    def __init__(self, config, model):
        super(EAMMTrainer, self).__init__(config, model)
        self.opt = config
        self.model = model
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("checkpoints/EAMM/shape_predictor_68_face_landmarks.dat")
        if self.opt['train']:
            self.train_params = {
                'jaco_net': self.opt['train_jaco_net'],
                'ldmark': self.opt['train_ldmark'],
                'generator': self.opt['train_generator'],
                'num_epochs': self.opt['train_num_epochs'],
                'train_num_repeats': self.opt['train_num_repeats'],
                'epoch_milestones': self.opt['train_epoch_milestones'],
                'lr_generator': self.opt['train_lr_generator'],
                'lr_discriminator': self.opt['train_lr_discriminator'],
                'lr_kp_detector': self.opt['train_lr_kp_detector'],
                'lr_audio_feature': self.opt['train_lr_audio_feature'],
                'batch_size': self.opt['train_batch_size'],
                'scales': self.opt['train_scales'],
                'checkpoint_freq': self.opt['train_checkpoint_freq'],
                'transform_params': {
                    'sigma_affine': self.opt['train_transform_sigma_affine'],
                    'sigma_tps': self.opt['train_transform_sigma_tps'],
                    'points_tps': self.opt['train_transform_points_tps'],
                },
                'loss_weights': {
                    'generator_gan': self.opt['train_loss_weights_generator_gan'],
                    'discriminator_gan': self.opt['train_loss_weights_discriminator_gan'],
                    'feature_matching': self.opt['train_loss_weights_feature_matching'],
                    'perceptual': self.opt['train_loss_weights_perceptual'],
                    'equivariance_value': self.opt['train_loss_weights_equivariance_value'],
                    'equivariance_jacobian': self.opt['train_loss_weights_equivariance_jacobian'],
                    'audio': self.opt['train_loss_weights_audio'],
                },
            }
            self._init_train()
    
    def _init_train(self):
        if self.opt['mode'] == 'train_part1':
            self._init_train_part1()
        elif self.opt['mode'] == 'train_part1_fine_tune':
            self._init_train_part1_fine_tune()
        elif self.opt['mode'] == 'train_part2':
            self._init_train_part2()
            
        self.step = 0
        self.train_itr = 0
        self.test_itr = 0
    
    def _init_train_part1(self):
        self.optimizer_audio_feature = torch.optim.Adam(
            self.itertools.chain(self.model.audio_feature.parameters(), self.model.kp_detector_a.parameters()),
            lr=self.opt["train_lr_audio_feature"],
            betas=(0.5, 0.999),
        )
        if self.opt['checkpoint'] is not None:
            self.start_epoch = self.Logger.load_cpk(
                self.opt['checkpoint'],
                self.model.generator,
                self.model.discriminator,
                self.model.kp_detector,
                self.model.audio_feature,
                None,
                None,
                None, #!
                None, #!
            )
        if self.opt['audio_checkpoint'] is not None:
            pretrain = torch.load(self.opt['audio_checkpoint'])
            self.model.kp_detector_a.load_state_dict(pretrain["kp_detector_a"])
            self.model.audio_feature.load_state_dict(pretrain["audio_feature"])
            self.optimizer_audio_feature.load_state_dict(pretrain["optimizer_audio_feature"])
            self.start_epoch = pretrain["epoch"]
        else:
            self.start_epoch = 0
            
        self.scheduler_audio_feature = self.MultiStepLR(
            self.optimizer_audio_feature,
            self.opt["train_epoch_milestones"],
            gamma=0.1,
            last_epoch=-1 + self.start_epoch * (self.opt["train_lr_audio_feature"] != 0),
        )
        
        self.generator_full = self.TrainPart1Model(
            self.model.kp_detector,
            self.model.kp_detector_a,
            self.model.audio_feature,
            self.model.generator,
            self.model.discriminator,
            self.train_params,
            self.opt['device_ids'],
        )
        self.discriminator_full = self.DiscriminatorFullModel(
            self.model.kp_detector, self.model.generator, self.model.discriminator, self.train_params
        )
        
        if self.gpu_available:
            self.generator_full = self.generator_full.to(self.device)
            self.discriminator_full = self.discriminator_full.to(self.device)
        else:
            self.generator_full = self.generator_full.cpu()
            self.discriminator_full = self.discriminator_full.cpu()

    def _init_train_part1_fine_tune(self):
        self.optimizer_generator = torch.optim.Adam(
            self.model.generator.parameters(), lr=self.train_params["lr_generator"], betas=(0.5, 0.999)
        )
        self.optimizer_discriminator = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.train_params["lr_discriminator"],
            betas=(0.5, 0.999),
        )
        self.optimizer_audio_feature = torch.optim.Adam(
            self.itertools.chain(self.model.audio_feature.parameters(), self.model.kp_detector_a.parameters()),
            lr=self.train_params["lr_audio_feature"],
            betas=(0.5, 0.999),
        )

        if self.opt['checkpoint'] is not None:
            self.start_epoch = self.Logger.load_cpk(
                self.opt['checkpoint'],
                self.model.generator,
                self.model.discriminator,
                self.model.kp_detector,
                self.model.audio_feature,
                self.optimizer_generator,
                self.optimizer_discriminator,
                None, #!
                None if self.train_params["lr_audio_feature"] == 0 else self.optimizer_audio_feature,
            )
        if self.opt['audio_checkpoint'] is not None:
            pretrain = torch.load(self.opt['audio_checkpoint'])
            self.model.kp_detector_a.load_state_dict(pretrain["kp_detector_a"])
            self.model.audio_feature.load_state_dict(pretrain["audio_feature"])
            self.optimizer_audio_feature.load_state_dict(pretrain["optimizer_audio_feature"])
            self.start_epoch = pretrain["epoch"]
        else:
            self.start_epoch = 0

        self.scheduler_generator = self.MultiStepLR(
            self.optimizer_generator,
            self.train_params["epoch_milestones"],
            gamma=0.1,
            last_epoch=self.start_epoch - 1,
        )
        self.scheduler_discriminator = self.MultiStepLR(
            self.optimizer_discriminator,
            self.train_params["epoch_milestones"],
            gamma=0.1,
            last_epoch=self.start_epoch - 1,
        )
        self.scheduler_audio_feature = self.MultiStepLR(
            self.optimizer_audio_feature,
            self.train_params["epoch_milestones"],
            gamma=0.1,
            last_epoch=-1 + self.start_epoch * (self.train_params["lr_audio_feature"] != 0),
        )
        self.generator_full = self.TrainPart1Model(
            self.model.kp_detector,
            self.model.kp_detector_a,
            self.model.audio_feature,
            self.model.generator,
            self.model.discriminator,
            self.train_params,
            self.opt['device_ids'],
        )
        self.discriminator_full = self.DiscriminatorFullModel(
            self.model.kp_detector, self.model.generator, self.model.discriminator, self.train_params
        )
        
        if self.gpu_available:
            self.generator_full = self.generator_full.to(self.device)
            self.discriminator_full = self.discriminator_full.to(self.device)
        else:
            self.generator_full = self.generator_full.cpu()
            self.discriminator_full = self.discriminator_full.cpu()
    
    def _init_train_part2(self):
        self.optimizer_emo_detector = torch.optim.Adam(
            self.model.emo_detector.parameters(),
            lr=self.train_params["lr_audio_feature"],
            betas=(0.5, 0.999),
        )
        self.optimizer_generator = torch.optim.Adam(
            self.model.generator.parameters(),
            lr=self.train_params["lr_generator"],
            betas=(0.5, 0.999),
        )
        self.optimizer_discriminator = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.train_params["lr_discriminator"],
            betas=(0.5, 0.999),
        )
        self.optimizer_audio_feature = torch.optim.Adam(
            self.itertools.chain(self.model.audio_feature.parameters(), self.model.kp_detector_a.parameters()),
            lr=self.train_params["lr_audio_feature"],
            betas=(0.5, 0.999),
        )

        if self.opt['checkpoint'] is not None:
            start_epoch = self.Logger.load_cpk(
                self.opt['checkpoint'],
                self.model.generator,
                self.model.discriminator,
                self.model.kp_detector,
                self.model.audio_feature,
                self.optimizer_generator,
                self.optimizer_discriminator,
                None, #!
                None if self.train_params["lr_audio_feature"] == 0 else self.optimizer_audio_feature,
            )
        if self.opt['emo_checkpoint'] is not None:
            pretrain = torch.load(self.opt['emo_checkpoint'])
            tgt_state = self.emo_detector.state_dict()
            strip = "module."
            if "emo_detector" in pretrain:
                self.emo_detector.load_state_dict(pretrain["emo_detector"])
                self.optimizer_emo_detector.load_state_dict(pretrain["optimizer_emo_detector"])
            for name, param in pretrain.items():
                if isinstance(param, nn.Parameter):
                    param = param.data
                if strip is not None and name.startswith(strip):
                    name = name[len(strip) :]
                if name not in tgt_state:
                    continue
                tgt_state[name].copy_(param)
                print(name)
        if self.opt['audio_checkpoint'] is not None:
            pretrain = torch.load(self.opt['audio_checkpoint'])
            self.kp_detector_a.load_state_dict(pretrain["kp_detector_a"])
            self.audio_feature.load_state_dict(pretrain["audio_feature"])
            self.optimizer_audio_feature.load_state_dict(pretrain["optimizer_audio_feature"])
            if "emo_detector" in pretrain:
                self.emo_detector.load_state_dict(pretrain["emo_detector"])
                self.optimizer_emo_detector.load_state_dict(pretrain["optimizer_emo_detector"])
            self.start_epoch = pretrain["epoch"]
        else:
            self.start_epoch = 0

        self.scheduler_emo_detector = self.MultiStepLR(
            self.optimizer_emo_detector,
            self.train_params["epoch_milestones"],
            gamma=0.1,
            last_epoch=-1 + start_epoch * (self.train_params["lr_audio_feature"] != 0),
        )
        self.generator_full = self.TrainPart2Model(
            self.model.kp_detector,
            self.model.emo_detector,
            self.model.kp_detector_a,
            self.model.audio_feature,
            self.model.generator,
            self.model.discriminator,
            self.train_params,
            self.opt['device_ids'],
        )
        self.discriminator_full = self.DiscriminatorFullModel(
            self.model.kp_detector, self.model.generator, self.model.discriminator, self.train_params
        )
        if self.gpu_available:
            self.generator_full = self.generator_full.to(self.device)
            self.discriminator_full = self.discriminator_full.to(self.device)
        else:
            self.generator_full = self.generator_full.cpu()
            self.discriminator_full = self.discriminator_full.cpu()

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if self.opt['mode'] == 'train_part1':
            return self._train_part1(train_data, epoch_idx, loss_func, show_progress)
        elif self.opt['mode'] == 'train_part1_fine_tune':
            return self._train_part1_fine_tune(train_data, epoch_idx, loss_func, show_progress)
        elif self.opt['mode'] == 'train_part2':
            return self._train_part2(train_data, epoch_idx, loss_func, show_progress)
    
    def _train_part1(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        for x in train_data:
            losses_generator, generated = self.generator_full(x)
            loss_values = [val.mean() for val in losses_generator.values()]
            loss = sum(loss_values)

            self.tensorboard.add_scalar("Train", loss, self.train_itr)
            self.tensorboard.add_scalar("Train_value", loss_values[0], self.train_itr)
            self.tensorboard.add_scalar("Train_heatmap", loss_values[1], self.train_itr)
            self.tensorboard.add_scalar("Train_jacobian", loss_values[2], self.train_itr)

            self.train_itr += 1
            loss.backward()
            self.optimizer_audio_feature.step()
            self.optimizer_audio_feature.zero_grad()

            losses = {
                key: value.mean().detach().data.cpu().numpy()
                for key, value in losses_generator.items()
            }

            self.step += 1
        
        self.scheduler_audio_feature.step()
        return losses
    
    def _train_part1_fine_tune(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        for x in train_data:
            losses_generator, generated = self.generator_full(x)
            loss_values = [val.mean() for val in losses_generator.values()]
            loss = sum(loss_values)

            self.tensorboard.add_scalar("Train", loss, self.train_itr)
            self.tensorboard.add_scalar("Train_value", loss_values[0], self.train_itr)
            self.tensorboard.add_scalar("Train_heatmap", loss_values[1], self.train_itr)
            self.tensorboard.add_scalar("Train_jacobian", loss_values[2], self.train_itr)
            self.tensorboard.add_scalar("Train_perceptual", loss_values[3], self.train_itr)

            self.train_itr += 1
            loss.backward()

            self.optimizer_audio_feature.step()
            self.optimizer_audio_feature.zero_grad()
            self.optimizer_generator.step()
            self.optimizer_generator.zero_grad()
            if self.train_params["loss_weights"]["discriminator_gan"] != 0:
                self.optimizer_discriminator.zero_grad()
            else:
                losses_discriminator = {}

            losses_generator.update(losses_discriminator)
            losses = {
                key: value.mean().detach().data.cpu().numpy()
                for key, value in losses_generator.items()
            }
            self.step += 1

        self.scheduler_generator.step()
        self.scheduler_discriminator.step()
        self.scheduler_audio_feature.step()
        
        return losses
    
    def _train_part2(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        for x in train_data:
            losses_generator, generated = self.generator_full(x)
            loss_values = [val.mean() for val in losses_generator.values()]
            loss = sum(loss_values)

            self.tensorboard.add_scalar("Train", loss, self.train_itr)
            self.tensorboard.add_scalar("Train_value", loss_values[0], self.train_itr)
            self.tensorboard.add_scalar("Train_jacobian", loss_values[1], self.train_itr)
            self.tensorboard.add_scalar("Train_classify", loss_values[2], self.train_itr)

            self.train_itr += 1
            loss.backward()

            self.optimizer_emo_detector.step()
            self.optimizer_emo_detector.zero_grad()

            losses = {
                key: value.mean().detach().data.cpu().numpy()
                for key, value in losses_generator.items()
            }
            self.step += 1

        self.scheduler_emo_detector.step()
        return losses

    def _valid_epoch(self, valid_data, loss_func=None, show_progress=False):
        if self.opt['mode'] == 'train_part1':
            return self._valid_part1(valid_data, loss_func, show_progress)
        elif self.opt['mode'] == 'train_part1_fine_tune':
            return self._valid_part1_fine_tune(valid_data, loss_func, show_progress)
        elif self.opt['mode'] == 'train_part2':
            return self._valid_part2(valid_data, loss_func, show_progress)

    def _valid_part1(self, valid_data, loss_func=None, show_progress=False):
        for x in valid_data:
            with torch.no_grad():
                losses_generator, generated = self.generator_full(x)
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                self.tensorboard.add_scalar("Test", loss, self.test_itr)
                self.tensorboard.add_scalar("Test_value", loss_values[0], self.test_itr)
                self.tensorboard.add_scalar("Test_heatmap", loss_values[1], self.test_itr)
                self.tensorboard.add_scalar("Test_jacobian", loss_values[2], self.test_itr)

                self.test_itr += 1
                losses = {
                    key: value.mean().detach().data.cpu().numpy()
                    for key, value in losses_generator.items()
                }
                
        return losses
  
    def _valid_part1_fine_tune(self, valid_data, loss_func=None, show_progress=False):
        for x in valid_data:
            with torch.no_grad():
                losses_generator, generated = self.generator_full(x)
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                self.tensorboard.add_scalar("Test", loss, self.test_itr)
                self.tensorboard.add_scalar("Test_value", loss_values[0], self.test_itr)
                self.tensorboard.add_scalar("Test_heatmap", loss_values[1], self.test_itr)
                self.tensorboard.add_scalar("Test_jacobian", loss_values[2], self.test_itr)
                self.tensorboard.add_scalar("Test_perceptual", loss_values[3], self.test_itr)

                self.test_itr += 1
                losses = {
                    key: value.mean().detach().data.cpu().numpy()
                    for key, value in losses_generator.items()
                }
                
        return losses
    
    def _valid_part2(self, valid_data, loss_func=None, show_progress=False):
        for x in valid_data:
            with torch.no_grad():
                losses_generator, generated = self.generator_full(x)
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                self.tensorboard.add_scalar("Test", loss, self.test_itr)
                self.tensorboard.add_scalar("Test_value", loss_values[0], self.test_itr)
                self.tensorboard.add_scalar("Test_jacobian", loss_values[1], self.test_itr)
                self.tensorboard.add_scalar("Test_classify", loss_values[2], self.test_itr)

                self.test_itr += 1
                losses = {
                    key: value.mean().detach().data.cpu().numpy()
                    for key, value in losses_generator.items()
                }
                
        return losses

    def load_checkpoints(
        self, opt, checkpoint_path, audio_checkpoint_path, emo_checkpoint_path, cpu=False
    ):
        """
        load checkpoints
        """
        with open(opt['config']) as f:
            config = self.yaml.load(f, Loader=self.yaml.FullLoader)

        generator = self.OcclusionAwareGenerator(
            **config["model_params"]["generator_params"],
            **config["model_params"]["common_params"]
        )
        if not cpu:
            generator.cuda()

        kp_detector = self.KPDetector(
            **config["model_params"]["kp_detector_params"],
            **config["model_params"]["common_params"]
        )
        if not cpu:
            kp_detector.cuda()

        kp_detector_a = self.KPDetector_a(
            **config["model_params"]["kp_detector_params"],
            **config["model_params"]["audio_params"]
        )

        audio_feature = self.AT_net2()
        if opt['type'].startswith("linear"):
            emo_detector = self.Emotion_k(
                block_expansion=32,
                num_channels=3,
                max_features=1024,
                num_blocks=5,
                scale_factor=0.25,
                num_classes=8,
            )
        elif opt['type'].startswith("map"):
            emo_detector = self.Emotion_map(
                block_expansion=32,
                num_channels=3,
                max_features=1024,
                num_blocks=5,
                scale_factor=0.25,
                num_classes=8,
            )
        if not cpu:
            kp_detector_a.cuda()
            audio_feature.cuda()
            emo_detector.cuda()

        if cpu:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            audio_checkpoint = torch.load(
                audio_checkpoint_path, map_location=torch.device("cpu")
            )
            emo_checkpoint = torch.load(
                emo_checkpoint_path, map_location=torch.device("cpu")
            )
        else:
            checkpoint = torch.load(checkpoint_path)
            audio_checkpoint = torch.load(audio_checkpoint_path)
            emo_checkpoint = torch.load(emo_checkpoint_path)

        generator.load_state_dict(checkpoint["generator"])
        kp_detector.load_state_dict(checkpoint["kp_detector"])
        audio_feature.load_state_dict(audio_checkpoint["audio_feature"])
        kp_detector_a.load_state_dict(audio_checkpoint["kp_detector_a"])
        emo_detector.load_state_dict(emo_checkpoint["emo_detector"])

        if not cpu:
            generator = generator.cuda()
            kp_detector = kp_detector.cuda()
            audio_feature = audio_feature.cuda()
            kp_detector_a = kp_detector_a.cuda()
            emo_detector = emo_detector.cuda()
        else:
            generator = generator.cpu()
            kp_detector = kp_detector.cpu()
            audio_feature = audio_feature.cpu()
            kp_detector_a = kp_detector_a.cpu()
            emo_detector = emo_detector.cpu()

        generator.eval()
        kp_detector.eval()
        audio_feature.eval()
        kp_detector_a.eval()
        emo_detector.eval()
        return generator, kp_detector, kp_detector_a, audio_feature, emo_detector

    def normalize_kp(
        self,
        kp_source,
        kp_driving,
        kp_driving_initial,
        adapt_movement_scale=False,
        use_relative_movement=False,
        use_relative_jacobian=False,
    ):
        """
        normalize keypoints.
        """
        if adapt_movement_scale:
            source_area = self.ConvexHull(kp_source["value"][0].data.cpu().numpy()).volume
            driving_area = self.ConvexHull(
                kp_driving_initial["value"][0].data.cpu().numpy()
            ).volume
            adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
        else:
            adapt_movement_scale = 1

        kp_new = {k: v for k, v in kp_driving.items()}

        if use_relative_movement:
            kp_value_diff = kp_driving["value"] - kp_driving_initial["value"]
            kp_value_diff *= adapt_movement_scale
            kp_new["value"] = kp_value_diff + kp_source["value"]

            if use_relative_jacobian:
                jacobian_diff = torch.matmul(
                    kp_driving["jacobian"], torch.inverse(kp_driving_initial["jacobian"])
                )
                kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source["jacobian"])

        return kp_new

    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def get_aligned_image(self, driving_video, opt):
        """
        emotion video also crop centering and resize to 256 x 256.
        """
        aligned_array = []

        video_array = np.array(driving_video)
        source_image = video_array[0]
        source_image = np.array(source_image * 255, dtype=np.uint8)
        gray = self.cv2.cvtColor(source_image, self.cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)  # detect human face
        for i, rect in enumerate(rects):
            template = self.predictor(gray, rect)  # detect 68 points
            template = self.shape_to_np(template)

        if opt['emotion'] == "surprised" or opt['emotion'] == "fear":
            template = template - [0, 10]
        for i in range(len(video_array)):
            image = np.array(video_array[i] * 255, dtype=np.uint8)
            gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 1)  # detect human face
            for j, rect in enumerate(rects):
                shape = self.predictor(gray, rect)  # detect 68 points
                shape = self.shape_to_np(shape)

            pts2 = np.float32(template[:35, :])
            pts1 = np.float32(shape[:35, :])  # eye and nose

            tform = self.tf.SimilarityTransform()
            tform.estimate(
                pts2, pts1
            )  # Set the transformation matrix with the explicit parameters.
            dst = self.tf.warp(image, tform, output_shape=(256, 256))

            dst = np.array(dst, dtype=np.float32)
            aligned_array.append(dst)

        return aligned_array

    def get_transformed_image(self, driving_video, opt):
        """
        augmentation for emotion images.
        """
        video_array = np.array(driving_video)
        with open(opt['config']) as f:
            config = self.yaml.load(f, Loader=self.yaml.FullLoader)
        transformations = self.AllAugmentationTransform(
            **config["dataset_params"]["augmentation_params"]
        )
        transformed_array = transformations(video_array)
        return transformed_array

    def make_animation_smooth(
        self,
        source_image,
        driving_video,
        transformed_video,
        deco_out,
        kp_loss,
        generator,
        kp_detector,
        kp_detector_a,
        emo_detector,
        opt,
        relative=True,
        adapt_movement_scale=True,
        cpu=False,
    ):
        """
        generate target images.
        """
        with torch.no_grad():
            predictions = []

            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(
                0, 3, 1, 2
            )

            if not cpu:
                source = source.cuda()
            else:
                source = source.cpu()

            driving = torch.tensor(
                np.array(driving_video)[np.newaxis].astype(np.float32)
            ).permute(0, 4, 1, 2, 3)
            transformed_driving = torch.tensor(
                np.array(transformed_video)[np.newaxis].astype(np.float32)
            ).permute(0, 4, 1, 2, 3)

            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector_a(deco_out[:, 0])

            emo_driving_all = []
            features = []
            kp_driving_all = []
            for frame_idx in tqdm(range(len(deco_out[0]))):
                driving_frame = driving[:, :, frame_idx]
                transformed_frame = transformed_driving[:, :, frame_idx]
                if not cpu:
                    driving_frame = driving_frame.cuda()
                    transformed_frame = transformed_frame.cuda()
                else:
                    driving_frame = driving_frame.cpu()
                    transformed_frame = transformed_frame.cpu()
                kp_driving = kp_detector_a(deco_out[:, frame_idx])
                kp_driving_all.append(kp_driving)
                if opt['add_emo']:
                    value = kp_driving["value"]
                    jacobian = kp_driving["jacobian"]
                    if opt['type'] == "linear_3":
                        emo_driving, _ = emo_detector(transformed_frame, value, jacobian)
                        features.append(
                            emo_detector.feature(transformed_frame).data.cpu().numpy()
                        )

                    emo_driving_all.append(emo_driving)
            features = np.array(features)
            if opt['add_emo']:
                one_euro_filter_v = self.OneEuroFilter(
                    mincutoff=1, beta=0.2, dcutoff=1.0, freq=100
                )  # 1 0.4
                one_euro_filter_j = self.OneEuroFilter(
                    mincutoff=1, beta=0.2, dcutoff=1.0, freq=100
                )  # 1 0.4

                for j in range(len(emo_driving_all)):
                    emo_driving_all[j]["value"] = (
                        one_euro_filter_v.process(emo_driving_all[j]["value"].cpu() * 100)
                        / 100
                    )
                    if not cpu:
                        emo_driving_all[j]["value"] = emo_driving_all[j]["value"].cuda()
                    emo_driving_all[j]["jacobian"] = (
                        one_euro_filter_j.process(
                            emo_driving_all[j]["jacobian"].cpu() * 100
                        )
                        / 100
                    )
                    if not cpu:
                        emo_driving_all[j]["jacobian"] = emo_driving_all[j][
                            "jacobian"
                        ].cuda()

            one_euro_filter_v = self.OneEuroFilter(mincutoff=0.05, beta=8, dcutoff=1.0, freq=100)
            one_euro_filter_j = self.OneEuroFilter(mincutoff=0.05, beta=8, dcutoff=1.0, freq=100)

            for j in range(len(kp_driving_all)):
                kp_driving_all[j]["value"] = (
                    one_euro_filter_v.process(kp_driving_all[j]["value"].cpu() * 10) / 10
                )
                if not cpu:
                    kp_driving_all[j]["value"] = kp_driving_all[j]["value"].cuda()
                kp_driving_all[j]["jacobian"] = (
                    one_euro_filter_j.process(kp_driving_all[j]["jacobian"].cpu() * 10) / 10
                )
                if not cpu:
                    kp_driving_all[j]["jacobian"] = kp_driving_all[j]["jacobian"].cuda()

            for frame_idx in tqdm(range(len(deco_out[0]))):
                if opt['check_add']:
                    kp_driving = kp_detector_a(deco_out[:, 0])
                else:
                    kp_driving = kp_driving_all[frame_idx]

                if opt['add_emo']:
                    emo_driving = emo_driving_all[frame_idx]
                    if opt['type'] == "linear_3":
                        kp_driving["value"][:, 1] = (
                            kp_driving["value"][:, 1] + emo_driving["value"][:, 0] * 0.2
                        )
                        kp_driving["jacobian"][:, 1] = (
                            kp_driving["jacobian"][:, 1]
                            + emo_driving["jacobian"][:, 0] * 0.2
                        )
                        kp_driving["value"][:, 4] = (
                            kp_driving["value"][:, 4] + emo_driving["value"][:, 1]
                        )
                        kp_driving["jacobian"][:, 4] = (
                            kp_driving["jacobian"][:, 4] + emo_driving["jacobian"][:, 1]
                        )
                        kp_driving["value"][:, 6] = (
                            kp_driving["value"][:, 6] + emo_driving["value"][:, 2]
                        )
                        kp_driving["jacobian"][:, 6] = (
                            kp_driving["jacobian"][:, 6] + emo_driving["jacobian"][:, 2]
                        )

                kp_norm = self.normalize_kp(
                    kp_source=kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                    use_relative_movement=relative,
                    use_relative_jacobian=relative,
                    adapt_movement_scale=adapt_movement_scale,
                )
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

                predictions.append(
                    np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
                )
        return predictions, features

    def test_auido(self, example_image, audio_feature, all_pose, opt):
        """
        generate audio feature (key points of motion).
        """
        with open(opt['config']) as f:
            para = self.yaml.load(f, Loader=self.yaml.FullLoader)

        if not opt['cpu']:
            audio_feature = audio_feature.cuda()
        else:
            audio_feature = audio_feature.cpu()

        audio_feature.eval()
        test_file = opt['in_file']
        pose = all_pose[:, :6]
        if len(pose) == 1:
            pose = np.repeat(pose, 100, 0)
        elif opt['smooth_pose']:
            one_euro_filter = self.OneEuroFilter(
                mincutoff=0.004, beta=0.7, dcutoff=1.0, freq=100
            )
            for j in range(len(pose)):
                pose[j] = one_euro_filter.process(pose[j])

        example_image = np.array(example_image, dtype="float32").transpose((2, 0, 1))

        speech, sr = self.librosa.load(test_file, sr=16000)
        speech = np.insert(speech, 0, np.zeros(1920))
        speech = np.append(speech, np.zeros(1920))
        mfcc = self.python_speech_features.mfcc(speech, 16000, winstep=0.01)

        print("=======================================")
        print("Start to generate images")

        ind = 3
        with torch.no_grad():
            fake_lmark = []
            input_mfcc = []
            while ind <= int(mfcc.shape[0] / 4) - 4:
                t_mfcc = mfcc[(ind - 3) * 4 : (ind + 4) * 4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc).cuda()
                input_mfcc.append(t_mfcc)
                ind += 1
            input_mfcc = torch.stack(input_mfcc, dim=0).cpu()

            if len(pose) < len(input_mfcc):
                gap = len(input_mfcc) - len(pose)
                n = int((gap / len(pose) / 2)) + 2
                pose = np.concatenate((pose, pose[::-1, :]), axis=0)
                pose = np.tile(pose, (n, 1))
            if len(pose) > len(input_mfcc):
                pose = pose[: len(input_mfcc), :]

            if not opt['cpu']:
                example_image = self.Variable(
                    torch.FloatTensor(example_image.astype(float))
                ).cuda()
                example_image = torch.unsqueeze(example_image, 0)
                pose = self.Variable(torch.FloatTensor(pose.astype(float))).cuda()
            else:
                example_image = self.Variable(
                    torch.FloatTensor(example_image.astype(float))
                ).cpu()
                example_image = torch.unsqueeze(example_image, 0).cpu()
                pose = self.Variable(torch.FloatTensor(pose.astype(float))).cpu()

            pose = pose.unsqueeze(0)

            input_mfcc = input_mfcc.unsqueeze(0)

            deco_out = audio_feature(
                example_image, input_mfcc, pose, para["train_params"]["jaco_net"], 1.6
            )

            return deco_out

    def save(self, path, frames, format):
        """
        save png.
        """
        if format == ".png":
            if not os.path.exists(path):
                os.makedirs(path)
            for j, frame in enumerate(frames):
                self.imageio.imsave(path + "/" + str(j) + ".png", frame)
        else:
            print("Unknown format %s" % format)
            exit()
    
    class VideoWriter(object):
        """
        VideoWriter.
        """
        def __init__(self, path, width, height, fps):
            fourcc = EAMMTrainer.cv2.VideoWriter_fourcc(*"XVID")
            self.path = path
            self.out = EAMMTrainer.cv2.VideoWriter(self.path, fourcc, fps, (width, height))

        def write_frame(self, frame):
            self.out.write(frame)

        def end(self):
            self.out.release()

    def concatenate(self, number, imgs, save_path):
        """
        concatenate generated frames to a video.
        """
        width, height = imgs.shape[-3:-1]
        imgs = imgs.reshape(number, -1, width, height, 3)
        if number == 2:
            left = imgs[0]
            right = imgs[1]

            im_all = []
            for i in range(len(left)):
                im = np.concatenate((left[i], right[i]), axis=1)
                im_all.append(im)
        if number == 3:
            left = imgs[0]
            middle = imgs[1][:,:,:,::-1]
            right = imgs[2][:,:,:,::-1]

            im_all = []
            for i in range(len(left)):
                im = np.concatenate((left[i], middle[i], right[i]), axis=1)
                im_all.append(im)
        if number == 4:
            left = imgs[0]
            left2 = imgs[1]
            right = imgs[2]
            right2 = imgs[3]

            im_all = []
            for i in range(len(left)):
                im = np.concatenate((left[i], left2[i], right[i], right2[i]), axis=1)
                im_all.append(im)
        if number == 5:
            left = imgs[0]
            left2 = imgs[1]
            middle = imgs[2]
            right = imgs[3]
            right2 = imgs[4]

            im_all = []
            for i in range(len(left)):
                im = np.concatenate(
                    (left[i], left2[i], middle[i], right[i], right2[i]), axis=1
                )
                im_all.append(im)

        self.imageio.mimsave(save_path, [self.skimage.img_as_ubyte(frame) for frame in im_all], fps=25)

    def add_audio(self, video_name=None, audio_dir=None):
        """
        add audio to the generated video.
        """
        command = (
            "ffmpeg -i "
            + video_name
            + " -i "
            + audio_dir
            + " -vcodec copy  -acodec copy -y  "
            + video_name.replace(".mp4", ".mov")
        )
        print(command)
        subprocess.call(command)
        # os.system(command)

    def crop_image(self, source_image):
        """
        All videos are aligned via centering (crop & resize) the location of the first frame’s face and resized to 256 × 256
        """
        template = np.load("checkpoints/EAMM/M003_template.npy")
        image = self.cv2.imread(source_image)
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)  # detect human face
        if len(rects) != 1:
            return 0
        for j, rect in enumerate(rects):
            shape = self.predictor(gray, rect)  # detect 68 points
            shape = self.shape_to_np(shape)

        pts2 = np.float32(template[:47, :])
        pts1 = np.float32(shape[:47, :])  # eye and nose
        tform = self.tf.SimilarityTransform()
        tform.estimate(
            pts2, pts1
        )  # Set the transformation matrix with the explicit parameters.

        dst = self.tf.warp(image, tform, output_shape=(256, 256))

        dst = np.array(dst * 255, dtype=np.uint8)
        return dst

    def smooth_pose(self, pose_file, pose_long):
        """
        smooth pose of the driven video.
        """
        start = np.load(pose_file)
        video_pose = np.load(pose_long)
        delta = video_pose - video_pose[0, :]
        print(len(delta))

        pose = np.repeat(start, len(delta), axis=0)
        all_pose = pose + delta

        return all_pose
    
    def _load_config(self, file):
        """
        load evaluate opt.
        """
        file = Path(file)
        assert file.exists()
        with open(file, "r", encoding="utf-8") as f:
            self.opt.update(
                self.yaml.load(f.read(), Loader=self.yaml.FullLoader)
            )
        self.logger.info(
            "\n".join(
                [
                    (
                        set_color("{}", "cyan") + " =" + set_color(" {}", "yellow")
                    ).format(arg, value)
                    for arg, value in self.opt.items()
                ]
            )
        )
    
    @torch.no_grad()
    def evaluate(self, load_best_model=True, model_file=None):
        if model_file is None:
            return
        
        self.opt = dict()
        self._load_config(model_file)
        
        all_pose = np.load(self.opt['pose_file']).reshape(-1, 7)
        if self.opt['pose_long']:
            all_pose = self.smooth_pose(self.opt['pose_file'], self.opt['pose_given'])

        source_image = self.skimage.img_as_float32(self.crop_image(self.opt['source_image']))
        source_image = self.st.resize(source_image, (256, 256))[..., :3]      

        reader = self.imageio.get_reader(self.opt['driving_video'])
        fps = reader.get_meta_data()["fps"]
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        driving_video = [self.st.resize(frame, (256, 256))[..., :3] for frame in driving_video]
        driving_video = self.get_aligned_image(driving_video, self.opt)
        transformed_video = self.get_transformed_image(driving_video, self.opt)
        transformed_video = np.array(transformed_video)

        (
            generator,
            kp_detector,
            kp_detector_a,
            audio_feature,
            emo_detector,
        ) = self.load_checkpoints(
            opt=self.opt,
            checkpoint_path=self.opt['checkpoint'],
            audio_checkpoint_path=self.opt['audio_checkpoint'],
            emo_checkpoint_path=self.opt['emo_checkpoint'],
            cpu=self.opt['cpu'],
        )

        deco_out = self.test_auido(source_image, audio_feature, all_pose, self.opt)
        if len(driving_video) < len(deco_out[0]):
            driving_video = np.resize(driving_video, (len(deco_out[0]), 256, 256, 3))
            transformed_video = np.resize(
                transformed_video, (len(deco_out[0]), 256, 256, 3)
            )
        else:
            driving_video = driving_video[: len(deco_out[0])]

        self.opt['add_emo'] = False
        predictions, _ = self.make_animation_smooth(
            source_image,
            driving_video,
            transformed_video,
            deco_out,
            self.opt['kp_loss'],
            generator,
            kp_detector,
            kp_detector_a,
            emo_detector,
            self.opt,
            relative=self.opt['relative'],
            adapt_movement_scale=self.opt['adapt_scale'],
            cpu=self.opt['cpu'],
        )
        
        result_path = Path(self.opt['result_path'])
        if not result_path.exists():
            result_path.mkdir(parents=True,exist_ok=True)
        
        self.imageio.mimsave(
            os.path.join(self.opt['result_path'], "neutral.mp4"),
            [self.skimage.img_as_ubyte(frame[:,:,::-1]) for frame in predictions],
            fps=fps,
        )
        predictions = np.array(predictions)

        self.opt['add_emo'] = True
        predictions1, _ = self.make_animation_smooth(
            source_image,
            driving_video,
            transformed_video,
            deco_out,
            self.opt['kp_loss'],
            generator,
            kp_detector,
            kp_detector_a,
            emo_detector,
            self.opt,
            relative=self.opt['relative'],
            adapt_movement_scale=self.opt['adapt_scale'],
            cpu=self.opt['cpu'],
        )

        self.imageio.mimsave(
            os.path.join(self.opt['result_path'], "emotion.mp4"),
            [self.skimage.img_as_ubyte(frame[:, :, ::-1]) for frame in predictions1],
            fps=fps,
        )
        self.add_audio(os.path.join(self.opt['result_path'], "emotion.mp4"), self.opt['in_file'])
        predictions1 = np.array(predictions1)
        all_imgs = np.concatenate((driving_video, predictions, predictions1), axis=0)
        save_path = os.path.join(self.opt['result_path'], "all.mp4")
        self.concatenate(3, all_imgs, save_path)
        self.add_audio(save_path, self.opt['in_file'])