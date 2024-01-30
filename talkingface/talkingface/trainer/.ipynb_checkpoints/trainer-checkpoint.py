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

from talkingface.utils.util.utils import Record
from talkingface.utils.util.icp import icp
from talkingface.utils.util.icp import *
from talkingface.utils.util.geo_math import area_of_polygon
from talkingface.utils.util.geo_math import area_of_signed_polygon

import time #需要额外加到requirement.txt



from talkingface.utils import(
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
)

from talkingface.data.dataset.audio2landmark_content_dataset import Audio2landmark_contentDataset
from talkingface.model.audio_driven_talkingface.audio2landmark_content import Audio2landmark_content

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
#         saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())
        saved_model_file = "ckpt_last_epoch.pth"
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
        device = self.config['device']
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
        device = self.config['device']
        resume_file = str(resume_file)
#         print("resume_file",resume_file)
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
#             print("========================================================")
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
    def evaluate(self, load_best_model=False, model_file=None):
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

class Audio2landmark_contentTrainer(Trainer):
    def __init__(self, config, model):
        super(Audio2landmark_contentTrainer, self).__init__(config, model)
        
#         root_dir = ROOT_DIR
#         # opt_parser.root_dir = ROOT_DIR
#         opt_parser.dump_dir = os.path.join(opt_parser.root_dir, 'dump')
#         opt_parser.ckpt_dir = os.path.join(opt_parser.root_dir, 'ckpt', opt_parser.name)
#         try_mkdir(opt_parser.ckpt_dir)
#         opt_parser.log_dir = os.path.join(opt_parser.root_dir, 'log')

#         # make directory for nn outputs
#         try_mkdir(opt_parser.dump_dir.replace('dump','nn_result'))
#         try_mkdir(os.path.join(opt_parser.dump_dir.replace('dump', 'nn_result'), opt_parser.name))
        
#     def fit(self, train_data, valid_data=None):
#         """
#         训练Audio2Landmark模型

#         Args:
#             train_data (DataLoader): 训练数据加载器
#             valid_data (DataLoader, optional): 验证数据加载器，默认为None

#         Returns:
#             None
#         """
#         for epoch_idx in range(self.start_epoch, self.epochs):
#             training_start_time = time()
#             train_loss = self._train_epoch(train_data, epoch_idx)
#             self.train_loss_dict[epoch_idx] = (
#                 sum(train_loss) if isinstance(train_loss, tuple) else train_loss
#             )
#             training_end_time = time()
#             train_loss_output = self._generate_train_loss_output(
#                 epoch_idx, training_start_time, training_end_time, train_loss
#             )

#             if self.verbose:
#                 self.logger.info(train_loss_output)

#             if self.eval_step <= 0 or not valid_data:
#                 if self.saved:
#                     self._save_checkpoint(epoch_idx)
#                 continue

#             if (epoch_idx + 1) % self.eval_step == 0:
#                 valid_start_time = time()
#                 valid_loss = self._valid_epoch(valid_data)
#                 valid_end_time = time()

#                 valid_loss_output = (
#                     set_color("valid result", "blue") + ": \n" + dict2str(valid_loss)
#                 )
#                 if self.verbose:
#                     self.logger.info(valid_loss_output)

#                 if self.early_stopping:
#                     (self.best_valid_score, self.cur_step, stop_flag, update_flag) = early_stopping(
#                         valid_loss["loss"],
#                         self.best_valid_score,
#                         self.cur_step,
#                         max_step=self.stopping_step,
#                         bigger=self.valid_metric_bigger,
#                     )

#                 if update_flag:
#                     if self.saved:
#                         self._save_checkpoint(epoch_idx)

#     def evaluate(self, eval_data):
#         """
#         评估Audio2Landmark模型

#         Args:
#             eval_data (DataLoader): 评估数据加载器

#         Returns:
#             None
#         """
#         self.model.eval()
#         total_loss_dict = {}
#         iter_data = tqdm(eval_data, total=len(eval_data), ncols=None, desc="Evaluating")

#         with torch.no_grad():
#             for batch_idx, batched_data in enumerate(iter_data):
#                 batched_data.to(self.device)
#                 losses_dict = self.model.calculate_loss(batched_data, valid=True)

#                 for key, value in losses_dict.items():
#                     if key in total_loss_dict:
#                         if not torch.is_tensor(value):
#                             total_loss_dict[key] += value
#                         else:
#                             losses_dict[key] = value.item()
#                             total_loss_dict[key] += value.item()
#                     else:
#                         if not torch.is_tensor(value):
#                             total_loss_dict[key] = value
#                         else:
#                             losses_dict[key] = value.item()
#                             total_loss_dict[key] = value.item()

#                 iter_data.set_description(
#                     set_color(f"Evaluating {losses_dict}", "pink")
#                 )

#         average_loss_dict = {}
#         for key, value in total_loss_dict.items():
#             average_loss_dict[key] = value / len(eval_data)

#         return average_loss_dict

#     def _train_epoch(self, train_data, epoch_idx):
#         self.model.train()
#         total_loss_dict = {}
#         step = 0
#         iter_data = tqdm(train_data, total=len(train_data), ncols=None)

#         for batch_idx, interaction in enumerate(iter_data):
#             self.optimizer.zero_grad()
#             step += 1
#             losses_dict = self.model.calculate_loss(interaction)
#             loss = losses_dict["loss"]

#             for key, value in losses_dict.items():
#                 if key in total_loss_dict:
#                     if not torch.is_tensor(value):
#                         total_loss_dict[key] += value
#                     else:
#                         losses_dict[key] = value.item()
#                         total_loss_dict[key] += value.item()
#                 else:
#                     if not torch.is_tensor(value):
#                         total_loss_dict[key] = value
#                     else:
#                         losses_dict[key] = value.item()
#                         total_loss_dict[key] = value.item()

#             iter_data.set_description(
#                 set_color(f"train {epoch_idx} {losses_dict}", "pink")
#             )

#             self._check_nan(loss)
#             loss.backward()
#             self.optimizer.step()

#         average_loss_dict = {}
#         for key, value in total_loss_dict.items():
#             average_loss_dict[key] = value / step

#         return average_loss_dict

#     def _valid_epoch(self, valid_data):
#         self.model.eval()
#         total_loss_dict = {}
#         iter_data = tqdm(
#             valid_data,
#             total=len(valid_data),
#             ncols=None,
#             desc=set_color("Validating", "pink"),
#         )

#         with torch.no_grad():
#             step = 0
#             for batch_idx, batched_data in enumerate(iter_data):
#                 step += 1
#                 batched_data.to(self.device)
#                 losses_dict = self.model.calculate_loss(batched_data, valid=True)

#                 for key, value in losses_dict.items():
#                     if key in total_loss_dict:
#                         if not torch.is_tensor(value):
#                             total_loss_dict[key] += value
#                         else:
#                             losses_dict[key] = value.item()
#                             total_loss_dict[key] += value.item()
#                     else:
#                         if not torch.is_tensor(value):
#                             total_loss_dict[key] = value
#                         else:
#                             losses_dict[key] = value.item()
#                             total_loss_dict[key] = value.item()

#                 iter_data.set_description(
#                     set_color(f"Valid {losses_dict}", "pink")
#                 )

#         average_loss_dict = {}
#         for key, value in total_loss_dict.items():
#             average_loss_dict[key] = value / step

#         return average_loss_dict
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
        self.std_face_id



#         loss_func = loss_func or self.model.calculate_loss
#         total_loss_dict = {}
#         step = 0
#         iter_data = (
#             tqdm(
#             train_data,
#             total=len(train_data),
#             ncols=None,
#             )
#             if show_progress
#             else train_data
#         )

#         for batch_idx, interaction in enumerate(iter_data):
#             self.optimizer.zero_grad()
#             step += 1
#             losses_dict = loss_func(interaction)
#             loss = losses_dict["loss"]

#             for key, value in losses_dict.items():
#                 if key in total_loss_dict:
#                     if not torch.is_tensor(value):
#                         total_loss_dict[key] += value
#                     # 如果键已经在总和字典中，累加当前值
#                     else:
#                         losses_dict[key] = value.item()
#                         total_loss_dict[key] += value.item()
#                 else:
#                     if not torch.is_tensor(value):
#                         total_loss_dict[key] = value
#                     # 否则，将当前值添加到字典中
#                     else:
#                         losses_dict[key] = value.item()
#                         total_loss_dict[key] = value.item()
#             iter_data.set_description(set_color(f"train {epoch_idx} {losses_dict}", "pink"))

#             self._check_nan(loss)
#             loss.backward()
#             self.optimizer.step()
#         average_loss_dict = {}
#         for key, value in total_loss_dict.items():
#             average_loss_dict[key] = value/step

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
        print("fit")
#         self.train()
        
        print('Run on device:', self.config['device'])
        device = self.config['device']

        # Step 1 : load opt_parser
        opt_parser = self.config
        std_face_id = np.loadtxt('talkingface/utils/dataset_utils/utils/STD_FACE_LANDMARKS.txt')
        jpg_shape=None# 后续写到yaml文件里面去
        if(jpg_shape is not None):
            std_face_id = jpg_shape
        std_face_id = std_face_id.reshape(1, 204)
        std_face_id = torch.tensor(std_face_id, requires_grad=False, dtype=torch.float).to(device)
        self.std_face_id=std_face_id

        self.train_data = Audio2landmark_contentDataset(dump_dir=self.config['dump_dir'],
                                                dump_name='autovc_retrain_mel',
                                                status='train',
                                               num_window_frames=self.config['num_window_frames'],
                                               num_window_step=self.config['num_window_step'])
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.config['batch_size'],
                                                           shuffle=False, num_workers=0,
                                                           collate_fn=self.train_data.my_collate_in_segments_noemb)
        print('TRAIN num videos: {}'.format(len(self.train_data)))

        self.eval_data = Audio2landmark_contentDataset(dump_dir=self.config['dump_dir'],
                                                 dump_name='autovc_retrain_mel',
                                                 status='test',
                                                 num_window_frames=self.config['num_window_frames'],
                                                 num_window_step=self.config['num_window_step'])
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_data, batch_size=self.config['batch_size'],
                                                            shuffle=False, num_workers=0,
                                                            collate_fn=self.eval_data.my_collate_in_segments_noemb)
        print('EVAL num videos: {}'.format(len(self.eval_data)))

        # Step 3: Load model
        self.C = Audio2landmark_content(num_window_frames=self.config['num_window_frames'], hidden_size=self.config['hidden_size'],
                                      in_size=self.config['in_size'], use_prior_net=self.config['use_prior_net'],
                                      bidirectional=False, drop_out=self.config['drop_out'])

        if(self.config['load_a2l_C_name'].split('/')[-1] != ''):
            ckpt = torch.load(self.config['load_a2l_C_name'])
            self.C.load_state_dict(ckpt['model_g_face_id'])
            print('======== LOAD PRETRAINED CONTENT BRANCH MODEL {} ========='.format(self.config['load_a2l_C_name']))
        self.C.to(device)

        self.t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)
        self.anchor_t_shape = np.loadtxt('talkingface/utils/dataset_utils/utils/STD_FACE_LANDMARKS.txt')
        self.anchor_t_shape = self.anchor_t_shape[self.t_shape_idx, :]

        self.opt_C = optim.Adam(self.C.parameters(), lr=self.config['lr'], weight_decay=self.config['reg_lr'])

        self.loss_mse = torch.nn.MSELoss()
        
        self.train()

    def __train_content__(self, fls, aus, face_id, is_training=True):
        device = self.config['device']

        fls_gt = fls[:, 0, :].detach().clone().requires_grad_(False)

        if (face_id.shape[0] == 1):
            face_id = face_id.repeat(aus.shape[0], 1)
        face_id = face_id.requires_grad_(False)

        fl_dis_pred, _ = self.C(aus, face_id)

        ''' lip region weight '''
        w = torch.abs(fls[:, 0, 66 * 3 + 1] - fls[:, 0, 62 * 3 + 1])
        w = torch.tensor([1.0]).to(device) / (w * 4.0 + 0.1)
        w = w.unsqueeze(1)
        lip_region_w = torch.ones((fls.shape[0], 204)).to(device)
        lip_region_w[:, 48*3:] = torch.cat([w] * 60, dim=1)
        lip_region_w = lip_region_w.detach().clone().requires_grad_(False)

        if (self.config['use_lip_weight']):
            # loss = torch.mean(torch.mean((fl_dis_pred + face_id - fls[:, 0, :]) ** 2, dim=1) * w)
            loss = torch.mean(torch.abs(fl_dis_pred +face_id[0:1].detach() - fls_gt) * lip_region_w)
        else:
            # loss = self.loss_mse(fl_dis_pred + face_id, fls[:, 0, :])
            loss = torch.nn.functional.l1_loss(fl_dis_pred+face_id[0:1].detach(), fls_gt)

        if (self.config['use_motion_loss']):
            pred_motion = fl_dis_pred[:-1] - fl_dis_pred[1:]
            gt_motion = fls_gt[:-1] - fls_gt[1:]
            loss += torch.nn.functional.l1_loss(pred_motion, gt_motion)

        ''' use laplacian smooth loss '''
        if (self.config['lambda_laplacian_smooth_loss'] > 0.0):
            n1 = [1] + list(range(0, 16)) + [18] + list(range(17, 21)) + [23] + list(range(22, 26)) + \
                 [28] + list(range(27, 35)) + [41] + list(range(36, 41)) + [47] + list(range(42, 47)) + \
                 [59] + list(range(48, 59)) + [67] + list(range(60, 67))
            n2 = list(range(1, 17)) + [15] + list(range(18, 22)) + [20] + list(range(23, 27)) + [25] + \
                 list(range(28, 36)) + [34] + list(range(37, 42)) + [36] + list(range(43, 48)) + [42] + \
                 list(range(49, 60)) + [48] + list(range(61, 68)) + [60]
            V = (fl_dis_pred + face_id[0:1].detach()).view(-1, 68, 3)
            L_V = V - 0.5 * (V[:, n1, :] + V[:, n2, :])
            G = fls_gt.view(-1, 68, 3)
            L_G = G - 0.5 * (G[:, n1, :] + G[:, n2, :])
            loss_laplacian = torch.nn.functional.l1_loss(L_V, L_G)
            loss += loss_laplacian

        if(is_training):
            self.opt_C.zero_grad()
            loss.backward()
            self.opt_C.step()

        if(not is_training):
            # ''' CALIBRATION '''
            np_fl_dis_pred = fl_dis_pred.detach().cpu().numpy()
            K = int(np_fl_dis_pred.shape[0] * 0.5)
            for calib_i in range(204):
                min_k_idx = np.argpartition(np_fl_dis_pred[:, calib_i], K)
                m = np.mean(np_fl_dis_pred[min_k_idx[:K], calib_i])
                np_fl_dis_pred[:, calib_i] = np_fl_dis_pred[:, calib_i] - m
            fl_dis_pred = torch.tensor(np_fl_dis_pred, requires_grad=False).to(device)

        return fl_dis_pred, face_id[0:1, :], loss

    def __train_pass__(self, epoch, log_loss, is_training=True):
        device = self.config['device']
        st_epoch = time.time()

        # Step 1: init setup
        if(is_training):
            self.C.train()
            data = self.train_data
            dataloader = self.train_dataloader
            status = 'TRAIN'
        else:
            self.C.eval()
            data = self.eval_data
            dataloader = self.eval_dataloader
            status = 'EVAL'

        random_clip_index = np.random.permutation(len(dataloader))[0:self.config['random_clip_num']]
        print('random visualize clip index', random_clip_index)

        # Step 2: train for each batch
        for i, batch in enumerate(dataloader):

            global_id, video_name = data[i][0][1][0], data[i][0][1][1][:-4]
            inputs_fl, inputs_au = batch
            inputs_fl_ori, inputs_au_ori = inputs_fl.to(device), inputs_au.to(device)

            std_fls_list, fls_pred_face_id_list, fls_pred_pos_list = [], [], []
            seg_bs = 512

            ''' pick a most closed lip frame from entire clip data '''
            close_fl_list = inputs_fl_ori[::10, 0, :]
            idx = self.__close_face_lip__(close_fl_list.detach().cpu().numpy())
            input_face_id = close_fl_list[idx:idx + 1, :]

            ''' register face '''
            if (self.config['use_reg_as_std']):
                landmarks = input_face_id.detach().cpu().numpy().reshape(68, 3)
                frame_t_shape = landmarks[self.t_shape_idx, :]
                T, distance, itr = icp(frame_t_shape, self.anchor_t_shape)
                landmarks = np.hstack((landmarks, np.ones((68, 1))))
                registered_landmarks = np.dot(T, landmarks.T).T
                input_face_id = torch.tensor(registered_landmarks[:, 0:3].reshape(1, 204), requires_grad=False,
                                             dtype=torch.float).to(device)

            for in_batch in range(self.config['in_batch_nepoch']):

                std_fls_list, fls_pred_face_id_list, fls_pred_pos_list = [], [], []

                if (is_training):
                    rand_start = np.random.randint(0, inputs_fl_ori.shape[0] // 5, 1).reshape(-1)
                    inputs_fl = inputs_fl_ori[rand_start[0]:]
                    inputs_au = inputs_au_ori[rand_start[0]:]
                else:
                    inputs_fl = inputs_fl_ori
                    inputs_au = inputs_au_ori

                for j in range(0, inputs_fl.shape[0], seg_bs):

                    # Step 3.1: load segments
                    inputs_fl_segments = inputs_fl[j: j + seg_bs]
                    inputs_au_segments = inputs_au[j: j + seg_bs]
                    fl_std = inputs_fl_segments[:, 0, :].data.cpu().numpy()

                    if(inputs_fl_segments.shape[0] < 10):
                        continue

                    fl_dis_pred_pos, input_face_id, loss = \
                        self.__train_content__(inputs_fl_segments, inputs_au_segments, input_face_id, is_training)

                    fl_dis_pred_pos = (fl_dis_pred_pos + input_face_id).data.cpu().numpy()
                    ''' solve inverse lip '''
                    fl_dis_pred_pos = self.__solve_inverse_lip2__(fl_dis_pred_pos)

                    fls_pred_pos_list += [fl_dis_pred_pos.reshape((-1, 204))]
                    std_fls_list += [fl_std.reshape((-1, 204))]

                    for key in log_loss.keys():
                        if (key not in locals().keys()):
                            continue
                        if (type(locals()[key]) == float):
                            log_loss[key].add(locals()[key])
                        else:
                            log_loss[key].add(locals()[key].data.cpu().numpy())


                if (epoch % self.config['jpg_freq'] == 0 and (i in random_clip_index or in_batch % self.config['jpg_freq'] == 1)):
                    def save_fls_av(fake_fls_list, postfix='', ifsmooth=True):
                        fake_fls_np = np.concatenate(fake_fls_list)
                        filename = 'fake_fls_{}_{}_{}.txt'.format(epoch, video_name, postfix)
                        np.savetxt(
                            os.path.join(self.config['dump_dir'], '../nn_result', self.config['name'], filename),
                            fake_fls_np, fmt='%.6f')
                        audio_filename = '{:05d}_{}_audio.wav'.format(global_id, video_name)
                        from util.vis import Vis_old
                        Vis_old(run_name=self.config['name'], pred_fl_filename=filename, audio_filename=audio_filename,
                                fps=62.5, av_name='e{:04d}_{}_{}'.format(epoch, in_batch, postfix),
                                postfix=postfix, root_dir=self.config['root_dir'], ifsmooth=ifsmooth)

                    if (self.config['show_animation'] and not is_training):
                        print('show animation ....')
                        save_fls_av(fls_pred_pos_list, 'pred_{}'.format(i), ifsmooth=True)
                        save_fls_av(std_fls_list, 'std_{}'.format(i), ifsmooth=False)
                        from util.vis import Vis_comp
                        Vis_comp(run_name=self.config['name'],
                                 pred1='fake_fls_{}_{}_{}.txt'.format(epoch, video_name, 'pred_{}'.format(i)),
                                 pred2='fake_fls_{}_{}_{}.txt'.format(epoch, video_name, 'std_{}'.format(i)),
                                 audio_filename='{:05d}_{}_audio.wav'.format(global_id, video_name),
                                fps=62.5, av_name='e{:04d}_{}_{}'.format(epoch, in_batch, 'comp_{}'.format(i)),
                                postfix='comp_{}'.format(i), root_dir=self.config['root_dir'], ifsmooth=False)

                    self.__save_model__(save_type='last_inbatch', epoch=epoch)

                if (self.config['verbose'] <= 1):
                    print('{} Epoch: #{} batch #{}/{} inbatch #{}/{}'.format(
                        status, epoch, i, len(dataloader),
                    in_batch, self.config['in_batch_nepoch']), end=': ')
                    for key in log_loss.keys():
                        print(key, '{:.5f}'.format(log_loss[key].per('batch')), end=', ')
                    print('')

        if (self.config['verbose'] <= 2):
            print('==========================================================')
            print('{} Epoch: #{}'.format(status, epoch), end=':')
            for key in log_loss.keys():
                print(key, '{:.4f}'.format(log_loss[key].per('epoch')), end=', ')
            print(
                'Epoch time usage: {:.2f} sec\n==========================================================\n'.format(
                    time.time() - st_epoch))
        self.__save_model__(save_type='last_epoch', epoch=epoch)
        if (epoch % self.config['ckpt_epoch_freq'] == 0):
            self.__save_model__(save_type='e_{}'.format(epoch), epoch=epoch)


    def __close_face_lip__(self, fl):
        device = self.config['device']
        facelandmark = fl.reshape(-1, 68, 3)
#         from util.geo_math import area_of_polygon
        min_area_lip, idx = 999, 0
        for i, fls in enumerate(facelandmark):
            area_of_mouth = area_of_polygon(fls[list(range(60, 68)), 0:2])
            if (area_of_mouth < min_area_lip):
                min_area_lip = area_of_mouth
                idx = i
        return idx

    def test(self):
        device = self.config['device']
        eval_loss = {key: Record(['epoch', 'batch']) for key in ['loss']}
        with torch.no_grad():
            self.__train_pass__(0, eval_loss, is_training=False)

    def train(self):
        device = self.config['device']
        train_loss = {key: Record(['epoch', 'batch']) for key in ['loss']}
        eval_loss = {key: Record(['epoch', 'batch']) for key in ['loss']}

        for epoch in range(self.config['nepoch']):
            self.__train_pass__(epoch=epoch, log_loss=train_loss)

            with torch.no_grad():
                self.__train_pass__(epoch, eval_loss, is_training=False)


    def __solve_inverse_lip2__(self, fl_dis_pred_pos_numpy):
        device = self.config['device']
        for j in range(fl_dis_pred_pos_numpy.shape[0]):
            init_face = self.std_face_id.detach().cpu().numpy()
#             from util.geo_math import area_of_signed_polygon
            fls = fl_dis_pred_pos_numpy[j].reshape(68, 3)
            area_of_mouth = area_of_signed_polygon(fls[list(range(60, 68)), 0:2])
            if (area_of_mouth < 0):
                fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3] = 0.5 *(fl_dis_pred_pos_numpy[j, 63 * 3:64 * 3] + fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3])
                fl_dis_pred_pos_numpy[j, 63 * 3:64 * 3] = fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3]
                fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3] = 0.5 *(fl_dis_pred_pos_numpy[j, 62 * 3:63 * 3] + fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3])
                fl_dis_pred_pos_numpy[j, 62 * 3:63 * 3] = fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3]
                fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3] = 0.5 *(fl_dis_pred_pos_numpy[j, 61 * 3:62 * 3] + fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3])
                fl_dis_pred_pos_numpy[j, 61 * 3:62 * 3] = fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3]
                p = max([j-1, 0])
                fl_dis_pred_pos_numpy[j, 55 * 3+1:59 * 3+1:3] = fl_dis_pred_pos_numpy[j, 64 * 3+1:68 * 3+1:3] \
                                                          + fl_dis_pred_pos_numpy[p, 55 * 3+1:59 * 3+1:3] \
                                                          - fl_dis_pred_pos_numpy[p, 64 * 3+1:68 * 3+1:3]
                fl_dis_pred_pos_numpy[j, 59 * 3+1:60 * 3+1:3] = fl_dis_pred_pos_numpy[j, 60 * 3+1:61 * 3+1:3] \
                                                          + fl_dis_pred_pos_numpy[p, 59 * 3+1:60 * 3+1:3] \
                                                          - fl_dis_pred_pos_numpy[p, 60 * 3+1:61 * 3+1:3]
                fl_dis_pred_pos_numpy[j, 49 * 3+1:54 * 3+1:3] = fl_dis_pred_pos_numpy[j, 60 * 3+1:65 * 3+1:3] \
                                                          + fl_dis_pred_pos_numpy[p, 49 * 3+1:54 * 3+1:3] \
                                                          - fl_dis_pred_pos_numpy[p, 60 * 3+1:65 * 3+1:3]
        return fl_dis_pred_pos_numpy


    def __save_model__(self, save_type, epoch):
        device = self.config['device']
        if (self.config['write']):
            torch.save({
                'model_g_face_id': self.C.state_dict(),
                'epoch': epoch
            }, os.path.join(self.config['ckpt_dir'], 'ckpt_{}.pth'.format(save_type)))
    
    @torch.no_grad()
    def evaluate(self, load_best_model=False, model_file=None):
        """
        Evaluate the model based on the test data.

        args: load_best_model: bool, whether to load the best model in the training process.
                model_file: str, the model file you want to evaluate.

        """
#         if load_best_model:
#             checkpoint_file = model_file or self.saved_model_file
#             checkpoint = torch.load(checkpoint_file, map_location=self.device)
#             self.model.load_state_dict(checkpoint["state_dict"])
#             self.model.load_other_parameter(checkpoint.get("other_parameter"))
#             message_output = "Loading model structure and parameters from {}".format(
#                 checkpoint_file
#             )
#             self.logger.info(message_output)
        self.model.eval()

        datadict = self.model.generate_batch()
        print(datadict)
#         testdict={'generated_video':["results/test1/epoch1test.mp4"],'real_video':["results/test1/epoch1001test.mp4"]}
        eval_result = self.evaluator.evaluate(datadict)
        self.logger.info(eval_result)

    