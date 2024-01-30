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
            print(checkpoint_file)
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            print(checkpoint)
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



class PC_AVSTrainer(Trainer):
    def __init__(self, config, model):
        super(PC_AVSTrainer, self).__init__(config, model)

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
    def evaluate(self, load_best_model=True, model_file=None):
    # if load_best_model:
    #     checkpoint_file = model_file or self.saved_model_file
    #     print(checkpoint_file)
    #     checkpoint = torch.load(checkpoint_file, map_location=self.device)
    #     print(checkpoint)
    #     self.model.load_state_dict(checkpoint["state_dict"])
    #     self.model.load_other_parameter(checkpoint.get("other_parameter"))
    #     message_output = "Loading model structure and parameters from {}".format(
    #         checkpoint_file
    #     )
    #     self.logger.info(message_output)
        self.model.eval()

        datadict = self.model.generate_batch()
        eval_result = self.evaluator.evaluate(datadict)
        self.logger.info(eval_result)
    