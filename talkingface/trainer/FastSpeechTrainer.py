from torch.utils.data import DataLoader

from talkingface.data.dataset.fastspeech_dataset import FastSpeechDataset
from talkingface.trainer.trainer import Trainer
import numpy as np
import torch
from torch import nn
import os
from tqdm import tqdm

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



class FastSpeechTrainer(Trainer):
    def __init__(self, config, model):
        super(FastSpeechTrainer, self).__init__(config, model)
        self.optimizer=self._build_optimizer()
        self._set_model_parallel()

    def _set_model_parallel(self):
        self.model=nn.DataParallel(self.model)
        self.model.calculate_loss=self.model.module.calculate_loss
        self.model.generate_batch=self.model.module.generate_batch
        self.model.other_parameter=self.model.module.other_parameter
        self.model.load_other_parameter=self.model.module.load_other_parameter
        self.model.__str__=self.model.module.__str__

    def _build_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     betas=(0.9, 0.98),
                                     eps=1e-9)
        scheduled_optim = ScheduledOptim(optimizer,
                                         self.config['decoder_dim'],
                                         self.config['n_warm_up_step'],
                                         self.config['restore_step'])
        return scheduled_optim

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()

        loss_func = loss_func or self.model.calculate_loss
        total_loss_dict = {}
        step = 0
        pbar = tqdm(total=len(train_data)*(self.config['batch_expand_size']))
        for epoch_idx, batchs in enumerate(train_data):
            for j,interaction in enumerate(batchs):
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
                pbar.set_description(set_color(f"train {epoch_idx*len(batchs)+j} {losses_dict}", "pink"))

                self._check_nan(loss)
                loss.backward()
                self.optimizer.step()
                pbar.update(1)
            average_loss_dict = {}
            for key, value in total_loss_dict.items():
                average_loss_dict[key] = value/step

        return average_loss_dict

    def _valid_epoch(self, valid_data, show_progress=False):
        self.model.train()

        loss_func = self.model.calculate_loss
        total_loss_dict = {}
        step = 0
        pbar = tqdm(total=len(valid_data)*(self.config['batch_expand_size']))
        for epoch_idx, batchs in enumerate(valid_data):
            for j,interaction in enumerate(batchs):
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
                pbar.set_description(set_color(f"Valid {losses_dict}", "pink"))
                pbar.update(1)
            average_loss_dict = {}
            for key, value in total_loss_dict.items():
                average_loss_dict[key] = value/step

        return average_loss_dict





class ScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, d_model, n_warmup_steps, current_steps=0):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = current_steps
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr_frozen(self, learning_rate_frozen):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = learning_rate_frozen
        self._optimizer.step()

    def step(self):
        self._update_learning_rate()
        self._optimizer.step()

    def get_learning_rate(self):
        learning_rate = 0.0
        for param_group in self._optimizer.param_groups:
            learning_rate = param_group['lr']

        return learning_rate

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return  self._optimizer.state_dict()

    def load_state_dict(self,dict):
        self._optimizer.load_state_dict(dict)
