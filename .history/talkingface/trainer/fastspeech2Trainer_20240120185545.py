from talkingface.trainer.trainer import Trainer
from talkingface.utils.fastspeech2_transformerblock.vocoder import AttrDict, Generator
from talkingface.data.dataset.fastspeech2_dataset import FastSpeech2Dataset
from time import time
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import numpy as np
import json
from talkingface.utils import (
    early_stopping,
    dict2str,
    set_color
)


class FastSpeech2Trainer(Trainer):
    def __init__(self, config, model):
        super(FastSpeech2Trainer, self).__init__(config, model)
        self.model, self.optimizer = self.get_model(
            config, self.device, train=True)
        self.loss = FastSpeech2Loss(
            config).to(self.device)
        self.vocoder = self.get_vocoder(config, self.device)

    def get_model(self, config, device, train=False):
        model = self.model.to(device)
        ckpt_dir = config["checkpoint_dir"]
        if os.path.isdir(ckpt_dir):
            # 获取目录中的所有文件
            ckpt_files = [f for f in os.listdir(
                ckpt_dir) if f.endswith('.pth')]
            # 按文件修改时间排序并获取最新的文件
            if ckpt_files:
                latest_ckpt = max(ckpt_files, key=lambda x: os.path.getmtime(
                    os.path.join(ckpt_dir, x)))
                ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
                # 加载检查点文件
                ckpt = torch.load(ckpt_path)
                # 更新模型的状态字典（state_dict）
                self.start_epoch = ckpt["epoch"] + 1
                self.cur_step = ckpt["cur_step"]
                model.load_state_dict(ckpt["state_dict"])
                model.load_other_parameter(ckpt.get("other_parameter"))
                print(f"Loaded model from {ckpt_path}")
                message_output = "Checkpoint loaded. Resume training from epoch {}".format(
                    self.start_epoch
                )
                self.logger.info(message_output)
            else:
                print("No checkpoint found in", ckpt_dir)
        else:
            print("Checkpoint directory not found:", ckpt_dir)

        if train:
            scheduled_optim = ScheduledOptim(
                model, config
            )
            scheduled_optim.load_state_dict(ckpt["optimizer"])
            model.train()
            return model, scheduled_optim

        model.eval()
        model.requires_grad_ = False
        return model

    def get_vocoder(self, config, device):
        name = config["vocoder"]["model"]
        speaker = config["vocoder"]["speaker"]

        if name == "MelGAN":
            if speaker == "LJSpeech":
                vocoder = torch.hub.load(
                    "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
                )
            elif speaker == "universal":
                vocoder = torch.hub.load(
                    "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
                )
            vocoder.mel2wav.eval()
            vocoder.mel2wav.to(device)
        elif name == "HiFi-GAN":
            with open("checkpoints/config.json", "r") as f:
                config = json.load(f)
            config = AttrDict(config)
            vocoder = Generator(config)
            if speaker == "LJSpeech":
                ckpt = torch.load("checkpoints/generator_LJSpeech.pth.tar")
            elif speaker == "universal":
                ckpt = torch.load("checkpoints/generator_universal.pth.tar")
            vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder.to(device)

        return vocoder

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        total_loss_dict = {}
        step = 0
        iter_data = tqdm(train_data, total=len(train_data),
                         ncols=None) if not show_progress else train_data
        total_loss_dict["loss"] = 0

        for batchs in iter_data:
            for batch in batchs:
                batch = to_device(batch, self.device)
                # Forward
                output = self.model(*(batch[2:]))

                # Calculate Loss
                losses = self.loss(batch, output)
                total_loss = losses[0] / \
                    self.config["optimizer"]["grad_acc_step"]
                total_loss.backward()
                total_loss_dict["loss"] += total_loss.item()

                if (step + 1) % self.config["optimizer"]["grad_acc_step"] == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(
                    ), self.config["optimizer"]["grad_clip_thresh"])
                    self.optimizer.step_and_update_lr()
                    self.optimizer.zero_grad()

                # Log losses
                losses = [l.item() for l in losses]
                total_loss_dict.update(
                    {f"loss_{i}": losses[i] for i in range(len(losses))})

                step += 1

            average_loss_dict = {key: value / step for key,
                                 value in total_loss_dict.items()}
            return average_loss_dict

    def _valid_epoch(self, valid_data, loss_func=None, show_progress=False):
        self.model.eval()  # 切换到评估模式
        self.model.requires_grad_ = False
        total_loss_dict = {}
        step = 0
        iter_data = tqdm(valid_data, total=len(valid_data),
                         ncols=None, desc="Validating") if not show_progress else valid_data
        total_loss_dict["loss"] = 0

        with torch.no_grad():  # 在验证时不计算梯度
            for batchs in iter_data:
                for batch in batchs:
                    batch = to_device(batch, self.device)

                    # Forward
                    output = self.model(*(batch[2:]))

                    # Calculate Loss
                    losses = self.loss(batch, output)
                    total_loss = losses[0].item()

                    total_loss_dict["loss"] += total_loss

                    # 累加每个损失项
                    for i, l in enumerate(losses):
                        key = f"loss_{i}"
                        total_loss_dict[key] = total_loss_dict.get(
                            key, 0) + l.item()

                    step += 1

            # 计算平均损失
            average_loss_dict = {key: value / step for key,
                                 value in total_loss_dict.items()}
            return average_loss_dict

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        saved_model_file = kwargs.pop(
            "saved_model_file", self.saved_model_file)
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.model.state_dict(),
            "other_parameter": self.model.other_parameter(),
            "optimizer": self.optimizer._optimizer.state_dict(),
        }
        torch.save(state, saved_model_file, pickle_protocol=4)
        if verbose:
            self.logger.info(
                set_color("Saving current", "blue") + f": {saved_model_file}"
            )

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):

        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        if not (self.config['resume_checkpoint_path'] == None) and self.config['resume']:
            self.resume_checkpoint(self.config['resume_checkpoint_path'])

        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(
                    train_loss, tuple) else train_loss
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
                valid_loss = self._valid_epoch(
                    valid_data=valid_data, show_progress=show_progress)

                (self.best_valid_score, self.cur_step, stop_flag, update_flag,) = early_stopping(
                    valid_loss['loss'],
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()

                valid_loss_output = (
                    set_color("valid result", "blue") +
                    ": \n" + dict2str(valid_loss)
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

        dataset = FastSpeech2Dataset(
            self.config,config['val_filelist'] sort=False, drop_last=False
        )
        batch_size = train_config["optimizer"]["batch_size"]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        # Get loss function
        Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

        # Evaluation
        loss_sums = [0 for _ in range(6)]
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                with torch.no_grad():
                    # Forward
                    output = model(*(batch[2:]))

                    # Cal Loss
                    losses = Loss(batch, output)

                    for i in range(len(losses)):
                        loss_sums[i] += losses[i].item() * len(batch[0])

        loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

        message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
            *([step] + [l for l in loss_means])
        )


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(
            src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(
            mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(
            log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, config, current_step=0):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=config["optimizer"]["betas"],
            eps=config["optimizer"]["eps"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = config["optimizer"]["warm_up_step"]
        self.anneal_steps = config["optimizer"]["anneal_steps"]
        self.anneal_rate = config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(config["transformer"]["encoder_hidden"], -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


def to_device(data, device):
    if len(data) == 12:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)
