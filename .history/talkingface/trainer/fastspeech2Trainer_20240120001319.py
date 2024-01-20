from talkingface.trainer.trainer import Trainer
import tqdm
import torch
import torch.nn as nn
import os
import numpy as np


class FastSpeech2Trainer(Trainer):
    def __init__(self, config, model):
        super(FastSpeech2Trainer, self).__init__(config, model)
        self.loss = FastSpeech2Loss(
            config["preprocess_config"], config["config"]).to(self.device)
        self.vocoder = get_vocoder(config["config"], self.device)

    def get_model(self, config, device, train=False):
        model = self.model.to(device)
        ckpt_dir = config["checkpoint_dir"]
        if os.path.isdir(ckpt_dir):
            # 获取目录中的所有文件
            ckpt_files = [f for f in os.listdir(
                ckpt_dir) if f.endswith('.pth.tar')]
            # 按文件修改时间排序并获取最新的文件
            if ckpt_files:
                latest_ckpt = max(ckpt_files, key=lambda x: os.path.getmtime(
                    os.path.join(ckpt_dir, x)))
                ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
                # 加载检查点文件
                ckpt = torch.load(ckpt_path)
                # 更新模型的状态字典（state_dict）
                model.load_state_dict(ckpt["model"])
                print(f"Loaded model from {ckpt_path}")
            else:
                print("No checkpoint found in", ckpt_dir)
        else:
            print("Checkpoint directory not found:", ckpt_dir)

        if train:
            scheduled_optim = ScheduledOptim(
                model, config
            )
            model.train()
            return model, scheduled_optim

        model.eval()
        model.requires_grad_ = False
        return model
    
    def get_vocoder(config, device):
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
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
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
                         ncols=None) if show_progress else train_data

        for batch_idx, batch in enumerate(iter_data):
            batch = to_device(batch, self.device)

            # Forward
            output = self.model(*(batch[2:]))

            # Calculate Loss
            losses = self.loss(batch, output)
            total_loss = losses[0] / self.config["optimizer"]["grad_acc_step"]
            total_loss.backward()

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

    def __init__(self, model, config):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=config["optimizer"]["betas"],
            eps=config["optimizer"]["eps"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = config["optimizer"]["warm_up_step"]
        self.anneal_steps = config["optimizer"]["anneal_steps"]
        self.anneal_rate = config["optimizer"]["anneal_rate"]
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
