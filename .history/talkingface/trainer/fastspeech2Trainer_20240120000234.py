from talkingface.trainer.trainer import Trainer
import tqdm
import torch
import torch.nn as nn


class FastSpeech2Trainer(Trainer):
    def __init__(self, config, model):
        super(FastSpeech2Trainer, self).__init__(config, model)
        self.loss = FastSpeech2Loss(config["preprocess_config"], config["model_config"]).to(self.device)
        self.vocoder = get_vocoder(config["model_config"], self.device)
    
    def get_model(self,config, device, train=False):

        model = self.model.to(device)
        if config:
            ckpt_path = os.path.join(
                config["path"]["ckpt_path"],
                "{}.pth.tar".format(args.restore_step),
            )
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model"])

        if train:
            scheduled_optim = ScheduledOptim(
                model, config, config, args.restore_step
            )
            if args.restore_step:
                scheduled_optim.load_state_dict(ckpt["optimizer"])
            model.train()
            return model, scheduled_optim

        model.eval()
        model.requires_grad_ = False
        return model

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        total_loss_dict = {}
        step = 0
        iter_data = tqdm(train_data, total=len(train_data), ncols=None) if show_progress else train_data

        for batch_idx, batch in enumerate(iter_data):
            batch = to_device(batch, self.device)

            # Forward
            output = self.model(*(batch[2:]))

            # Calculate Loss
            losses = self.loss(batch, output)
            total_loss = losses[0] / self.config["optimizer"]["grad_acc_step"]
            total_loss.backward()

            if (step + 1) % self.config["optimizer"]["grad_acc_step"] == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config["optimizer"]["grad_clip_thresh"])
                self.optimizer.step_and_update_lr()
                self.optimizer.zero_grad()

            # Log losses
            losses = [l.item() for l in losses]
            total_loss_dict.update({f"loss_{i}": losses[i] for i in range(len(losses))})

            step += 1

        average_loss_dict = {key: value / step for key, value in total_loss_dict.items()}
        return average_loss_dict
    
class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
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

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

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
    
    