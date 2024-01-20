from talkingface.trainer.trainer import Trainer
import tqdm

class FastSpeech2Trainer(Trainer):
    def __init__(self, config, model):
        super(FastSpeech2Trainer, self).__init__(config, model)
        self.loss = FastSpeech2Loss(config["preprocess_config"], config["model_config"]).to(self.device)
        self.vocoder = get_vocoder(config["model_config"], self.device)

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