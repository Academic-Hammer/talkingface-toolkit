import os

from logging import getLogger
from time import time
#import dlib, json, subprocess
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
#train   talkingface.data.dataset.data_objects     talkingface.utils.voice_conversion_talkingface
from talkingface.data.dataset.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from talkingface.utils.voice_conversion_talkingface.params_model import *
from talkingface.model.voice_conversion.diffvc import SpeakerEncoder
from talkingface.utils.voice_conversion_talkingface.DiffVC_speaker_encoder_utils import Profiler as Profiler
from pathlib import Path

from torch.utils.data import DataLoader

from talkingface.data.dataset.diffvc_dataset import diffvcDataset, VCDecBatchCollate,VCDecDataset
from talkingface.model.voice_conversion.diffvc import diffvc
from talkingface.model.voice_conversion.diffvc import FastGL
from talkingface.utils.utils import save_plot, save_audio

#visualization
from talkingface.data.dataset.data_objects.speaker_verification_dataset import SpeakerVerificationDataset
from datetime import datetime
from time import perf_counter as timer
import matplotlib.pyplot as plt
import numpy as np
import talkingface.properties.model.params as params

from talkingface.model.voice_conversion.diffvc import diffvc
# import webbrowser
#import visdom
#import umap
#import talkingface.properties.model.diffvctrain_params as params

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
#from talkingface.data.dataprocess.wav2lip_process import Wav2LipAudio
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

class diffvcTrainer(Trainer):
    def __init__(self, config, model):
        super(diffvcTrainer, self).__init__(config, model)
        self.optimizer = config["optimizer"]
        self.train_loader = config["train_loader"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.save_every = config["save_every"]
        self.log_dir = config["log_dir"]
        torch.manual_seed(config["random_seed"])
        np.random.seed(config["random_seed"])
        n_mels = params.n_mels
        sampling_rate = params.sampling_rate
        n_fft = params.n_fft
        hop_size = params.hop_size

        random_seed = params.seed
        test_size = params.test_size

        data_dir=params.data_dir
        val_file=params.val_file
        exc_file=params.exc_file

        log_dir = 'logs_dec'
        enc_dir = 'logs_enc'
        epochs = 10
        batch_size = 32
        learning_rate = 1e-4
        save_every = 1


     

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        os.makedirs(log_dir, exist_ok=True)

        print('Initializing data loaders...')
        train_set = VCDecDataset(data_dir, val_file, exc_file)
        collate_fn = VCDecBatchCollate()
        train_loader = DataLoader(train_set, batch_size=batch_size, 
                                    collate_fn=collate_fn, num_workers=4, drop_last=True)

        print('Initializing and loading models...')
        #fgl = FastGL(n_mels, sampling_rate, n_fft, hop_size).cuda()
        fgl = FastGL(n_mels, sampling_rate, n_fft, hop_size)
            
            #.cuda()
        model.load_encoder(os.path.join(enc_dir, 'enc.pt'))

        print('Encoder:')
        print(model.encoder)
       # print('Number of parameters = %.2fm\n' % (model.encoder.nparams/1e6))
        print('Decoder:')
        print(model.decoder)
       # print('Number of parameters = %.2fm\n' % (model.decoder.nparams/1e6))

        print('Initializing optimizers...')
        optimizer = torch.optim.Adam(params=model.decoder.parameters(), lr=learning_rate)

        print('Start training.')
        torch.backends.cudnn.benchmark = True
        iteration = 0
        for epoch in range(1, epochs + 1):
            print(f'Epoch: {epoch} [iteration: {iteration}]')
            model.train()
            losses = []
            for batch in tqdm(train_loader, total=len(train_set)//batch_size):
                    mel, mel_ref = batch['mel1'].cuda(), batch['mel2']
                    #.cuda()
                    c, mel_lengths = batch['c'].cuda(), batch['mel_lengths']
                    #.cuda()
                    model.zero_grad()
                    loss = model.compute_loss(mel, mel_lengths, mel_ref, c)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1)
                    optimizer.step()
                    losses.append(loss.item())
                    iteration += 1

            losses = np.asarray(losses)
            msg = 'Epoch %d: loss = %.4f\n' % (epoch, np.mean(losses))
            print(msg)
            with open(f'{log_dir}/train_dec.log', 'a') as f:
                    f.write(msg)
            losses = []

            if epoch % save_every > 0:
                continue

            model.eval()
            print('Inference...\n')
            with torch.no_grad():
                mels = train_set.get_valid_dataset()
                for i, (mel, c) in enumerate(mels):
                    if i >= test_size:
                            break
                    mel = mel.unsqueeze(0).float()
                        #.cuda()
                    c = c.unsqueeze(0).float()
                        #.cuda()
                    mel_lengths = torch.LongTensor([mel.shape[-1]])
                        #.cuda()
                    mel_avg, mel_rec = model(mel, mel_lengths, mel, mel_lengths, c, 
                                                n_timesteps=100)
                    if epoch == save_every:
                            save_plot(mel.squeeze().cpu(), f'{log_dir}/original_{i}.png')
                            audio = fgl(mel)
                            save_audio(f'{log_dir}/original_{i}.wav', sampling_rate, audio)
                    save_plot(mel_avg.squeeze().cpu(), f'{log_dir}/average_{i}.png')
                    audio = fgl(mel_avg)
                    save_audio(f'{log_dir}/average_{i}.wav', sampling_rate, audio)
                    save_plot(mel_rec.squeeze().cpu(), f'{log_dir}/reconstructed_{i}.png')
                    audio = fgl(mel_rec)
                    save_audio(f'{log_dir}/reconstructed_{i}.wav', sampling_rate, audio)

            print('Saving model...\n')
            ckpt = model.state_dict()
            torch.save(ckpt, f=f"{log_dir}/vc_{epoch}.pt")                      


class diffVC_encoder_train:
    def sync(device: torch.device):
        # FIXME
        return
        # For correct profiling (cuda operations are async)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
              backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
              no_visdom: bool):
        # Create a dataset and a dataloader
        dataset = SpeakerVerificationDataset(clean_data_root)
        loader = SpeakerVerificationDataLoader(
            dataset,
            diffVC_encoder_train.speakers_per_batch,
            diffVC_encoder_train.utterances_per_speaker,
            num_workers=8,
        )

        # Setup the device on which to run the forward pass and the loss. These can be different,
        # because the forward pass is faster on the GPU whereas the loss is often (depending on your
        # hyperparameters) faster on the CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # FIXME: currently, the gradient is None if loss_device is cuda
        loss_device = torch.device("cpu")

        # Create the model and the optimizer
        model = SpeakerEncoder(device, loss_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=diffVC_encoder_train.learning_rate_init)
        init_step = 1

        # Configure file path for the model
        state_fpath = models_dir.joinpath(run_id + ".pt")
        backup_dir = models_dir.joinpath(run_id + "_backups")

        # Load any existing model
        if not force_restart:
            if state_fpath.exists():
                print("Found existing model \"%s\", loading it and resuming training." % run_id)
                checkpoint = torch.load(state_fpath)
                init_step = checkpoint["step"]
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                optimizer.param_groups[0]["lr"] = diffVC_encoder_train.learning_rate_init
            else:
                print("No model \"%s\" found, starting training from scratch." % run_id)
        else:
            print("Starting the training from scratch.")
        model.train()

        # Initialize the visualization environment
        vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
        vis.log_dataset(dataset)
        vis.log_params()
        device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        vis.log_implementation({"Device": device_name})

        # Training loop
        profiler = Profiler(summarize_every=10, disabled=False)
        for step, speaker_batch in enumerate(loader, init_step):
            profiler.tick("Blocking, waiting for batch (threaded)")

            # Forward pass
            inputs = torch.from_numpy(speaker_batch.data).to(device)
            diffVC_encoder_train.sync(device)
            profiler.tick("Data to %s" % device)
            embeds = model(inputs)
            diffVC_encoder_train.sync(device)
            profiler.tick("Forward pass")
            embeds_loss = embeds.view((diffVC_encoder_train.speakers_per_batch, diffVC_encoder_train.utterances_per_speaker, -1)).to(loss_device)
            loss, eer = model.loss(embeds_loss)
            diffVC_encoder_train.sync(loss_device)
            profiler.tick("Loss")

            # Backward pass
            model.zero_grad()
            loss.backward()
            profiler.tick("Backward pass")
            model.do_gradient_ops()
            optimizer.step()
            profiler.tick("Parameter update")

            # Update visualizations
            # learning_rate = optimizer.param_groups[0]["lr"]
            vis.update(loss.item(), eer, step)

            # Draw projections and save them to the backup folder
            if umap_every != 0 and step % umap_every == 0:
                print("Drawing and saving projections (step %d)" % step)
                backup_dir.mkdir(exist_ok=True)
                projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
                embeds = embeds.detach().cpu().numpy()
                vis.draw_projections(embeds, diffVC_encoder_train.utterances_per_speaker, step, projection_fpath)
                vis.save()

            # Overwrite the latest version of the model
            if save_every != 0 and step % save_every == 0:
                print("Saving the model (step %d)" % step)
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, state_fpath)

            # Make a backup
            if backup_every != 0 and step % backup_every == 0:
                print("Making a backup (step %d)" % step)
                backup_dir.mkdir(exist_ok=True)
                backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, backup_fpath)

            profiler.tick("Extras (visualizations, saving)")


class Visualizations:
    colormap = np.array([
        [76, 255, 0],
        [0, 127, 70],
        [255, 0, 0],
        [255, 217, 38],
        [0, 135, 255],
        [165, 0, 165],
        [255, 167, 255],
        [0, 255, 255],
        [255, 96, 38],
        [142, 76, 0],
        [33, 0, 127],
        [0, 0, 0],
        [183, 183, 183],
    ], dtype=np.float) / 255

    def __init__(self, env_name=None, update_every=10, server="http://localhost", disabled=False):
        # Tracking data
        self.last_update_timestamp = timer()
        self.update_every = update_every
        self.step_times = []
        self.losses = []
        self.eers = []
        print("Updating the visualizations every %d steps." % update_every)

        # If visdom is disabled TODO: use a better paradigm for that
        self.disabled = disabled
        if self.disabled:
            return

            # Set the environment name
        now = str(datetime.now().strftime("%d-%m %Hh%M"))
        if env_name is None:
            self.env_name = now
        else:
            self.env_name = "%s (%s)" % (env_name, now)

        # Connect to visdom and open the corresponding window in the browser
        try:
            self.vis = visdom.Visdom(server, env=self.env_name, raise_exceptions=True)
        except ConnectionError:
            raise Exception("No visdom server detected. Run the command \"visdom\" in your CLI to "
                            "start it.")
        # webbrowser.open("http://localhost:8097/env/" + self.env_name)

        # Create the windows
        self.loss_win = None
        self.eer_win = None
        # self.lr_win = None
        self.implementation_win = None
        self.projection_win = None
        self.implementation_string = ""

    def log_params(self):
        if self.disabled:
            return
        from talkingface.utils.voice_conversion_talkingface import params_data
        from talkingface.utils.voice_conversion_talkingface import params_model
        param_string = "<b>Model parameters</b>:<br>"
        for param_name in (p for p in dir(params_model) if not p.startswith("__")):
            value = getattr(params_model, param_name)
            param_string += "\t%s: %s<br>" % (param_name, value)
        param_string += "<b>Data parameters</b>:<br>"
        for param_name in (p for p in dir(params_data) if not p.startswith("__")):
            value = getattr(params_data, param_name)
            param_string += "\t%s: %s<br>" % (param_name, value)
        self.vis.text(param_string, opts={"title": "Parameters"})

    def log_dataset(self, dataset: SpeakerVerificationDataset):
        if self.disabled:
            return
        dataset_string = ""
        dataset_string += "<b>Speakers</b>: %s\n" % len(dataset.speakers)
        dataset_string += "\n" + dataset.get_logs()
        dataset_string = dataset_string.replace("\n", "<br>")
        self.vis.text(dataset_string, opts={"title": "Dataset"})

    def log_implementation(self, params):
        if self.disabled:
            return
        implementation_string = ""
        for param, value in params.items():
            implementation_string += "<b>%s</b>: %s\n" % (param, value)
            implementation_string = implementation_string.replace("\n", "<br>")
        self.implementation_string = implementation_string
        self.implementation_win = self.vis.text(
            implementation_string,
            opts={"title": "Training implementation"}
        )

    def update(self, loss, eer, step):
        # Update the tracking data
        now = timer()
        self.step_times.append(1000 * (now - self.last_update_timestamp))
        self.last_update_timestamp = now
        self.losses.append(loss)
        self.eers.append(eer)
        print(".", end="")

        # Update the plots every <update_every> steps
        if step % self.update_every != 0:
            return
        time_string = "Step time:  mean: %5dms  std: %5dms" % \
                      (int(np.mean(self.step_times)), int(np.std(self.step_times)))
        print("\nStep %6d   Loss: %.4f   EER: %.4f   %s" %
              (step, np.mean(self.losses), np.mean(self.eers), time_string))
        if not self.disabled:
            self.loss_win = self.vis.line(
                [np.mean(self.losses)],
                [step],
                win=self.loss_win,
                update="append" if self.loss_win else None,
                opts=dict(
                    legend=["Avg. loss"],
                    xlabel="Step",
                    ylabel="Loss",
                    title="Loss",
                )
            )
            self.eer_win = self.vis.line(
                [np.mean(self.eers)],
                [step],
                win=self.eer_win,
                update="append" if self.eer_win else None,
                opts=dict(
                    legend=["Avg. EER"],
                    xlabel="Step",
                    ylabel="EER",
                    title="Equal error rate"
                )
            )
            if self.implementation_win is not None:
                self.vis.text(
                    self.implementation_string + ("<b>%s</b>" % time_string),
                    win=self.implementation_win,
                    opts={"title": "Training implementation"},
                )

        # Reset the tracking
        self.losses.clear()
        self.eers.clear()
        self.step_times.clear()

    def draw_projections(self, embeds, utterances_per_speaker, step, out_fpath=None,
                         max_speakers=10):
        max_speakers = min(max_speakers, len(Visualizations.colormap))
        embeds = embeds[:max_speakers * utterances_per_speaker]

        n_speakers = len(embeds) // utterances_per_speaker
        ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
        colors = [Visualizations.colormap[i] for i in ground_truth]

        reducer = umap.UMAP()
        projected = reducer.fit_transform(embeds)
        plt.scatter(projected[:, 0], projected[:, 1], c=colors)
        plt.gca().set_aspect("equal", "datalim")
        plt.title("UMAP projection (step %d)" % step)
        if not self.disabled:
            self.projection_win = self.vis.matplot(plt, win=self.projection_win)
        if out_fpath is not None:
            plt.savefig(out_fpath)
        plt.clf()

    def save(self):
        if not self.disabled:
            self.vis.save([self.env_name])


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
    