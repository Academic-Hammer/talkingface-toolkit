# load packages
import random
import yaml
import time
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import traceback
import warnings
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

#from meldataset import build_dataloader

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from talkingface.model.text_to_speech_talkingface.StyleTTS2 import *
from talkingface.data.dataset.StyleTTS2_dataset import *
from talkingface.trainer import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer
#from talkingface.config import Config


#华为云上支持的版本和该框架版本有些不兼容，有报错，这里改成手动配置各个部分
def run(
        model,
        dataset,
        config_file_list=None,
        config_dict=None,
        saved=True,
        evaluate_model_file=None
):
    #手动设置配置文件
    config = yaml.safe_load(open('./talkingface/properties/model/StyleTTS2.yml'))
    batch_size = config.get('batch_size', 10)

    epochs = config.get('epochs_2nd', 200)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    max_len = config.get('max_len', 200)
    
    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = 'cuda'

    
    #手动加载数据集，使用talkingface.data.dataset.StyleTTS2_dataset中的build_dataloader
    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})
    
    #手动加载model，使用talkingface.model.text_to_speech.StyleTTS2中的StyleTTS2
    model_abs = StyleTTS2()
    model=model_abs.loadmodel(config)

    #手动加载trainer
    trainer = StyleTTS2Trainer(config,model)

    #装入数据并训练评估
    trainer.fit(train_dataloader,val_dataloader)

   








'''

def run_talkingface(
        model=None,
        dataset=None,
        config_file_list=None,
        config_dict=None,
        saved=True,
        queue=None,
        evaluate_model_file=None
):
    """A fast running api, which include the complete process of training and testing a model on a specified dataset
    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """

    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    #data processing
    # print(not (os.listdir(config['preprocessed_root'])))
    if config['need_preprocess'] and (not (os.path.exists(config['preprocessed_root'])) or not (os.listdir(config['preprocessed_root']))):
        get_preprocess(config['dataset'])(config).run()

    train_dataset, val_dataset = create_dataset(config)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_data_loader = data_utils.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # load model
    model = get_model(config["model"])(config).to(config["device"])
    logger.info(model)

    trainer = get_trainer(config["model"])(config, model)

    # model training
    if config['train']:
        trainer.fit(train_data_loader, val_data_loader, saved=saved, show_progress=config["show_progress"])
        # print(1)

    if not config['train'] and evaluate_model_file is None:
        print("error: no model file to evaluate without training")
        return
    # model evaluating
    trainer.evaluate(model_file = evaluate_model_file)

    


'''