import logging
import os

import sys
import torch.distributed as dist
from collections.abc import MutableMapping
from logging import getLogger
import os
from torch.utils import data as data_utils
from ray import tune

from talkingface.config import Config

from talkingface.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
    get_preprocess,
    create_dataset
)

def run(
        model,
        dataset,
        config_file_list=None,
        config_dict=None,
        saved=True,
        evaluate_model_file=None
):
    
    res = run_talkingface(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
        saved=saved,
        evaluate_model_file=evaluate_model_file,
    )
    
    return res

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

    


