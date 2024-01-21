'''
Description: 
Author: Fu Yuxuan
Date: 2024-01-15 20:16:39
LastEditTime: 2024-01-19 22:21:57
'''
from argparse import ArgumentParser, Namespace
import torch
from talkingface.trainer.trainer import Trainer
import yaml 
import sys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="Adaptive_VC", help="name of models")
    parser.add_argument("--dataset", type=str, default="VCTK", help="name of datasets")
    parser.add_argument('-config', '-c', default='./talkingface/properties/model/adaptive-VC.yaml')
    parser.add_argument('-data_dir', '-d', default='./autodl-fs/preprocessed_data')
    parser.add_argument('-train_set', default='train_128')
    parser.add_argument('-train_index_file', default='train_samples_128.json')
    parser.add_argument('-logdir', default='./tf-logs/')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_opt', action='store_true')
    parser.add_argument('-store_model_path', default='./saved/adaptive_vc/vctk_model')
    parser.add_argument('-load_model_path', default='./saved/adaptive_vc/vctk_model')
    parser.add_argument('-summary_steps', default=500, type=int)
    parser.add_argument('-save_steps', default=5000, type=int)
    parser.add_argument('-tag', '-t', default='init')
    parser.add_argument('-iters', default=20000, type=int)

    args = parser.parse_args()
    
    # load config file 
    with open(args.config) as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config=config, args=args)

    if args.iters > 0:
        trainer.train(n_iterations=args.iters)