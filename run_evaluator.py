from argparse import ArgumentParser, Namespace
import torch
from talkingface.evaluator.evaluator import Evaluator
import yaml 
import sys


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-attr', '-a', default="./autodl-fs/preprocessed_data/attr.pkl")
    parser.add_argument('-config', '-c', default='./saved/adaptive_vc/vctk_model.config.yaml', help='config file path')
    parser.add_argument('-model', '-m', default ='./saved/adaptive_vc/vctk_model.ckpt', help='model path')
    parser.add_argument('-source', '-s', default='./autodl-fs/VCTK-Corpus/wav48/p231/p231_006.wav')
    parser.add_argument('-target', '-t', default='./autodl-fs/VCTK-Corpus/wav48/p252/p253_032.wav')
    parser.add_argument('-output', '-o', default='./result/evaluate/eval.wav')
    parser.add_argument('-sample_rate', '-sr', help='sample rate', default=24000, type=int)
    args = parser.parse_args()
    # load config file 
    with open(args.config) as f:
        config = yaml.safe_load(f)
    evaluator = Evaluator(config=config, args=args)
    evaluator.inference_from_path()