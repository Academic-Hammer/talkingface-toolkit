import argparse
from talkingface.quick_start import run
import torch
if __name__ == "__main__":
    torch.cuda.is_available()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default=None, help="name of datasets"
    )
    parser.add_argument("--evaluate_model_file", type=str, default=None, help="The model file you want to evaluate")
    parser.add_argument("--config_files", type=str, default=None, help="config files")


    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    run(
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        evaluate_model_file=args.evaluate_model_file
    )