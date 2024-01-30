

import os
import argparse

from talkingface.config.config import Config
from talkingface.model.image_driven_talkingface.styleheat.utils.logging import init_logging, make_logging_dir
from talkingface.model.image_driven_talkingface.styleheat.utils.trainer import get_model_optimizer_and_scheduler_with_pretrain, set_random_seed, get_trainer, get_model_optimizer_and_scheduler
from talkingface.model.image_driven_talkingface.styleheat.utils.distributed import init_dist
from talkingface.model.image_driven_talkingface.styleheat.utils.distributed import master_only_print as print


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--checkpoints_dir', default='result', help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_known_args()
    return args


def fit(args,
        opt,
        train_data,
        valid_data,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None):
    # get training options


    print('Single GPU Training.')


    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    make_logging_dir(logdir, date_uid)
    os.system(f'cp {args.config} {opt.logdir}')
    # create a dataset
    train_dataset = train_data
    val_dataset = valid_data

    # create a model
    net_G, net_G_ema, opt_G, sch_G = get_model_optimizer_and_scheduler_with_pretrain(opt)

    trainer = get_trainer(opt, net_G, net_G_ema, opt_G, sch_G, train_dataset)
    current_epoch, current_iteration = trainer.load_checkpoint(opt, args.which_iter)

    # training flag
    if args.debug:
        trainer.test_everything(train_dataset, val_dataset, current_epoch, current_iteration)
        exit()

    # Start training.
    for epoch in range(current_epoch, opt.max_epoch):
        print('Epoch {} ...'.format(epoch))
        if not args.single_gpu:
            train_dataset.sampler.set_epoch(current_epoch)
        trainer.start_of_epoch(current_epoch)
        for it, data in enumerate(train_dataset):
            data = trainer.start_of_iteration(data, current_iteration)
            trainer.optimize_parameters(data)
            current_iteration += 1
            trainer.end_of_iteration(data, current_epoch, current_iteration)

            if current_iteration >= opt.max_iter:
                print('Done with training!!!')
                break
        current_epoch += 1
        trainer.end_of_epoch(data, val_dataset, current_epoch, current_iteration)
        trainer.test(val_dataset, output_dir=os.path.join(logdir, 'evaluation'), test_limit=10)


