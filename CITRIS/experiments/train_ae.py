"""
Run file to train an autoencoder.
"""
import jsonargparse

from myutil import add_id_ood_common_args
from pretorch_util import assign_visible_gpus
assign_visible_gpus(free_mem_based=True)
import torch.utils.data as data
import pytorch_lightning as pl

import sys
sys.path.append('../')
from models.ae import Autoencoder
from experiments.datasets import get_DataClass_for_datadir
from experiments.utils import train_model, print_params, maybe_minify_args

def main():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="configuration file *.yml. Can be overriden with direct args",
                        action=jsonargparse.ActionConfigFile)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--c_hid', type=int, default=64)
    parser.add_argument('--num_latents', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--noise_level', type=float, default=0.05)
    parser.add_argument('--regularizer_weight', type=float, default=1e-6)
    parser.add_argument('--logger_name', type=str, default='')
    parser.add_argument('--files_to_save', type=str, nargs='+', default='')
    parser.add_argument('--max_dataset_size', type=int, default=-1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    add_id_ood_common_args(parser)
    args = parser.parse_args()
    maybe_minify_args(args)
    pl.seed_everything(args.seed)

    print('Loading datasets...')
    DataClass = get_DataClass_for_datadir(args.data_dir)
    if args.wandb_tags is None:
        args.wandb_tags = []
    args.DataClass = DataClass
    if 'complexpong' in args.data_dir:
        args.wandb_tags.append('og_pong_ID_data')
    
    train_dataset = DataClass(
        data_folder=args.data_dir, split='train', single_image=True, seq_len=1,max_dataset_size=args.max_dataset_size,mini=args.mini)
    val_dataset = DataClass(
        data_folder=args.data_dir, split='val', single_image=True, seq_len=1,max_dataset_size=args.max_dataset_size,mini=args.mini)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)
    print(f'Length training dataset: {len(train_dataset)} / Train loader: {len(train_loader)}')
    print(f'Length val dataset: {len(val_dataset)} / Test loader: {len(val_loader)}')

    args.max_iters = args.max_epochs * len(train_loader)
    model_args = vars(args)
    model_args['img_width'] = train_dataset.get_img_width()
    if hasattr(train_dataset, 'get_inp_channels'):
        model_args['c_in'] = train_dataset.get_inp_channels()
    print(f'Image size: {model_args["img_width"]}')
    model_class = Autoencoder
    logger_name = f'AE_{args.num_latents}l_{args.c_hid}hid'
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name

    print_params(logger_name, model_args)

    train_model(model_class=model_class,
                train_loader=train_loader,
                val_loaders_dict={'ae': val_loader},
                progress_bar_refresh_rate=0 if args.cluster else 1,
                logger_name=logger_name,
                gradient_clip_val=0.1,
                **model_args)


if __name__ == '__main__':
    main()