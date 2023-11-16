"""
General training function with PyTorch Lightning
"""

import os
import json

import torch
import torch.utils.data as data
# from torch.utils.data import BatchSampler, RandomSampler

from constants import CITRIS_CKPT_ROOT, PONG_OOD_ROOT, TC3DI_OOD_ROOT, AE_SHAPES_CKPTS, AE_PONG_CKPTS
import wandb
import pytorch_lightning as pl
from experiments.ood_utils import update_wandb_config
from models.shared import AutoregNormalizingFlow
from models.shared.callbacks import ManualCheckpointCallback
from models.shared.next_step_predictor import NextStepPredictor
from pytorch_lightning.callbacks import ModelCheckpoint
from shutil import copyfile
import pandas as pd
import sys

from util import get_ckptpath_for_wid

sys.path.append('../')
from experiments.datasets import BallInBoxesDataset, InterventionalPongDataset, Causal3DDataset, VoronoiDataset, PinballDataset

def get_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_default_parser(parser):
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    # parser.add_argument('--batch_size', type=int, default=512)
    # parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--exclude_vars', type=str, nargs='+', default=None)
    parser.add_argument('--exclude_objects', type=int, nargs='+', default=None)
    parser.add_argument('--coarse_vars', action='store_true')
    parser.add_argument('--data_img_width', type=int, default=-1)
    parser.add_argument('--seq_len', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--imperfect_interventions', action='store_true')
    parser.add_argument('--logger_name', type=str, default='')
    parser.add_argument('--files_to_save', type=str, nargs='+', default='')
    return parser

def load_datasets(args):
    pl.seed_everything(args.seed)
    print('Loading datasets...')
    if 'require_imgs' in args and not args.require_imgs:
        require_imgs = False
    else:
        require_imgs = True
    # region Kind of data (ball in boxes, pong, etc.)
    if any([s in args.data_dir for s in ['causal3d', 'pong']]):
        train_args = {}
        if 'max_train_samples' in vars(args) and args.max_train_samples is not None:
            train_args |= {'max_dataset_size': args.max_train_samples}
    if 'causal3d' not in args.data_dir:
        if args.skip_cma is None:
            args.skip_cma = True # default for non-causal3d
    if 'ball_in_boxes' in args.data_dir:
        data_name = 'ballinboxes'
        DataClass = BallInBoxesDataset
        dataset_args = {}
        test_args = lambda train_set: {'causal_vars': train_set.target_names_l}
    elif 'pong' in args.data_dir:
        data_name = 'pong'
        DataClass = InterventionalPongDataset
        dataset_args = {}
        test_args = lambda train_set: {'causal_vars': train_set.target_names_l}
        if args.ood_data_dir is None:
            args.ood_data_dir = PONG_OOD_ROOT
    elif 'causal3d' in args.data_dir:
        data_name = 'causal3d'
        DataClass = Causal3DDataset
        dataset_args = {'coarse_vars': args.coarse_vars, 'exclude_vars': args.exclude_vars, 'exclude_objects': args.exclude_objects, 'require_imgs': require_imgs}
        test_args = lambda train_set: {'causal_vars': train_set.full_target_names}
        if args.ood_data_dir is None:
            args.ood_data_dir = TC3DI_OOD_ROOT
        if args.skip_cma is None:
            args.skip_cma = False # default for causal3d
    elif 'voronoi' in args.data_dir:
        extra_name = args.data_dir.split('voronoi')[-1]
        if extra_name[-1] == '/':
            extra_name = extra_name[:-1]
        extra_name = extra_name.replace('/', '_')
        data_name = 'voronoi' + extra_name
        DataClass = VoronoiDataset
        dataset_args = {}
        test_args = lambda train_set: {'causal_vars': train_set.target_names_l}
    elif 'pinball' in args.data_dir:
        data_name = 'pinball' + args.data_dir.split('pinball')[-1].replace('/','')
        DataClass = PinballDataset
        dataset_args = {}
        test_args = lambda train_set: {'causal_vars': train_set.target_names_l}
    else:
        assert False, f'Unknown data class for {args.data_dir}'
    # endregion
    dataset_args |= {'mini': args.mini}
    train_ds_args = {
        'data_folder': args.data_dir, 'split': 'train', 'single_image': False, 'triplet': False, 'seq_len': args.seq_len, 'deconf': args.deconf, **dataset_args, **train_args
    }
    if 'return_latents_train' in vars(args) and args.return_latents_train: # Nathan
        # train_dataset = DataClass(
        #     data_folder=args.data_dir, split='train', single_image=False, triplet=False, seq_len=args.seq_len, return_latents=True, deconf=args.deconf, **dataset_args, **train_args)
        train_ds_args['return_latents'] = True
    if 'train_ae' in vars(args) and args.train_ae:
        train_ds_args['include_og_imgs'] = True
    # if args.cheat_cfd:
    if 'cheat_cfd' in vars(args) and args.cheat_cfd:
        train_ds_args['return_latents'] = True
    train_dataset = DataClass(**train_ds_args)

    if not args.no_alignment_measuring:
        val_indep_dataset = DataClass(
            data_folder=args.data_dir, split='val_indep', single_image=True, triplet=False, return_latents=True, **dataset_args, **test_args(train_dataset))
        val_triplet_dataset = DataClass(
            data_folder=args.data_dir, split='val', single_image=False, triplet=True, return_latents=True, **dataset_args, **test_args(train_dataset))
        test_indep_dataset = DataClass(
            data_folder=args.data_dir, split='test_indep', single_image=True, triplet=False, return_latents=True, **dataset_args, **test_args(train_dataset))
        test_triplet_dataset = DataClass(
            data_folder=args.data_dir, split='test', single_image=False, triplet=True, return_latents=True, **dataset_args, **test_args(train_dataset))
        if not args.skip_cma:
            test_cma_fp_dataset = DataClass(
                data_folder=args.data_dir, split='test', single_image=False, triplet=False, include_og_imgs= args.include_og_imgs, **dataset_args, **test_args(train_dataset),
                cma=True)
            val_cma_fp_dataset = DataClass(
                data_folder=args.data_dir, split='val', single_image=False, triplet=False, include_og_imgs= args.include_og_imgs, **dataset_args, **test_args(train_dataset),
                cma=True)
            test_cma_tlp_dataset = DataClass(
                data_folder=args.data_dir, split='test', single_image=False, triplet=False, return_latents=True, **dataset_args, **test_args(train_dataset),
                cma=True)
            val_cma_tlp_dataset = DataClass(
                data_folder=args.data_dir, split='val', single_image=False, triplet=False, return_latents=True, **dataset_args, **test_args(train_dataset),
                cma=True)
            train_cma_fp_dataset = DataClass(
                data_folder=args.data_dir, split='train', single_image=False, triplet=False, include_og_imgs= args.include_og_imgs, **dataset_args, **test_args(train_dataset),
                cma=True, **train_args)
            train_cma_tlp_dataset = DataClass(
                data_folder=args.data_dir, split='train', single_image=False, triplet=False, return_latents=True, **dataset_args, **test_args(train_dataset),
                cma=True, **train_args)
    val_fp_dataset = DataClass(
        data_folder=args.data_dir, split='val', single_image=False, triplet=False, seq_len=args.seq_len,
        return_latents=True, # added for FramePredictionCallback (Nathan)
        include_og_imgs= args.include_og_imgs, deconf=args.deconf, **dataset_args, **test_args(train_dataset))
    test_fp_dataset = DataClass(
        data_folder=args.data_dir, split='test', single_image=False, triplet=False, seq_len=args.seq_len, include_og_imgs= args.include_og_imgs, deconf=args.deconf, **dataset_args, **test_args(train_dataset))
    val_tlp_dataset = DataClass(
        data_folder=args.data_dir, split='val', single_image=False, triplet=False, seq_len=args.seq_len, return_latents=True, deconf=args.deconf, **dataset_args, **test_args(train_dataset))
    test_tlp_dataset = DataClass(
        data_folder=args.data_dir, split='test', single_image=False, triplet=False, seq_len=args.seq_len, return_latents=True, deconf=args.deconf, **dataset_args, **test_args(train_dataset))

    if args.exclude_objects is not None and data_name == 'causal3d':
        test_indep_dataset = {
            'orig_wo_' + '_'.join([str(o) for o in args.exclude_objects]): test_indep_dataset
        }
        val_indep_dataset = {
            next(iter(test_indep_dataset.keys())): val_indep_dataset
        }
        dataset_args.pop('exclude_objects')
        for o in args.exclude_objects:
            val_indep_dataset[f'exclusive_obj_{o}'] = DataClass(
                                data_folder=args.data_dir, split='val_indep', single_image=True, triplet=False, return_latents=True, exclude_objects=[i for i in range(7) if i != o], **dataset_args, **test_args(train_dataset))
            test_indep_dataset[f'exclusive_obj_{o}'] = DataClass(
                                data_folder=args.data_dir, split='test_indep', single_image=True, triplet=False, return_latents=True, exclude_objects=[i for i in range(7) if i != o], **dataset_args, **test_args(train_dataset))
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    if not args.no_alignment_measuring:
        val_triplet_loader = data.DataLoader(val_triplet_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
        test_triplet_loader = data.DataLoader(test_triplet_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
        if not args.skip_cma:
            # test_cma_loader = data.DataLoader(test_cma_dataset, batch_size=args.batch_size,
            #                                 shuffle=False, drop_last=False, num_workers=args.num_workers)
            # val_cma_loader = data.DataLoader(val_cma_dataset, batch_size=args.batch_size,
            #                                 shuffle=False, drop_last=False, num_workers=args.num_workers)
            test_cma_fp_loader = data.DataLoader(test_cma_fp_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
            val_cma_fp_loader = data.DataLoader(val_cma_fp_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
            test_cma_tlp_loader = data.DataLoader(test_cma_tlp_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
            val_cma_tlp_loader = data.DataLoader(val_cma_tlp_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
            train_cma_fp_loader = data.DataLoader(train_cma_fp_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
            train_cma_tlp_loader = data.DataLoader(train_cma_tlp_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    val_fp_loader = data.DataLoader(val_fp_dataset, batch_size=args.batch_size,
                                    shuffle=False, drop_last=True, num_workers=args.num_workers)
    test_fp_loader = data.DataLoader(test_fp_dataset, batch_size=args.batch_size,
                                     shuffle=False, drop_last=True, num_workers=args.num_workers)
    val_tlp_loader = data.DataLoader(val_tlp_dataset, batch_size=args.batch_size,
                                        shuffle=False, drop_last=True, num_workers=args.num_workers)
    test_tlp_loader = data.DataLoader(test_tlp_dataset, batch_size=args.batch_size,
                                        shuffle=False, drop_last=True, num_workers=args.num_workers)


    # region printing dataset sizes
    print(f'Training dataset size: {len(train_dataset)} / {len(train_loader)}')
    if not args.no_alignment_measuring:
        print(f'Val triplet dataset size: {len(val_triplet_dataset)} / {len(val_triplet_loader)}')
        if isinstance(val_indep_dataset, dict):
            print(f'Val correlation dataset sizes: { {key: len(val_indep_dataset[key]) for key in val_indep_dataset} }')
        else:
            print(f'Val correlation dataset size: {len(val_indep_dataset)}')
        print(f'Test triplet dataset size: {len(test_triplet_dataset)} / {len(test_triplet_loader)}')
        if isinstance(test_indep_dataset, dict):
            print(f'Test correlation dataset sizes: { {key: len(test_indep_dataset[key]) for key in test_indep_dataset} }')
        else:
            print(f'Test correlation dataset size: {len(test_indep_dataset)}')
    # endregion

    # datasets = {
    #     'train': train_dataset,
    #     'val': val_indep_dataset,
    #     'val_triplet': val_triplet_dataset,
    #     'test': test_indep_dataset,
    #     'test_triplet': test_triplet_dataset,
    #     'val_fp': val_fp_dataset,
    #     'test_fp': test_fp_dataset,
    #     'val_tlp': val_tlp_dataset,
    #     'test_tlp': test_tlp_dataset,
    # }
    datasets = {
        'train': train_dataset,
        'val_fp': val_fp_dataset,
        'test_fp': test_fp_dataset,
        'val_tlp': val_tlp_dataset,
        'test_tlp': test_tlp_dataset,
    }
    if not args.no_alignment_measuring:
        datasets |= {
            'val_triplet': val_triplet_dataset,
            'test_triplet': test_triplet_dataset,
            'val_corr': val_indep_dataset,
            'test_corr': test_indep_dataset,
        }
    if not args.skip_cma:
        datasets |= {
        'val_cma_fp': val_cma_fp_dataset,
        'test_cma_fp': test_cma_fp_dataset,
        'val_cma_tlp': val_cma_tlp_dataset,
        'test_cma_tlp': test_cma_tlp_dataset,
        'train_cma_fp': train_cma_fp_dataset,
        'train_cma_tlp': train_cma_tlp_dataset,
    }
    # data_loaders = {
    #     'train': train_loader,
    #     'val_triplet': val_triplet_loader,
    #     'test_triplet': test_triplet_loader,
    #     'val_fp': val_fp_loader,
    #     'test_fp': test_fp_loader,
    #     'val_tlp': val_tlp_loader,
    #     'test_tlp': test_tlp_loader,
    # }
    data_loaders = {
        'train': train_loader,
        'val_fp': val_fp_loader,
        'test_fp': test_fp_loader,
        'val_tlp': val_tlp_loader,
        'test_tlp': test_tlp_loader,
    }
    if not args.no_alignment_measuring:
        data_loaders |= {
            'val_triplet': val_triplet_loader,
            'test_triplet': test_triplet_loader,
        }
        if not args.skip_cma:
            data_loaders |= {
            'val_cma_fp': val_cma_fp_loader,
            'test_cma_fp': test_cma_fp_loader,
            'val_cma_tlp': val_cma_tlp_loader,
            'test_cma_tlp': test_cma_tlp_loader,
            'train_cma_fp': train_cma_fp_loader,
            'train_cma_tlp': train_cma_tlp_loader,
        }
    return datasets, data_loaders, data_name

def print_params(logger_name, model_args):
    num_chars = max(50, 11+len(logger_name))
    print('=' * num_chars)
    print(f'Experiment {logger_name}')
    print('-' * num_chars)
    for key in sorted(list(model_args.keys())):
        print(f'-> {key}: {model_args[key]}')
    print('=' * num_chars)

def train_model(model_class, train_loader,
                # val_loaders,
                # test_loaders=None,
                val_loaders_dict,
                test_loaders_dict=None,
                logger_name=None,
                max_epochs=200,
                progress_bar_refresh_rate=1,
                check_val_every_n_epoch=1,
                debug=False,
                offline=False,
                op_before_running=None,
                load_pretrained=False,
                # root_dir=None, # Nathan
                files_to_save=None,
                gradient_clip_val=1.0,
                cluster=False,
                callback_kwargs=None,
                seed=42,
                save_last_model=True,
                val_track_metric='val_loss',
                data_dir=None,
                img_callback_freq=50,
                wandb_tags=None,
                DataClass=None,
                prename='',
                postname='',
                notrain=False,
                wandb_id = None,
                wandb_resume_id = None,
                ckpt_path = None,
                detect_anomaly=False,
                no_alignment_measuring=False,
                ewid=None,
                accumulate_grad_batches=1,
                manual_epochs_to_save_at=None,
                fp_log_freq=10,
                **kwargs):
    val_loaders_list = list(val_loaders_dict.values()) # Counts on new python preserving order of dict
    test_loaders_list = None if test_loaders_dict is None else list(test_loaders_dict.values())
    trainer_args = {}
    if wandb_tags is None:
        wandb_tags = []
    if DataClass is not None:
        wandb_tags.append(DataClass.SHORT_NAME)
    if 'anti_align' in kwargs and kwargs['anti_align']:
        wandb_tags.append('anti_align')
    if hasattr(model_class, 'DATA_SETTING'):
        wandb_tags.append(model_class.DATA_SETTING)
    if 'train_on_decorr' in kwargs and kwargs['train_on_decorr']:
        wandb_tags.append('train_on_decorr')
    if 'cma_true_parentage' in kwargs and kwargs['cma_true_parentage']:
        wandb_tags.append('parent_only')

    # isNSP = isinstance(model, NextStepPredictor) # use model_class instead of model
    isNSP = issubclass(model_class, NextStepPredictor)
    if isNSP:
        wandb_tags.append('PT')
    logger = set_logger(kwargs, logger_name, postname, prename, trainer_args, wandb_id, wandb_resume_id, wandb_tags, ewid)
    if progress_bar_refresh_rate == 0:
        trainer_args['enable_progress_bar'] = False
    trainer_args['detect_anomaly'] = detect_anomaly
    trainer_args['accumulate_grad_batches'] = accumulate_grad_batches


    maybe_mini = 'mini_' if kwargs['mini'] else ''
    ckpt_dir_path = os.path.join(CITRIS_CKPT_ROOT, f'{maybe_mini}{logger.experiment.id}_{model_class.__name__}_{DataClass.SHORT_NAME}')

    # Nathan
    fit_args = {}
    if ckpt_path is None:
        if wandb_resume_id is not None:
            fit_args['ckpt_path'] = os.path.join(ckpt_dir_path,'last.ckpt')
    else:
        fit_args['ckpt_path'] = ckpt_path

    if callback_kwargs is None:
        callback_kwargs = dict()
    callbacks = model_class.get_callbacks(exmp_triplet_inputs=  next(iter(val_loaders_dict['triplet'])) if 'triplet' in val_loaders_dict  else None,
                                          exmp_fp_inputs =      next(iter(val_loaders_dict['fp']))      if 'fp' in val_loaders_dict       else None,
                                          cluster=cluster, img_callback_freq=img_callback_freq, no_alignment_measuring=no_alignment_measuring,
                                          fp_loader= val_loaders_dict['fp'] if 'fp' in val_loaders_dict else None, fp_log_freq=fp_log_freq,
                                          **callback_kwargs)
    if not debug:
        callbacks.append(
                ModelCheckpoint(
                    # save_weights_only=True, # Nathan: seems more useful to save the entire model, not just the weights
                                dirpath=ckpt_dir_path,
                                mode="min",
                                monitor=val_track_metric,
                                save_last=save_last_model,
                                every_n_epochs=check_val_every_n_epoch)
            )
        if kwargs['do_ood_logging']:
            infix = '_sar' if kwargs['use_mean_to_autoregress_str'] == 'False' else '_mar'
            for ood_log_nshots in kwargs['ood_log_nsshots']:
                callbacks.append(
                    ModelCheckpoint(
                        filename=f'ood_{ood_log_nshots}_shots_best_{{epoch}}.ckpt',
                        dirpath=ckpt_dir_path,
                        monitor=f'ood_{ood_log_nshots}sh_avg_cnd{infix}_loss',
                        save_top_k=1,
                        every_n_epochs=kwargs['ood_log_freq'],
                        mode='min',
                        save_last=False
                    )
                )

    if manual_epochs_to_save_at is not None:
        callbacks.append(ManualCheckpointCallback(epochs_to_save_at=manual_epochs_to_save_at, ckpt_dir=ckpt_dir_path))
    trainer = pl.Trainer(
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=max_epochs,
                         callbacks=callbacks,
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         gradient_clip_val=gradient_clip_val,
                         log_every_n_steps=kwargs["log_every_n_steps"] if "log_every_n_steps" in kwargs else 50,
                         **trainer_args)
    trainer.logger._default_hp_metric = None

    fix_files_to_save(files_to_save, logger)

    pretrained_filename = get_pretrained_filename(ewid)
    if load_pretrained and (pretrained_filename is not None) and os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        test_model_ckpt(pretrained_filename, model_class, test_loaders_list, trainer) #, val_loaders=val_loaders)
        # print("USING VAL LOADERS INSTEAD OF TEST LOADERS FOR DEBUGGING","="*100)
        # test_model_ckpt(pretrained_filename, model_class, val_loaders, trainer)
    else:
        if load_pretrained:
            print("Warning: Could not load any pretrained models despite", load_pretrained)
        pl.seed_everything(seed)  # To be reproducible
        model = model_class(DataClass=DataClass,standalone=True,**kwargs) # standalone: to distinguish from when model is part of a larger model
        update_wandb_config(model)
        if op_before_running is not None:
            op_before_running(model)
        if 'debug_xmse_logging' in kwargs and kwargs['debug_xmse_logging']: # temporary
            import glob
            glob_match_str = f'/cw/liir_code/NoCsBack/nathan/p2/CITRIS/experiments/checkpoints/{kwargs["debug_wid"]}_*/manual_epoch_{kwargs["debug_epoch"]}.ckpt'
            ckpt_path = glob.glob(glob_match_str)[0]
            model = model_class.load_from_checkpoint(ckpt_path)
            trainer.model = model
            nfp_cb = [c for c in trainer.callbacks if c.__class__.__name__ == 'FramePredictionCallback'][0]
            trainer.callbacks = [nfp_cb]
            trainer.validate(model, val_loaders_list[0], verbose=True)
            # move model to GPU
            model = model.cuda()
            nfp_cb.log_nfp_results(trainer, model)
            exit()
        if 'debug_ood_logging' in kwargs and kwargs['debug_ood_logging']: # temporary
            import glob
            glob_match_str = f'/cw/liir_code/NoCsBack/nathan/p2/CITRIS/experiments/checkpoints/{kwargs["debug_wid"]}_*/manual_epoch_{kwargs["debug_epoch"]}.ckpt'
            ckpt_path = glob.glob(glob_match_str)[0]
            model = model_class.load_from_checkpoint(ckpt_path)
            trainer.model = model
            ood_cb = [c for c in trainer.callbacks if c.__class__.__name__ == 'FewShotOODCallback'][0]
            trainer.callbacks = [ood_cb]
            ood_cb.log_ood_results(model, trainer, log_with_wandb=True)
            exit()


        if not notrain:
            trainer.fit(model, train_loader, val_loaders_list, **fit_args)
        else:
            # Save the initial model
            trainer.model = model
            trainer.save_checkpoint(ckpt_dir_path + '/epoch=0-step=0.ckpt')
            trainer.checkpoint_callback.best_model_path = ckpt_dir_path + '/epoch=0-step=0.ckpt'

        if test_loaders_list is not None:
            model_paths = [(trainer.checkpoint_callback.best_model_path, "best")]
            if save_last_model:
                model_paths += [(trainer.checkpoint_callback.last_model_path, "last")]
            for file_path, prefix in model_paths:
                if os.path.exists(file_path): # Nathan-added
                    test_model_ckpt(file_path, model_class, test_loaders_list, trainer, prefix) #, val_loaders=val_loaders)


def get_pretrained_filename(ewid):
    # # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(
    #     'checkpoints/', model_class.__name__ + ".ckpt")
    # return pretrained_filename
    # check if ewid is an existing filename
    if ewid is None:
        return None
    if os.path.isfile(ewid):
        return ewid
    else:
        return get_ckptpath_for_wid(ewid)

def fix_files_to_save(files_to_save, logger):
    if files_to_save is not None:
        log_dir = logger.experiment.dir  # Nathan
        os.makedirs(log_dir, exist_ok=True)
        for file in files_to_save:
            if os.path.isfile(file):
                filename = file.split('/')[-1]
                copyfile(file, os.path.join(log_dir, filename))
                print(f'=> Copied {filename}')
            else:
                print(f'=> File not found: {file}')


def set_logger(kwargs, logger_name, postname, prename, trainer_args, wandb_id, wandb_resume_id, wandb_tags,ewid):
    if not (logger_name is None or logger_name == ''):
        logger_name = logger_name.split('/')
        if kwargs['nowandb']:  # For when debugging
            os.environ['WANDB_MODE'] = 'disabled'
        assert not (wandb_id and wandb_resume_id), "Can't specify both wandb_id and wandb_resume_id"
        assert not (ewid and wandb_resume_id), "Can't specify both ewid and wandb_resume_id"
        if wandb_resume_id:
            wandb_id = wandb_resume_id
        logger = pl.loggers.WandbLogger(project='p2_citris',
                                        entity='liir-kuleuven',
                                        name=prename + logger_name[0] + (f'_{ewid}' if ewid is not None else '') + postname,
                                        version=logger_name[1] if len(logger_name) > 1 else None,
                                        id=wandb_id,
                                        save_dir=CITRIS_CKPT_ROOT,
                                        # Add notes
                                        notes=kwargs['notes'],
                                        tags=wandb_tags, )
        # logger.experiment.config['cli_args'] = " ".join(sys.argv[1:])
        # update with allow_vaL_change=True
        logger.experiment.config.update({'cli_args': " ".join(sys.argv[1:])}, allow_val_change=True)
        trainer_args['logger'] = logger
    return logger


def test_model_ckpt(file_path, model_class, test_loaders, trainer=None, prefix=""): # val_loaders=None):
    for c in trainer.callbacks:
        if hasattr(c, 'set_test_prefix'):
            c.set_test_prefix(prefix)
        if hasattr(c, 'set_test_module_ckpt_path'):
            c.set_test_module_ckpt_path(file_path)
    # Automatically loads the model with the saved hyperparameters
    model = model_class.load_from_checkpoint(file_path)
    # val_results = trainer.validate(model, dataloaders=val_loaders, verbose=False)  # list of dicts, TODO remove
    test_results = trainer.test(model, dataloaders=test_loaders, verbose=False) # list of dicts
    # test_result = test_results[0]
    full_test_result = {k: v for d in test_results for k, v in d.items()}
    # ext = 'grouped_latents_' if model_class.__name__ == 'MixedFactorPredictor' else ''
    ext = ''
    if model_class != AutoregNormalizingFlow:
        log_cma_test_results(ext, full_test_result, prefix, trainer)
    trainer.logger.log_table(key=f"test_table_full{f'_{prefix}' if prefix != '' else ''}", dataframe=pd.DataFrame(full_test_result,index=[0]))
    trainer.logger.log_table(key=f"test_table_normed{f'_{prefix}' if prefix != '' else ''}", dataframe=pd.DataFrame({k:v for k,v in full_test_result.items() if (('norm' in k) or ('corr_callback' in k))},index=[0]))
    print('=' * 50,
          f'Test results ({prefix}):',
          '-' * 50,
          '\n'.join([f'{key}: {full_test_result[key]}' for key in full_test_result]),
          '=' * 50, sep='\n')
    # log_file = os.path.join(trainer.logger.log_dir, f'test_results_{prefix}.json')
    log_file = os.path.join(trainer.logger.experiment.dir, f'test_results_{prefix}.json')  # Nathan
    with open(log_file, 'w') as f:
        json.dump(full_test_result, f, indent=4)


def log_cma_test_results(ext, full_test_result, prefix, trainer):
    cma_summary_metrics = ['cma_test_mse_et1_mean', 'test_mse_et1_mean'] + \
                          ['cma_test_z_mseloss', 'test_z_mseloss'] + \
                          [f'corr_callback_{m}_matrix_{d}_{ext}test' for m in ['r2', 'spearman'] for d in
                           ['diag', 'max_off_diag']]
    cma_summary_results = {k: full_test_result[k] for k in cma_summary_metrics if
                           k in full_test_result}  # preserve order
    trainer.logger.log_table(key=f"test_table_cma{f'_{prefix}' if prefix != '' else ''}",
                             dataframe=pd.DataFrame(cma_summary_results, index=[0]))

    cma_detail_metrics = [el for pair in zip([f'cma_test_mse_et1_{i}' for i in range(0, 10)],
                                             [f'test_mse_et1_{i}' for i in range(0, 10)]) for el in pair]
    cma_detail_results = {k: full_test_result[k] for k in cma_detail_metrics if k in full_test_result}  # preserve order
    if len(cma_detail_results) > 0:
        cma_details_df = pd.DataFrame([{k.split('test_')[-1]: v for k, v in cma_detail_results.items() if 'cma' in k},
                                          {k.split('test_')[-1]: v for k, v in cma_detail_results.items() if 'cma' not in k}],
                                            index=['cma', 'no_cma']) # the .split('test_')[-1] is to remove the cma_test_ or test_ prefix, but it's a bit hacky
        # log heatmap of cma_details_df
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.heatmap(cma_details_df, annot=True, fmt='.3f', mask=cma_details_df.isnull(),
                    cmap='Reds', annot_kws={'size': 8})
        fig = plt.gcf()
        n_cols = len(cma_details_df.columns)
        n_rows = len(cma_details_df.index)
        fig.set_size_inches((n_cols * .8 + 2, n_rows * .2 + 2))
        trainer.logger.experiment.log({'cma_heatmap': wandb.Image(plt)})


def maybe_minify_args(args):
    if args.mini:
        args.num_workers = 0
        args.check_val_every_n_epoch = 1
        args.max_epochs = 1 if args.max_epochs >= 0 else -args.max_epochs # Aka, if I do mini but want to override automatically setting max_epochs to 1, I set it to -(num_epochs)
        args.log_every_n_steps = 1
        args.img_callback_freq = 1
        args.batch_size = 1 if args.batch_size >= 0 else -args.batch_size # Aka, if I do mini but want to override automatically setting batch_size to 1, I set it to -(batch_size)
        args.num_ood_epochs = 1 if args.num_ood_epochs >= 0 else -args.num_ood_epochs # Aka, if I do mini but want to override automatically setting num_ood_epochs to 1, I set it to -(num_ood_epochs)
        import warnings
        warnings.filterwarnings("ignore", ".*does not have many workers.*")


def coral_loss(id_values, ood_values):
    id_cov = torch.cov(id_values.T)
    ood_cov = torch.cov(ood_values.T)
    # d = id_cov.shape[0]
    # l_coral = (1 / (4 * (d ** 2))) * torch.linalg.norm(id_cov - ood_cov, ord='fro') ** 2
    l_coral = (1 / (4 )) * torch.linalg.norm(id_cov - ood_cov, ord='fro') ** 2 # I think the division by d^2 is already included in torch.linalg.norm
    return l_coral


def maybe_id_and_ood_unpack(batch, coral):
    if isinstance(batch, dict) and set(batch.keys()) == {'id_batch', 'ood_batch'}:
        coral = True
        # for k, b in batch.items():
        #     ctt1 = b['lat']
        #     assert (ctt1.shape[0] == 1) and (len(ctt1.shape) == len(['1', 'B', 'seq_len', 'C']))
        # #     batch[k] = {k2:t.squeeze(0) for k2,t in b.items()}
        # id_batch_size = batch['id_batch']['lat'].shape[0]
        # ood_batch_size = batch['ood_batch']['lat'].shape[0]
        # # concat items
        # batch = {k:torch.cat([b[k] for b in batch.values()], dim=0) for k in batch['id_batch'].keys()}
        updated_batch = batch['id_batch']
        updated_batch.update({f'ood_{k}': v for k, v in batch['ood_batch'].items()})
        ctt1 = updated_batch['lat']
        assert (ctt1.shape[0] == 1) and (len(ctt1.shape) == len(['1', 'B', 'seq_len', 'C']))
        for k, b in updated_batch.items():
            updated_batch[k] = b.squeeze(0)
        id_batch_size = updated_batch['lat'].shape[0]
        ood_batch_size = updated_batch['ood_lat'].shape[0]
    else:
        # id_batch_size = batch[-1].shape[0] # -1 because if require_imgs == False, then batch[0] is an empty list
        id_batch_size = batch['targets'].shape[0]
        ood_batch_size = 0
        updated_batch = batch
    return updated_batch, coral, id_batch_size, ood_batch_size
    # ctt1 = batch[-1]
    # if (ctt1.shape[0] == 2) and len(ctt1.shape) == len(['2', 'B', 'seq_len', 'C']):
    #     # Indicates that a combo of ID and OOD data was passed in.
    #     coral = True
    #     id_batch_size = len(ctt1[0]) # Bi
    #     ood_batch_size = len(ctt1[1]) # Bo
    #     batch = [torch.cat([el[0], el[1]], dim=0) for el in batch]

    #     # In coral setting, __get_item__ directly returns a batch, so the added dimension needs to be removed
    #     ctt1 = batch[-1]
    #     assert (ctt1.shape[0] == 1) and (len(ctt1.shape) == len(['1', 'B', 'seq_len', 'C']))
    #     batch = [x.squeeze(0) for x in batch]


def update_aeckpt_arg(args):
    # if args.flow_wid is not None:
    if 'flow_wid' in args and args.flow_wid is not None:
        assert args.autoencoder_checkpoint == "", "Can't specify both flow_wid and autoencoder_checkpoint"
        args.autoencoder_checkpoint = AutoregNormalizingFlow.load_from_checkpoint(get_ckptpath_for_wid(args.flow_wid)).hparams.autoencoder_checkpoint
        assert args.autoencoder_checkpoint != "", "No autoencoder checkpoint found for given flow_wid"
        print(f"Using autoencoder checkpoint used by loaded flow: {args.autoencoder_checkpoint}")
    else:
        if args.autoencoder_checkpoint == "":
            print("No explicit autoencoder checkpoint given, selecting checkpoint based on seed")
            AE_CKPTS = AE_PONG_CKPTS if 'pong' in args.data_dir else AE_SHAPES_CKPTS
            index = args.seed % len(AE_CKPTS)
            args.autoencoder_checkpoint = AE_CKPTS[index]
        print(f"Using autoencoder checkpoint: {args.autoencoder_checkpoint}")


