""" Train a diffusion model on images. """
import argparse
import json
import torch
import os
import numpy as np
import torch_xla.core.xla_model as xm
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from transformers import AutoTokenizer
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp
from improved_diffusion.rounding import load_models, load_tokenizer
import torch.distributed as dist
import wandb


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)
    
    dist_util.setup_dist()  # Set up distributed training

    logger.configure()
    logger.log("creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.to(xm.xla_device())  # Move model to TPU device

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    logger.log(f'saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "diffusion_lm"),
        name=args.checkpoint_path,
    )
    wandb.config.update(args.__dict__, allow_val_change=True)

    if args.experiment_mode == 'conditional_gen':
        assert args.modality in ['e2e']
        assert args.padding_mode == 'pad'
    
    logger.log("creating data loader...")
    
    if args.modality == 'image':
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )
        data_valid = None
    else:
        print('load data', '*' * 50)
        
        # Load data based on modality without referencing non-existent models.
        if args.modality in ['roc-aug', 'commonGen-aug']:
            tokenizer = load_tokenizer(args.modality, args.experiment, 
                'path/to/your/tokenizer')  # Update this line with a valid tokenizer path or remove it if unnecessary.
            rev_tokenizer = {v: k for k, v in tokenizer.items()}
            print(len(rev_tokenizer), 'loading from tokenizer.')
        elif args.use_bert_tokenizer == 'yes':
            rev_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            rev_tokenizer = None

        # Ensure in_channel is set correctly before assertion
        print(f"Value of in_channel: {args.in_channel}")  # Debugging line
        

        
        data = load_data_text(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args=args,
            task_mode=args.modality,
            padding_mode=args.padding_mode,
            load_vocab=rev_tokenizer,
            model=None,  # Set to None or remove if not needed.
        )

    next(data)  # Prepare the iterator

    # Load models based on modality and experiment settings; ensure paths are valid.
    model2, tokenizer = load_models(args.modality, args.experiment, 
                                    args.model_name_or_path, args.in_channel, 
                                    args.checkpoint_path, extra_args=args)

    if args.modality in ['book', 'use_bert_tokenizer']:
        rev_tokenizer = tokenizer  # BERT tokenizer BPE.
    else:
        rev_tokenizer = {v: k for k, v in tokenizer.items()}

    data_valid = load_data_text(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        data_args=args,
        task_mode=args.modality,
        padding_mode=args.padding_mode,
        split='valid',
        load_vocab=rev_tokenizer,
        model=model2,
    )

def create_argparser():
    defaults = dict(
         data_dir="",
         schedule_sampler="uniform",
         lr=1e-4,
         weight_decay=0.0,
         lr_anneal_steps=0,
         batch_size=1,
         microbatch=-1,  # -1 disables microbatches
         ema_rate="0.9999",
         log_interval=50,
         save_interval=50000,
         resume_checkpoint="",
         use_fp16=False,
         fp16_scale_growth=1e-3,
         seed=101,
         gradient_clipping=-1.0,
         eval_interval=2000,
         checkpoint_path='diff_models',
         in_channel=128  # Set default value for in_channel based on your command input.
     )

    text_defaults = dict(
         modality='text',
         dataset_name='wikitext',
         dataset_config_name='wikitext-2-raw-v1',
         config='diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml',
         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
         experiment='gpt2_pre_compress',
         model_arch='conv-unet',
         roc_train='diffusion_lm/ROCstory',
         wiki_train='diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
         e2e_train='e2e_data',
         yelp_train='diffusion_lm/yelpnlg-resources/yelpnlg-corpus',
         commonGen_train='diffusion_lm/common-gen/commongen_data',
         emb_scale_factor=1.0,
         noise_level=0.0,
         cache_mode='no',
         use_bert_tokenizer='no',
         padding_mode='block',
         preprocessing_num_workers=1
     )
     
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
     
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
