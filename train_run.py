import sys
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments.')
    
    # Experiment and task parameters
    parser.add_argument('--experiment', type=str, default='no-rep', help='Experiment type.')
    parser.add_argument('--task', type=str, default='wp', help='Task name.')

    # Random index parameter
    parser.add_argument('--rand_idx', type=str, default='no', help='Random index usage.')

    # Model parameters
    parser.add_argument('--pretrained_model', type=str, default='gpt2', help='Pretrained model name.')
    parser.add_argument('--model_type', type=str, default='gpt2', help='Model type.')

    # Dataset parameters
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset name.')
    parser.add_argument('--dataset_config_name', type=str, default='wikitext-103-raw-v1', help='Dataset config name.')
    parser.add_argument('--train_file', type=str, default='wikitext', help='Training file name.')
    parser.add_argument('--validation_file', type=str, default='wikitext', help='Validation file name.')

    # Directory and logging parameters
    parser.add_argument('--dir_name', type=str, default=None, help='Directory name for outputs.')
    parser.add_argument('--notes', type=str, default=None, help='Notes for the experiment.')
    parser.add_argument('--block_size', type=int, default=100, help='Block size for training.')

    # Training parameters
    parser.add_argument('--seed', type=int, default=101, help='Random seed.')
    parser.add_argument('--bsz', type=int, default=10, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=5, help='Number of epochs.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps.')
    parser.add_argument('--learning_rate', type=float, default=5e-05, help='Learning rate.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer.')
    parser.add_argument('--percent', type=float, default=1.0, help='Percent of data to use.')

    # Submission and additional parameters
    parser.add_argument('--submit', type=str, default='no', help='Submit flag.')
    parser.add_argument('--use_big', type=str, default='no', help='Use big model flag.')
    
    # Additional app parameters
    parser.add_argument('--app', type=str, default='', help='Additional app parameters.')

    args = parser.parse_args()

    folder_name = "classifier_models"

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    # Constructing model file name and logging directory based on experiment and dataset
    if args.dataset_name == 'none':
        Model_FILE = f"{args.experiment}_e={args.epoch}_b={args.bsz * args.gradient_accumulation_steps}_" \
                     f"{args.pretrained_model}_{os.path.basename(args.train_file)}_{args.seed}_{args.task}"
        logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
        Model_FILE = os.path.join(folder_name, Model_FILE)
        app = f" --train_file={args.train_file} --validation_file={args.validation_file} --task={args.task}"
        app += " " + args.app
    else:
        Model_FILE = f"{args.experiment}_e={args.epoch}_b={args.bsz * args.gradient_accumulation_steps}_" \
                     f"{args.pretrained_model}_{args.dataset_config_name}_{args.seed}_{args.task}"
        logging_dir = os.path.join(folder_name, 'runs', Model_FILE)
        Model_FILE = os.path.join(folder_name, Model_FILE)
        app = f" --dataset_name={args.dataset_name} --dataset_config_name={args.dataset_config_name} --task={args.task}"
        app += " " + args.app

    # Constructing the command line for run_clm.py
    COMMANDLINE = f"python transformers/examples/pytorch/language-modeling/run_clm.py " \
                  f"--output_dir={Model_FILE} " \
                  f"--model_name_or_path={args.pretrained_model} " \
                  f"--tokenizer_name={args.pretrained_model} " \
                  f"--per_device_train_batch_size={args.bsz} " \
                  f"--per_device_eval_batch_size={args.bsz} " \
                  f"--save_steps=50000 " \
                  f"--num_train_epochs={args.epoch} " \
                  f"--do_train --eval_steps=10000 --eval_strategy=steps " \
                  f"--do_eval --dataloader_num_workers=4 " \
                  f"--save_total_limit=1 " \
                  f"--overwrite_output_dir " \
                  f"--logging_dir={logging_dir} " \
                  f"--block_size={args.block_size} " \
                  f"--disable_tqdm=True --model_type={args.model_type} " \
                  f"--gradient_accumulation_steps={args.gradient_accumulation_steps} " \
                  f"--seed={args.seed}"

    COMMANDLINE += app

    # Save command to a shell script file
    with open(Model_FILE + '.sh', 'w') as f:
        print(COMMANDLINE, file=f)

    print(COMMANDLINE)

    # Execute the command if not submitting
    if args.submit == 'no':
        os.system(COMMANDLINE)
