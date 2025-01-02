#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import datasets
import stanza
import spacy_stanza
from datasets import load_dataset, load_metric
import torch
import torch_xla.core.xla_model as xm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from custom_trainer import GPT2LMHeadModelCompress, BERTModelCompress, AutoEncoderWithNoise, GPT2VAE, AR_for_cont,\
    Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree, Classifier_Consistency

from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False, pad_mask_id=None):
    if pad_mask_id is None:
        pad_mask_id = pad_token_id
    result = torch.full([len(examples), max_length], pad_token_id).tolist()
    mask_ = torch.full([len(examples), max_length], pad_mask_id).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    experiment: Optional[str] = field(
        default='compress',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    learned_emb: Optional[str] = field(
        default='no',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    padding_mode: Optional[str] = field(
        default='block',
        metadata={"help": "blcok or pad"},
    )
    roc_train: Optional[str] = field(
        default='/juice/scr/xlisali/diffusion_lm/ROCstory',
        metadata={"help": "roc story path"},
    )
    wiki_train: Optional[str] = field(
        default='/u/scr/xlisali/diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
        metadata={"help": "simple wiki path"},
    )
    e2e_train: Optional[str] = field(
        default='/u/scr/xlisali/e2e_data',
        metadata={"help": "simple wiki path"},
    )

    reduced_emb: Optional[int] = field(
        default=8,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    rounding_mode: Optional[str] = field(
        default='gpt2',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    sigma: Optional[float] = field(
        default=1.0,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    n_embd: Optional[int] = field(
        default=16,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    init_emb: Optional[str] = field(
        default="",
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    task: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    synth_config:  Optional[str] = field(
        default='/juice/scr/xlisali/diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k32_trainc20000.yaml', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def get_corpus_rocstory(data_args):
    '''

    :param data_args:  --> this is actually the model_args in the main function.
    :return:
    '''
    import csv, json
    from collections import Counter, defaultdict
    from spacy.lang.en import English
    import numpy as np

    # print(data_args.task, 'DEBUG', '*---'*100)
    # print(model_args.task, 'DEBUG', '*---' * 100)
    if data_args.experiment.startswith('roc') and data_args.task == 'infill':
        print('loading dataset from ROCStory')
        nlp = English()
        tokenizer = nlp.tokenizer
        sentence_lst = []
        with open(f'{data_args.roc_train}/ROCstory_full.csv', 'r') as csvfile:
            roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            for idx, row in enumerate(roc_reader):
                if idx == 0:
                    continue
                sentences = row[2:]
                for ii in [1, 2, 3]:
                    sent = " ".join([sentences[ii-1], sentences[ii+1], sentences[ii]])
                    example = [x.text for x in tokenizer(sent)]
                    sentence_lst.append(example)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('roc') and data_args.task == 'classifier':
        print('loading dataset from ROCStory')
        nlp = English()
        tokenizer = nlp.tokenizer
        sentence_lst = []
        with open(f'{data_args.roc_train}/ROCstory_full.csv', 'r') as csvfile:
            roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            for idx, row in enumerate(roc_reader):
                if idx == 0:
                    continue
                sentences = row[2:]
                sentences = [[x.text for x in tokenizer(sent)] for sent in sentences]
                for ii in [1, 2, 3]:
                    # sent = " ".join([sentences[ii-1], sentences[ii+1], sentences[ii]])
                    example = [sentences[ii-1], sentences[ii+1], sentences[ii], 1]
                    sentence_lst.append(example)
        np.random.shuffle(sentence_lst)

        # construct negative examples/
        wrong_lst = []
        for idx, sent in enumerate(sentence_lst[:-1]):
            wrong_mid = sentence_lst[idx+1][2]
            wrong_tup = (sent[0], sent[1], wrong_mid, 0)
            wrong_lst.append(wrong_tup)

        sentence_lst = sentence_lst + wrong_lst

        print(sentence_lst[:2], sentence_lst[-2:])
        return sentence_lst, {}




    elif data_args.experiment.startswith('roc') and data_args.task != 'data_teacher':
        print('loading dataset from ROCStory')
        nlp = English()
        tokenizer = nlp.tokenizer
        sentence_lst = []
        with open(f'{data_args.roc_train}/roc_train.json', 'r') as roc_reader:
            for row in roc_reader:
                sentences = json.loads(row)[0].strip()
        # with open(data_args.roc_train, 'r') as csvfile:
        #     roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
        #     for row in roc_reader:
        #         sentences = " ".join(row[2:])
                word_lst = [x.text for x in tokenizer(sentences)]
                sentence_lst.append(word_lst)
        # sentence_lst = sentence_lst[1:]
        print(sentence_lst[:2])
    elif data_args.experiment.startswith('roc') and data_args.task == 'data_teacher':
        print('loading dataset from ROCStory')
        sentence_lst = []
        with open(data_args.roc_train, 'r') as csvfile:
            roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            for row in roc_reader:
                sentences = " ".join(row[2:])
                sentence_lst.append(sentences)
        sentence_lst = sentence_lst[1:]
        print(sentence_lst[:2])
        return sentence_lst, None
    elif data_args.experiment.startswith('simple-wiki'):
        print('loading dataset from simple wikipedia')
        sentence_lst = []
        with open(data_args.wiki_train, 'r') as ff:
            for row in ff:
                word_lst = row.split()
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])
    elif data_args.experiment.startswith('e2e-tgt') and data_args.task == 'data_teacher':
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('e2e-tgt') and data_args.task == 'finetuneUNK':
        '''
            Used to evaluate fluency: first load e2e-vocab, and then UNK the oov words in the training data. 
        '''
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        # load vocab.
        tokenizer2 = load_tokenizer('e2e-tgt', 'random',
                                   '/u/scr/nlp/xlisali/predictability/diffusion_models_v6/diff_e2e-tgt_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart')
        vocab = {v: k for k, v in tokenizer2.items()}
        print(len(tokenizer2), len(vocab), 'loaded vocabs')

        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                tokenized = [x.text for x in tokenizer(word_lst)]
                word_lst1 = [x if x in vocab else 'UNK' for x in tokenized]
                word_lst1 = " ".join(word_lst1)
                word_lst2 = [vocab.get(x.text, vocab['UNK']) for x in tokenizer(word_lst)]
                word_lst2 = " ".join([tokenizer2[x] for x in word_lst2])
                # print(word_lst1, word_lst2)
                assert word_lst1 == word_lst2

                # print(word_lst1)
                sentence_lst.append(word_lst1)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('e2e-tgt') and data_args.task == 'right2left':
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                word_lst = list(reversed([x.text for x in tokenizer(word_lst)]))
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('e2e-tgt'):
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                word_lst = [x.text for x in tokenizer(word_lst)]
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])


    elif data_args.experiment.startswith('e2e-back'):
        ordered_ = ['name', 'Type', 'area', 'customer rating', 'near',
                    'family friendly', 'food', 'price']
        full_dict = defaultdict(lambda:Counter())
        def ordered_fill(src_lst, mode='full', full_dict=None):
            pair_lst = {x.split(':')[0].lstrip().strip():x.split(':')[1].lstrip().strip() for x in src_lst.split('|')}
            # print(pair_lst, 'hello')
            if mode == 'full':
                for x in ordered_:
                    v = pair_lst.get(x, 'none')
                    result_lst.append(f"{x} : {v}")
                return "|".join(result_lst)
            else:
                # print(pair_lst)
                v = pair_lst.get(mode, 'none')
                full_dict[mode][v] += 1
                # print(v)
                return f"{mode} : {v}"

        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        vocab_lst = []
        with open(path, 'r') as ff:
            for row in ff:
                src_lst, word_lst = row.split('||')
                # src_lst = ordered_fill(src_lst, 'food')
                # src_lst = ordered_fill(src_lst, 'price')

                word_lst = [x.text for x in tokenizer(word_lst)]
                for mode in ordered_:
                    src_lst3 = ordered_fill(src_lst, mode, full_dict)
                    src_lst2 = [x.text for x in tokenizer(src_lst3)]
                    sentence_lst.append((word_lst, src_lst2))
                vocab_lst.append(word_lst)

                # src_lst = ordered_fill(src_lst, 'area')
                # word_lst = [x.text for x in tokenizer(word_lst)]
                # src_lst = [x.text for x in tokenizer(src_lst)]
                # sentence_lst.append((word_lst, src_lst))
        print(sentence_lst[:2])
        print(full_dict)

        counter = Counter()
        for input_ids in vocab_lst:
            counter.update(input_ids)
            # counter.update(src_ids)

    # get tokenizer.
    if not data_args.experiment.startswith('e2e-back'):
        counter = Counter()
        for input_ids in sentence_lst:
            counter.update(input_ids)

    vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
    for k, v in counter.items():
        if v > 10:
            vocab_dict[k] = len(vocab_dict)
    print(len(counter), len(vocab_dict))

    return sentence_lst, vocab_dict


def main():
    import torch
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if not torch.to(xm.xla_device()).is_available():
        training_args.device = torch.device("cpu")
        logger.warning("TPU not available, using CPU instead.")
    else:
        try:
            xm.set_rng_state(training_args.seed)
            logger.info(f"TPU available, local ordinal: {xm.get_local_ordinal()}")
        except Exception as e:
            training_args.device = torch.device("cpu")
            logger.warning(f"TPU initialization failed: {e}, using CPU instead.")
    
    training_args.device = torch.device("cpu")
    logger.warning("Forcing CPU usage to avoid TPU initialization errors.")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if model_args.experiment.startswith('synth'):
        import yaml, torch
        sys.path.insert(0, '/juice/scr/xlisali/diffusion_lm/synthetic_data/rnns-stacks')
        from dataset import Dataset as SynthDataset
        args_synth = yaml.load(open(data_args.synth_config))
        # device = torch.device("cuda:0" if torch.to(xm.xla_device()).is_available() else "cpu")
        # args_synth['device'] = device
        print(args_synth)
        dataset_synth = SynthDataset(args_synth)
        print(dataset_synth.train_dataset[:5])
        from datasets import Dataset
        train_datasets = Dataset.from_dict({'text': dataset_synth.train_dataset})
        raw_datasets = datasets.DatasetDict()
        raw_datasets['train'] = train_datasets
        raw_datasets['validation'] = Dataset.from_dict({'text': dataset_synth.test_dataset})
        raw_datasets.vocab = dataset_synth.vocab
    elif model_args.experiment.startswith('pos'):
        import yaml, torch, json
        from collections import Counter, defaultdict
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, )
        dataset = dataset.load_from_disk('/u/scr/nlp/xlisali/wikitext-2-pos')
        counter = Counter()
        print(dataset)
        for input_ids in dataset['train']:
            counter.update(input_ids['pos'])
        print(counter)
        print(dataset)
        vocab_dict = {'START': 0, 'END': 1}
        for k in counter.keys():
            vocab_dict[k] = len(vocab_dict)

        dataset.vocab = vocab_dict
        from datasets import Dataset
        raw_datasets = dataset

    ###################### LOAD DATASETS & dictionary #########################
    elif model_args.experiment.startswith('roc') or\
            model_args.experiment.startswith('simple-wiki') or \
            model_args.experiment.startswith('e2e-tgt') or \
            model_args.experiment.startswith('e2e-back'):
        train_dataset, vocab = get_corpus_rocstory(model_args) # TODO: include validation sets.
        print(len(vocab), 'derived vocabs')

        if model_args.experiment.startswith('roc'):
            tokenizer = load_tokenizer('roc', 'random',
                                       '/u/scr/nlp/xlisali/predictability/diffusion_models_v7/diff_roc_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart')
            vocab = {v: k for k, v in tokenizer.items()}
            print(len(tokenizer), len(vocab), 'loaded vocabs')

        # train_dataset = train_dataset[:100]
        from datasets import Dataset

        if model_args.task == 'classifier':
            print(len(train_dataset))
            train_dataset = list(zip(*train_dataset))
            print(len(train_dataset))
            train_datasets = Dataset.from_dict({'left_text': train_dataset[0],
                                                'right_text':train_dataset[1],
                                                'mid_text':train_dataset[2],
                                                'label':train_dataset[3]})
        else:
            train_datasets = Dataset.from_dict({'text': train_dataset})
        raw_datasets = train_datasets.train_test_split(0.01)
        print(raw_datasets)

        if model_args.experiment in ['e2e-tgt-pos', 'e2e-tgt-gen-pos']:
            pos_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}
            pos_lst = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB',
                       'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ',
                       'PUNCT', 'SYM', 'X']
            for x in pos_lst:
                pos_vocab[x] = len(pos_vocab)
        elif model_args.experiment in ['e2e-tgt-tree', 'e2e-tgt-gen-tree', 'e2e-tgt-gen-spans']:
            import benepar
            parser = benepar.Parser("benepar_en3")
            tree_vocab = parser._parser.config["label_vocab"]

        raw_datasets.vocab = vocab
        raw_datasets['validation'] = raw_datasets['test']

    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset
