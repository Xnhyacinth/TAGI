#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
from pathlib import Path
import random
import string
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets.utils import set_progress_bar_enabled
from datasets import load_dataset, load_metric

import torch
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from model import T5LoraWrapper, LoRAT5
from ni_collator import DataCollatorForNI
from ni_trainer import NIKDTrainer, NITrainer, DenserEvalCallback
from compute_metrics import compute_metrics, compute_grouped_metrics


set_progress_bar_enabled(False)
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    t_model: str = field(
        default=None,
        metadata={
            "help": "Path to teacher model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    r: Optional[int] = field(
        default=32,
        metadata={
            "help": "The lora rank of the model. If the model is not a lora model, this argument will be ignored."
        },
    )
    encoding_dim: Optional[int] = field(
        default=255,
        metadata={
            "help": "The lora rank of the model. If the model is not a lora model, this argument will be ignored."
        },
    )
    prefix_length: Optional[int] = field(
        default=0,
        metadata={
            "help": "The length of gen prefix."
        },
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The temperature."
        },
    )
    load_hypernet_weights: str = field(
        default=None,
        metadata={"help": "Path to hypernet weights, otherwise random init."},
    )
    name: str = field(
        default=None,
        metadata={"help": "Path to hypernet weights, otherwise random init."},
    )
    alpha_kd: Optional[float] = field(
        default=0.4,
        metadata={"help": "weights of KD loss."}
    )
    use_kl: bool = field(
        default=False,
        metadata={
            "help": "Whether to use kl loss."
        },
    )
    use_ce: bool = field(
        default=False,
        metadata={
            "help": "Whether to use ce loss."
        },
    )
    use_hd: bool = field(
        default=False,
        metadata={
            "help": "Whether to use hidden states loss."
        },
    )
    use_attn: bool = field(
        default=False,
        metadata={
            "help": "Whether to use attention loss."
        },
    )
    select: bool = field(
        default=False,
        metadata={
            "help": "Whether to select layers for hd & attn loss."
        },
    )
    prompt: bool = field(
        default=False,
        metadata={
            "help": "Whether to use full prompt."
        },
    )
    kd: bool = field(
        default=False,
        metadata={
            "help": "Whether to knowledge distillation."
        },
    )
    whitening: bool = field(
        default=False,
        metadata={
            "help": "Whether to use whitening algorithm."
        },
    )
    custom_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to use concat for input."
        },
    )
    do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether to do_sample."
        },
    )
    hyperencoder: bool = field(
        default=False,
        metadata={
            "help": "Whether to do_sample."
        },
    )
    loramse: bool = field(
        default=False,
        metadata={
            "help": "Whether to use loramse."
        },
    )
    logit_stand: bool = field(
        default=False,
        metadata={
            "help": "Whether to use logit_stand."
        },
    )
    pooling: Optional[str] = field(
        default="first_last_avg", metadata={"help": "Method for getting the instructions' features."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={
                      "help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions train/dev/test splits."}
    )
    task_dir: str = field(
        default=None, metadata={"help": "The directory for saving the NaturalInstructions tasks json files."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_num_instances_per_task: int = field(
        default=None, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=500, metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_task_definition: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to preappend task definition before the task input."}
    )
    num_pos_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    s_num_pos_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    num_neg_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context negative examples."}
    )
    add_explanation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to add explanation for both the postive examples and negtive examples."}
    )
    tk_instruct: Optional[bool] = field(
        default=False,
        metadata={"help": "tk_instruct will train a model combining all valid instruction encodings. This will overwrite the other settings about instruction encoding."}
    )
    data_type: Optional[str] = field(
        default=None, metadata={"help": "The task type of model."}
    )

    def __post_init__(self):
        pass


@dataclass
class NITrainingArguments(Seq2SeqTrainingArguments):
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={
                          "help": "Whether to run the model as a demo in the terminal."})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, NITrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    checkpointpath = Path(training_args.output_dir)
    checkpointpath.mkdir(parents=True, exist_ok=True)
    with open(checkpointpath / 'options.txt', 'w') as o:
        for k, v in sorted(training_args.__dict__.items(), key=lambda x: x[0]):
            o.write(f'{k} = {v}\n')
        for k, v in sorted(model_args.__dict__.items(), key=lambda x: x[0]):
            o.write(f'{k} = {v}\n')
        for k, v in sorted(data_args.__dict__.items(), key=lambda x: x[0]):
            o.write(f'{k} = {v}\n')

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(training_args.output_dir, "training.log"),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

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

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # Get the NaturalInstructions dataset
    raw_datasets = load_dataset(
        "src/ni_dataset.py",
        data_dir=data_args.data_dir,
        task_dir=data_args.task_dir,
        cache_dir=model_args.cache_dir,
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        data_type=data_args.data_type
    )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.custom_model:
        from modeling_t5 import T5ForConditionalGeneration
        from configuration_t5 import T5Config
        model_cls = T5ForConditionalGeneration
        config_cls = T5Config
    else:
        model_cls = AutoModelForSeq2SeqLM
        config_cls = AutoConfig
        # from modeling_t5 import T5ForConditionalGeneration
        # from configuration_t5 import T5Config
        # model_cls = T5ForConditionalGeneration
        # config_cls = T5Config
    config = config_cls.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        # cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        # cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if training_args.do_predict and not training_args.do_train:
        model_cls = LoRAT5
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            device_map='auto',
            torch_dtype=torch.bfloat16,
            config=config,
            # cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model_args.name = model.config.name
    else:
        model_cls = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            # cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel()
                            for p in model.parameters() if p.requires_grad)
        return trainable_num, total_num

    if model_args.kd:
        t_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.t_model,
            from_tf=bool(".ckpt" in model_args.t_model),
            torch_dtype=torch.bfloat16,
            # cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        ).cuda()
        for layer in t_model.modules():
            for _, param in layer.named_parameters():
                param.requires_grad = False
        if training_args.do_train:
            model_args.d_model = t_model.config.d_model
            hypernet_config = {
                "pooler_d_model": t_model.config.d_model,
                "embedding_dim": model_args.r,
                "encoding_dim": model_args.encoding_dim,
                "custom_model": model_args.custom_model,
                "name": model_args.name,
                "whitening": model_args.whitening,
                "hyperencoder": model_args.hyperencoder,
            }
            if "prefix" in model_args.name:
                hypernet_config["prefix_length"] = model_args.prefix_length
                hypernet_config["max_source_length"] = data_args.max_source_length
            model_cls.config.update(hypernet_config)
            model = LoRAT5(model_cls.config)
            model.load_t5(model_cls.state_dict())
            trainable_params, all_param = get_parameter_number(model)
            logger.info(
                f"trainable params: {trainable_params / 2 ** 20:.2f}M || all params: {all_param / 2 ** 20:.2f}M || trainable%: {100 * trainable_params / all_param:.2f}%")
    else:
        model = model_cls
    if "t5-xxl" not in model_args.model_name_or_path:
        model.resize_token_embeddings(len(tokenizer))

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    def compute_kernel_bias(vecs, n_components=256):
        """compute kernel and bias
        vecs.shape = [num_samples, embedding_size]
        transfer:y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :n_components], -mu

    def transform_and_normalize(vecs, kernel=None, bias=None):
        """ normalization
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

    def preprocess_function(sample):
        sources, prefixs, instances, s_sources = [], [], [], []
        for instance, task, defi, pos, neg in zip(sample['Instance'], sample['Task'], sample['Definition'], sample['Positive Examples'], sample['Negative Examples']):
            add_task_name = data_args.add_task_name
            add_task_definition = data_args.add_task_definition
            num_pos_examples = data_args.num_pos_examples
            num_neg_examples = data_args.num_neg_examples
            add_explanation = data_args.add_explanation
            s_num_pos_examples = data_args.s_num_pos_examples

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "

            task_name = ""
            if add_task_name:
                task_name += task + ". "

            definition = ""
            if add_task_definition:
                if isinstance(defi, list):
                    # TODO: should we use <Definition>?
                    definition = "Definition: " + defi[0].strip()
                else:
                    definition = "Definition: " + defi.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"

            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(pos[:num_pos_examples]):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if len(tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= data_args.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    break

            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(neg[:num_neg_examples]):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if len(tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= data_args.max_source_length:
                    neg_examples.append(neg_example_str)
                else:
                    break

            source = task_name + definition + \
                "".join(pos_examples) + "".join(neg_examples) + task_input
            s_source = task_name + definition + \
                "".join(pos_examples[:s_num_pos_examples]) + task_input
            tokenized_source = tokenizer(source)["input_ids"]
            if len(tokenized_source) <= data_args.max_source_length:
                sources.append(source)
            else:
                sources.append(tokenizer.decode(
                    tokenized_source[:data_args.max_source_length], skip_special_tokens=True))
            tokenized_s_source = tokenizer(s_source)["input_ids"]
            if len(tokenized_s_source) <= data_args.max_source_length:
                s_sources.append(s_source)
            else:
                s_sources.append(tokenizer.decode(
                    tokenized_s_source[:data_args.max_source_length], skip_special_tokens=True))

            # prefix
            prefix = task_name + definition + \
                "".join(pos_examples[:s_num_pos_examples]
                        ) + "".join(neg_examples)
            prefixs.append(prefix)
            if task not in prefixs_tasks.keys():
                prefixs_tasks[task] = prefix

            # instance
            instances.append(task_input)
        sample['source'] = sources
        sample['s_source'] = s_sources
        sample['instance'] = instances
        sample['prefix'] = prefixs
        return sample

    def process_prefixs(prefixs_tasks):
        prefixs = list(prefixs_tasks.values())
        print(len(prefixs))
        padding = "max_length" if data_args.pad_to_max_length else "longest"
        t_model.eval()
        with torch.no_grad():
            pooled_sentence_list, instruction_input_list, attention_mask_list = [], [], []
            g = 16
            for i in range(0, len(prefixs), g):
                last = i + g if i + g < len(prefixs) else len(prefixs)
                prefix_inputs = tokenizer(
                    prefixs[i: last],
                    max_length=data_args.max_source_length,
                    padding=padding,
                    return_tensors="pt",
                    truncation=True,
                    pad_to_multiple_of=8 if training_args.fp16 else None
                )
                attention_mask_list.append(prefix_inputs["attention_mask"])
                prefix_inputs = prefix_inputs.to(model.device)
                # hidden_states = t_model.encoder(**prefix_inputs, return_dict=True, output_hidden_states=True).hidden_states
                hidden_states = model.encoder(
                    **prefix_inputs, return_dict=True, output_hidden_states=True).hidden_states

                if pooling == 'first_last_avg':
                    pooled_sentence = (hidden_states[-1] + hidden_states[1])
                elif pooling == 'last_avg':
                    pooled_sentence = (hidden_states[-1])
                elif pooling == 'last2avg':
                    pooled_sentence = (hidden_states[-1] + hidden_states[-2])
                else:
                    raise Exception("unknown pooling {}".format(pooling))

                instruction_input_list.append(pooled_sentence)
                pooled_sentence = pooled_sentence.mean(dim=1)
                pooled_sentence_list.append(pooled_sentence.float())

            pooled_sentence = torch.cat(pooled_sentence_list, 0).cpu().numpy()
            if model_args.whitening:
                kernel, bias = compute_kernel_bias(pooled_sentence, 255)
                pooled_sentence = transform_and_normalize(
                    pooled_sentence, kernel=kernel, bias=bias)
            if model_args.custom_model:
                instruction_input = torch.cat(instruction_input_list, 0)
                attention_mask = torch.cat(attention_mask_list, 0)
                return dict(zip(list(prefixs_tasks.keys()), pooled_sentence.tolist())), dict(zip(list(prefixs_tasks.keys()), instruction_input.tolist())), dict(zip(list(prefixs_tasks.keys()), attention_mask.tolist()))
            else:
                return dict(zip(list(prefixs_tasks.keys()), pooled_sentence.tolist())), None, None

    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # raw_datasets['train'] = raw_datasets['train'].select(range(200))
    # raw_datasets['test'] = raw_datasets['test'].select(range(10))
    if model_args.whitening:
        pooling = model_args.pooling
        prefixs_tasks = {}

        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            batch_size=2048,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        task_features, instruction_inputs, attention_masks = process_prefixs(
            prefixs_tasks)

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(
                data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(
                range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(
                range(data_args.max_predict_samples))

    # Data collator
    model_args.s_num_pos_examples = data_args.s_num_pos_examples
    if model_args.kd:
        data_map, lora_dict = None, None
        if model_args.loramse:
            with open('src/data_dict.json', 'r') as f:
                data_dict = json.load(f)
                data_map = data_dict['data_map']
            lora_dict = {}
            output_lora_path = 'output_meta'
            if data_args.s_num_pos_examples == 0:
                output_lora_path = 'output_meta_pos0'
            for file in os.listdir(output_lora_path):
                try:
                    with open(f'{output_lora_path}/{file}/param_tensors.json', 'r') as f:
                        lora_d = json.load(f)
                        if 'ko' not in model_args.name:
                            lora_d.pop('param_tensor_A')
                            lora_d.pop('param_tensor_B')
                        else:
                            lora_d.pop('param_tensor_qv_A')
                            lora_d.pop('param_tensor_qv_B')
                        lora_dict[data_map[file]] = lora_d
                except:
                    pass
        data_collator = DataCollatorForNI(
            tokenizer,
            model=t_model,
            padding="max_length" if data_args.pad_to_max_length else "longest",
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            add_task_name=data_args.add_task_name,
            add_task_definition=data_args.add_task_definition,
            num_pos_examples=data_args.num_pos_examples,
            num_neg_examples=data_args.num_neg_examples,
            add_explanation=data_args.add_explanation,
            tk_instruct=data_args.tk_instruct,
            kd=model_args.kd,
            task_features=task_features if model_args.whitening else None,
            instruction_inputs=instruction_inputs if model_args.whitening else None,
            attention_masks=attention_masks if model_args.whitening else None,
            args=model_args,
            student_input=data_args.s_num_pos_examples != data_args.num_pos_examples if training_args.do_train else False,
            lora_dict=lora_dict,
            data_map=data_map
        )
    else:
        data_collator = DataCollatorForNI(
            tokenizer,
            model=model,
            padding="max_length" if data_args.pad_to_max_length else "longest",
            max_source_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            add_task_name=data_args.add_task_name,
            add_task_definition=data_args.add_task_definition,
            num_pos_examples=data_args.num_pos_examples,
            num_neg_examples=data_args.num_neg_examples,
            add_explanation=data_args.add_explanation,
            tk_instruct=data_args.tk_instruct,
            args=model_args
        )
    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will aslo be used in evaluation.
    training_args.remove_unused_columns = False

    # Metric
    def compute_ni_metrics(dataset, preds, save_prefix=None):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = [e["Instance"]["output"] for e in dataset]
        result = compute_metrics(
            predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(
            predictions=decoded_preds, references=references, groups=dataset["Task"])
        result.update(result_per_task)
        categories = ["_".join(it[0].lower().split())
                      for it in dataset["Categories"]]
        result_per_category = compute_grouped_metrics(
            predictions=decoded_preds, references=references, groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(
            pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Definition": example["Definition"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result

    # model = T5LoraWrapper(model, model_args.r, model_args.load_hypernet_weights, model_args)
    # if model_args.load_hypernet_weights is not None:
    #     model.load_state_dict(torch.load(model_args.load_hypernet_weights), strict=False, map_location=torch.device('cpu'))
    # trainable_params, all_param = get_parameter_number(model)
    # logger.info(f"trainable params: {trainable_params / 2 ** 20:.2f}M || all params: {all_param / 2 ** 20:.2f}M || trainable%: {100 * trainable_params / all_param:.2f}%")

    # Initialize our Trainer
    if model_args.kd:
        trainer = NIKDTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
            callbacks=[
                DenserEvalCallback] if training_args.denser_evaluation else None
        )
        trainer.post_init(model_args, t_model)
    else:
        trainer = NITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
    )
    # trainer.post_init(model_args, t_model)
    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        model.config.save_pretrained(training_args.output_dir)

        # save_state = {}
        # for param_tensor in model.state_dict():
        #     if 'hypernet' in param_tensor:
        #         save_state.update({param_tensor:model.state_dict()[param_tensor]})
        # torch.save(save_state, f'{training_args.output_dir}/hypernet_weights.pt')

        # torch.save(trainer.model.hypernet.state_dict(), model_args.save_adapter_path)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        all_metrics.update(metrics)

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        all_metrics.update(metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(
                predict_dataset)
        )
        metrics["predict_samples"] = min(
            max_predict_samples, len(predict_dataset))

        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        all_metrics.update(metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                # output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                # with open(output_prediction_file, "w") as writer:
                #     writer.write("\n".join(predictions))
                output_prediction_file = os.path.join(
                    training_args.output_dir, "predicted_examples.jsonl")
                with open(output_prediction_file, "w") as fout:
                    for example, prediction in zip(predict_dataset, predictions):
                        example["prediction"] = prediction
                        fout.write(json.dumps(example) + "\n")

    if (training_args.do_train or training_args.do_eval or training_args.do_predict) and trainer.is_world_process_zero():
        with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

    if training_args.do_demo:
        logger.info("Serving the model as a demo...")
        user_input = ''
        trainer._max_length = max_length
        trainer._num_beams = num_beams
        while True:
            user_input = input(
                "Please enter your input to the model, or enter 'quit' to exit: ")
            if user_input.lower() == "quit":
                break
            inputs = tokenizer([user_input], return_tensors="pt")
            _, preds, _ = trainer.prediction_step(
                model, inputs=inputs, prediction_loss_only=False)
            print(
                f"Model generates: {tokenizer.decode(preds[0], skip_special_tokens=True)}\n\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
