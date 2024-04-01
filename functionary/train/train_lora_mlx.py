import json
import math
import os
import pathlib
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.distributed
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx.utils import tree_flatten
from mlx.nn.losses import cross_entropy

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import transformers
from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
    LlamaTokenizer,
    LlamaTokenizerFast,
    AutoTokenizer,
    Trainer,
)

from functionary.prompt_template import get_prompt_template_by_version
from functionary.train.custom_datasets import read_dataset, SimpleCollator
from functionary.mlx_compat.peft import LoraConfig, get_peft_model
from functionary.mlx_compat.transformers import MLXTrainer

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def print_rank0(*arg):
    if LOCAL_RANK == 0:
        print(*arg)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    training_ratio: float = field(
        default=1.0, metadata={"help": "percentage of data used for training"}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the eval data."}
    )
    eval_ratio: float = field(
        default=1.0, metadata={"help": "percentage of data used for evluation"}
    )
    packing: bool = field(
        default=False, metadata={"help": "Whether use packing or not"}
    )
    pack_length: int  = field(
        default=None, metadata={"help": "Packing length."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    report_to: str = field(
        default="wandb", metadata={"help": "Report logging to wandb"}
    )

    keep_assistant_prefix: bool = field(
        default=True,
        metadata={
            "help": "Whether to mask the assistant prefix `<|from|>assistant\n<|recipient|>` during training"
        },
    )

    prompt_template_version: str = field(
        default="v2", metadata={"help": "choose prompt template to use for training"}
    )


@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: str = "all"  # all for all linear; "q_proj v_proj"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) or isinstance(module, nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    lora_param_count = 0
    all_param = 0
    embedding_lm_head_param_count = 0
    for name, param in tree_flatten(model.trainable_parameters()):
        num_params = param.size

        # all_param += num_params
        # if param.requires_grad:
        print_rank0(f"trainable: {name}, num_params: {num_params}, dtype: {param.dtype}")
        if "lm_head" in name or "embed_tokens" in name:
            embedding_lm_head_param_count += num_params
        else:
            lora_param_count += num_params

    for name, param in tree_flatten(model.parameters()):
        num_params = param.size

        all_param += num_params

    trainable_params = embedding_lm_head_param_count + lora_param_count
    print_rank0(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    print_rank0(
        f"embedding_lm_head_param_count: {embedding_lm_head_param_count} = {embedding_lm_head_param_count * 100 / all_param} %"
    )
    print_rank0(
        f"loara_param: {lora_param_count} = {lora_param_count * 100 / all_param} %"
    )


def prepare_model_for_training(
    model: transformers.AutoModelForCausalLM,
    training_args: TrainingArguments,
    lora_args: LoraArguments,
):
    if lora_args.lora_target_modules == "all":
        target_modules = find_all_linear_names(model)
    else:
        modules = lora_args.lora_target_modules.split(" ")
        target_modules = [mod.strip() for mod in modules if len(mod.strip()) > 0]

    print_rank0("target_modules: ", target_modules)
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        # task_type="CAUSAL_LM",
        # modules_to_save=["lm_head", "embed_tokens"],  # because we retrain the embedding
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)
    return model


# Borrowed from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L68
def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
# Borrowed from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L68
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def extract_unmasked_chunks(labels: List[int], masked_value) -> List[List[int]]:
    """This function is used to extract unmasked chunks of integer
    For example, labels = [-100, -100, 1, 2, 3, -100, -100, 4, 5] --> chunks = [[1,2,3], [4,5]]
    Args:
        labels (List[int]): list of integer containing token_id and -100

    Returns:
        List[List[int]]: list of chunk, for example: [[1,2,3], [4,5]]
    """
    chunks = []
    chunk = []
    for token_id in labels:
        if token_id != masked_value:
            chunk.append(token_id)
        else:
            if len(chunk) > 0:
                chunks.append(chunk)
                chunk = []
    if len(chunk) > 0:
        chunks.append(chunk)
    return chunks


def print_some_examples(ds, tokenizer):
    data_loader = DataLoader(ds, batch_size=1)
    count = 0
    for batch in data_loader:
        if count == 0:
            print_rank0("keys in batch: ", batch.keys())
        print_rank0("--------------****Example data point****---------------")
        print_rank0("device: ", batch["input_ids"].device)
        print_rank0("shape of input_ids: ", batch["input_ids"].shape)  # B x L
        print_rank0("shape of labels: ", batch["labels"].shape)
        print_rank0("shape of attention_mask: ", batch["attention_mask"].shape)
        # print_rank0('input_ids: ', batch["input_ids"].tolist())
        # print_rank0('labels: ', batch["labels"].tolist())
        print_rank0("attention mask: ", batch["attention_mask"])
        input_ids = batch["input_ids"][0].tolist()
        input_chunk = extract_unmasked_chunks(input_ids, tokenizer.pad_token_id)
        assert len(input_chunk) == 1
        print_rank0("+ inputs: ")
        print_rank0(tokenizer.decode(input_chunk[0]))
        labels = batch["labels"][0].tolist()
        label_chunks = extract_unmasked_chunks(labels, -100)
        print_rank0("----------")
        for chunk in label_chunks:
            print_rank0("+ chunk: ")
            print_rank0(tokenizer.decode(chunk))
        count += 1
        if count == 5:
            break


def initialize_tokenizer(
    model: transformers.AutoModelForCausalLM,
    model_name_or_path: str,
    model_max_length: int,
    cache_dir: str,
    prompt_template_version: str,
):
    """Initialize tokenizer and add special tokens, resizing vocab and embedding"""
    # note that must set legacy=True, read more: https://github.com/huggingface/transformers/issues/25176
    if model.config.model_type == 'qwen2':
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            model_max_length=model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            model_max_length=model_max_length,
            legacy=True,
        )

    # Add special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if model.config.model_type == 'qwen2':
        tokenizer.pad_token = "<|endoftext|>"

    # prompt_template = prompt_template = get_prompt_template_by_version(
    #     prompt_template_version
    # )
    # special_tokens = {
    #     "additional_special_tokens": prompt_template.get_additional_tokens()
    # }
    # num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    # # Resize embedding
    # model.resize_token_embeddings(len(tokenizer))
    # if num_new_tokens > 0:
    #     input_embeddings = model.get_input_embeddings().weight.data
    #     output_embeddings = model.get_output_embeddings().weight.data

    #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
    #         dim=0, keepdim=True
    #     )
    #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
    #         dim=0, keepdim=True
    #     )

    #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
    #     output_embeddings[-num_new_tokens:] = output_embeddings_avg

    return tokenizer


def train():
    argument_parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = argument_parser.parse_args_into_dataclasses()

    assert not lora_args.q_lora, "QLoRA is not supported for MLX"
    assert not training_args.gradient_checkpointing, "Gradient checkpointing is not supported"

    print_rank0("lora args: ", lora_args)

    print_rank0("training args: ", training_args)

    # model = load_model_with_rope_scaling(
    #     model_args, training_args, lora_args, data_args
    # )
    # print_rank0(model)

    # tokenizer = initialize_tokenizer(
    #     model,
    #     model_args.model_name_or_path,
    #     training_args.model_max_length,
    #     training_args.cache_dir,
    #     training_args.prompt_template_version,
    # )
    #
    model, tokenizer = mlx_lm.load(model_args.model_name_or_path)

    if data_args.packing:
        if model.model_type == "qwen2":
            print_rank0("using Monkey-patched Qwen2")

            def patched_call(self, inputs, cache=None):
                h = self.embed_tokens(inputs)

                mask = cache
                if mask is None:
                    if h.shape[1] > 1:
                        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
                        mask = mask.astype(h.dtype)
                else:
                    slices = [mask[i] for i in range(mask.shape[0])]
                    masks = [(~mx.triu(m[:, None] == m[None])) * -1e9 for m in slices]
                    mask = mx.expand_dims(mx.stack(masks, 0), 1)
                    mask = mask.astype(h.dtype)

                cache = [None] * len(self.layers)

                for e, layer in enumerate(self.layers):
                    h, cache[e] = layer(h, mask, cache[e])

                return self.norm(h), cache

            model.model.__call__ = patched_call.__get__(model.model, type(model.model))
        else:
            print_rank0("packing only supports models: Qwen2")
            sys.exit(1)

    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.padding_side = "right"
    tokenizer.pad_token = "<|endoftext|>"

    print_rank0(model)

    assert data_args.train_data_path is not None, "Please provide a training data file."

    if True:
        train_dataset = read_dataset(data_args, training_args, tokenizer, "train")
        print_rank0("****** Examples from train_dataset *****")
        print_some_examples(train_dataset, tokenizer)
        print_rank0("final train size: ", len(train_dataset))

        if training_args.do_eval:
            eval_dataset = read_dataset(data_args, training_args, tokenizer, "eval")
            print_rank0("final eval size: ", len(eval_dataset))
            print_rank0("****** Examples from eval_dataset *****")
            print_some_examples(eval_dataset, tokenizer)

    print_rank0("tokenizer.model_max_length: ", tokenizer.model_max_length)

    model = prepare_model_for_training(model, training_args, lora_args)

    if lora_args.q_lora:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if type(module).__name__ == 'RMSNorm':
                module = module.to(torch.bfloat16)
            if 'lm_head' in name or 'embed_tokens' in name or "wte" in name or "wpe" in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    def preprocess_logits_for_metrics(logits, labels):
        """Preprocesses the logits during evaluation by computing the greedy token predictions for
        accuracy calculation and loss values for perplexity calculation. Both pred_ids and loss are
        of shape (batch_size x seq_len)"""
        pred_ids = logits.argmax(axis=-1)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss_mask = (shift_labels != -100)
        loss = cross_entropy(shift_logits, shift_labels) * loss_mask
        loss = loss.reshape(logits.shape[0], -1).mean(axis=-1)

        return pred_ids, loss

    def compute_metrics(eval_preds):
        """Computes next-token accuracy and perplexity metrics for evaluation"""
        predictions = eval_preds.predictions[0][:, :-1]
        labels = eval_preds.label_ids[:, 1:]

        # Calculate accuracy
        acc_count = 0
        total_num = 0
        for pred, label in zip(
            predictions.flatten().tolist(), labels.flatten().tolist()
        ):
            if label != -100:
                if label == pred:
                    acc_count += 1
                total_num += 1

        # Calculate perplexity
        loss = eval_preds.predictions[1].tolist()
        loss = sum(loss) / len(loss)
        perplexity = math.exp(loss)

        return {"accuracy": acc_count / total_num, "perplexity": perplexity}

    data_collator = None
    if not data_args.packing:
        data_collator = SimpleCollator(tokenizer)

    if training_args.do_eval:
        trainer = MLXTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            data_collator=data_collator,
        )
    else:
        trainer = MLXTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

    # Resuming is not supported
    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    trainer.train()
    trainer.save_state()

    if training_args.local_rank == 0:
        flat_params = tree_flatten(model.trainable_parameters())
        mx.savez(os.path.join(training_args.output_dir, "mlx_model.npz"), **dict(flat_params))

if __name__ == "__main__":
    train()
