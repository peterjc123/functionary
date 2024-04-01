import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from transformers import AutoModelForCausalLM, AutoTokenizer
from functionary.prompt_template import get_prompt_template_by_version
import torch
import typer
import transformers
import math
import mlx.core as mx
import numpy as np


def merge_weight(save_folder: str, pretrained_path: str, checkpoint: str, lora_rank: int, lora_alpha: float, lora_dropout: float):
    print("save to: ", save_folder)
    print("pretrained: ", pretrained_path)
    print("checkpoint: ", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # prompt_template = get_prompt_template_by_version(prompt_template_version)
    # special_tokens = {"additional_special_tokens": prompt_template.get_additional_tokens()}
    # num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    # print("number of new tokens: ", num_new_tokens)

    config = transformers.AutoConfig.from_pretrained(
        pretrained_path, trust_remote_code=True
    )

    if config.model_type == 'qwen2':
        tokenizer.pad_token = "<|endoftext|>"

    # orig_ctx_len = getattr(config, "max_position_embeddings", None)
    # if orig_ctx_len and model_max_length > orig_ctx_len:
    #     print("need to scale ...")
    #     scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
    #     config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    # config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print("model = ", model)
    # model.resize_token_embeddings(len(tokenizer))

    mlx_weights = mx.load(os.path.join(checkpoint, "mlx_model.npz"))
    target_modules = set()
    weight_map = {}
    for name, weight in mlx_weights.items():
        if '.lora_B.' in name:
            module_name = name.split('.lora_B')[0]
            weight_map.setdefault(module_name, {})
            target_modules.add(module_name)
            weight_map[module_name]['lora_B'] = weight
        elif '.lora_A.' in name:
            module_name = name.split('.lora_A')[0]
            weight_map.setdefault(module_name, {})
            target_modules.add(module_name)
            weight_map[module_name]['lora_A'] = weight
        else:
            raise AssertionError(f"Unknown weight: {name}")

    for name, adapter_dict in weight_map.items():
        cur_mod = model
        for key in name.split('.'):
            cur_mod = getattr(cur_mod, key)
        weight = getattr(cur_mod, 'weight')
        delta_weight = (adapter_dict['lora_B'] @ adapter_dict['lora_A']) * (lora_alpha * 1. / lora_rank)
        delta_weight_torch = torch.from_numpy(np.array(delta_weight.astype(mx.float32))).to(torch.bfloat16)
        weight.data += delta_weight_torch

    model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)
    print("final lora model: ", model)


if __name__ == "__main__":
    typer.run(merge_weight)
