"""
Model loading and dual forward-pass utilities for token-level RAG attribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-7B"
    max_gpu_memory_gib: int = 6
    max_model_tokens: int = 2048
    max_response_tokens: int = 256
    max_query_tokens: int = 128
    enforce_cuda: bool = True
    gpu_index: int = 0


def _find_subsequence(arr: List[int], subseq: List[int]) -> int:
    if not subseq or len(subseq) > len(arr):
        return -1
    m = len(subseq)
    for i in range(len(arr) - m + 1):
        if arr[i : i + m] == subseq:
            return i
    return -1


def load_generator_model(cfg: ModelConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    if cfg.enforce_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Disable enforce_cuda to allow CPU fallback.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        max_memory = {cfg.gpu_index: f"{cfg.max_gpu_memory_gib}GiB"}
        device_map = {"": cfg.gpu_index} if cfg.enforce_cuda else "auto"
    else:
        max_memory = {"cpu": "32GiB"}
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch.float16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def _build_sequences(
    tokenizer,
    query: str,
    context: str,
    response: str,
    max_model_tokens: int,
    max_response_tokens: int,
    max_query_tokens: int,
) -> Tuple[List[int], List[int], int, int, List[int]]:
    response_ids = tokenizer.encode(response, add_special_tokens=False)[:max_response_tokens]
    query_ids = tokenizer.encode(query, add_special_tokens=False)[:max_query_tokens]
    context_ids = tokenizer.encode(context, add_special_tokens=False)

    q_prefix = tokenizer.encode("Question: ", add_special_tokens=False)
    ctx_prefix = tokenizer.encode("\nContext: ", add_special_tokens=False)
    ans_prefix_with = tokenizer.encode("\nAnswer: ", add_special_tokens=False)
    ans_prefix_wo = tokenizer.encode("\nAnswer: ", add_special_tokens=False)

    overhead_with = len(q_prefix) + len(query_ids) + len(ctx_prefix) + len(ans_prefix_with) + len(response_ids)
    available_context = max_model_tokens - overhead_with
    if available_context < 0:
        reduce_by = -available_context
        keep = max(8, len(response_ids) - reduce_by)
        response_ids = response_ids[:keep]
        overhead_with = len(q_prefix) + len(query_ids) + len(ctx_prefix) + len(ans_prefix_with) + len(response_ids)
        available_context = max(0, max_model_tokens - overhead_with)

    context_ids = context_ids[: max(0, available_context)]

    with_ids = q_prefix + query_ids + ctx_prefix + context_ids + ans_prefix_with + response_ids
    without_ids = q_prefix + query_ids + ans_prefix_wo + response_ids

    if len(without_ids) > max_model_tokens:
        excess = len(without_ids) - max_model_tokens
        keep = max(8, len(response_ids) - excess)
        response_ids = response_ids[:keep]
        with_ids = q_prefix + query_ids + ctx_prefix + context_ids + ans_prefix_with + response_ids
        without_ids = q_prefix + query_ids + ans_prefix_wo + response_ids

    with_resp_start = len(q_prefix) + len(query_ids) + len(ctx_prefix) + len(context_ids) + len(ans_prefix_with)
    without_resp_start = len(q_prefix) + len(query_ids) + len(ans_prefix_wo)

    return with_ids, without_ids, with_resp_start, without_resp_start, response_ids


def dual_forward_pass(
    model,
    tokenizer,
    query: str,
    context: str,
    response: str,
    cfg: ModelConfig,
):
    """
    Runs a dual pass to get p(y_t | x, y_<t, z) and p(y_t | x, y_<t).

    Returns:
        p_with_list: list[np.ndarray] shape [vocab]
        p_without_list: list[np.ndarray] shape [vocab]
        response_token_ids: np.ndarray shape [n_tokens]
    """
    with_ids, without_ids, with_start, without_start, response_ids = _build_sequences(
        tokenizer=tokenizer,
        query=query,
        context=context,
        response=response,
        max_model_tokens=cfg.max_model_tokens,
        max_response_tokens=cfg.max_response_tokens,
        max_query_tokens=cfg.max_query_tokens,
    )

    if len(response_ids) < 2:
        return [], [], np.array([], dtype=np.int64)

    device = model.get_input_embeddings().weight.device
    with_inputs = torch.tensor([with_ids], dtype=torch.long, device=device)
    without_inputs = torch.tensor([without_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits_with = model(with_inputs).logits[0]
        logits_without = model(without_inputs).logits[0]

    max_scored = min(
        len(response_ids),
        logits_with.shape[0] - with_start,
        logits_without.shape[0] - without_start,
    )

    p_with_list = []
    p_without_list = []

    for j in range(max_scored):
        pos_with = with_start + j - 1
        pos_without = without_start + j - 1
        if pos_with < 0 or pos_without < 0:
            continue
        if pos_with >= logits_with.shape[0] or pos_without >= logits_without.shape[0]:
            continue

        pw = F.softmax(logits_with[pos_with], dim=-1).detach().cpu().to(torch.float32).numpy()
        p0 = F.softmax(logits_without[pos_without], dim=-1).detach().cpu().to(torch.float32).numpy()
        p_with_list.append(pw)
        p_without_list.append(p0)

    del with_inputs, without_inputs, logits_with, logits_without
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(p_with_list)
    return p_with_list, p_without_list, np.array(response_ids[:n], dtype=np.int64)
