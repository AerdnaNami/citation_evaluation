#!/usr/bin/env python3
# train.py
#
# Config-driven QLoRA SFT for a decoder-only model on prompt+completion data (JSON or JSONL),
# plus post-train evaluation on the *eval_path* file using:
#
# Pairwise F1 definition:
# “For each pair of annotators, we treat one annotator’s data as an experiment and the other
# annotator’s data as gold truth and compute precision, recall and F1 based on that.
# Then, to get the final F1 score, all pairwise F1 scores are micro-averaged.”
#
# Usage:
#   python train.py --config config.json --category Unsupported_claim
#
# Notes:
# - If eval_path examples contain multiple annotators (e.g. "annotations": {"ann1":[...], "ann2":[...]}),
#   we compute pairwise micro-averaged F1 across annotators.
# - If eval_path contains only a single gold (e.g. "completion" or "gold_spans"), we compute model-vs-gold F1.

import argparse
import json
import os
import socket
import platform
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from itertools import permutations, combinations

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

IGNORE_INDEX = -100


# -------------------------
# Config helpers
# -------------------------
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_dicts(base: dict, override: dict) -> dict:
    out = dict(base or {})
    if override is None:
        return out
    if not isinstance(override, dict):
        raise TypeError(f"Expected category config to be a dict, got {type(override)}: {override}")
    out.update(override)
    return out


# -------------------------
# Robust JSON/JSONL reader
# -------------------------
def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                preview = line[:200].replace("\n", "\\n")
                raise ValueError(f"Bad JSON on line {lineno} in {path}: {e}\nPreview: {preview}") from e
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                # allow non-dicts but keep consistent error later
                rows.append({"_value": obj})
    return rows


def _normalize_json(obj: Any, path: str) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        out: List[Dict[str, Any]] = []
        for it in obj:
            if isinstance(it, dict):
                out.append(it)
            else:
                out.append({"_value": it})
        return out
    if isinstance(obj, dict):
        for k in ("data", "rows", "examples", "items"):
            if k in obj and isinstance(obj[k], list):
                out: List[Dict[str, Any]] = []
                for it in obj[k]:
                    if isinstance(it, dict):
                        out.append(it)
                    else:
                        out.append({"_value": it})
                return out
    raise ValueError(f"{path} JSON must be a list of examples (or dict containing one). Got: {type(obj)}")


def read_examples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"{path} is empty")

    if text[0] in "[{":
        try:
            obj = json.loads(text)
            return _normalize_json(obj, path)
        except json.JSONDecodeError:
            return _load_jsonl(path)
    return _load_jsonl(path)


# -------------------------
# Dataset (train/dev)
# -------------------------
class PromptCompletionDataset(Dataset):
    def __init__(self, path: str, prompt_key: str = "prompt", completion_key: str = "completion"):
        self.rows: List[Dict[str, Any]] = []
        self.prompt_key = prompt_key
        self.completion_key = completion_key

        self.rows = read_examples(path)
        self._validate_rows(path)

    def _validate_rows(self, path: str) -> None:
        cleaned = []
        bad = 0
        for r in self.rows:
            if not isinstance(r, dict) or len(r) == 0:
                bad += 1
                continue
            if self.prompt_key not in r or self.completion_key not in r:
                bad += 1
                continue
            if not isinstance(r[self.prompt_key], str) or not isinstance(r[self.completion_key], str):
                bad += 1
                continue
            if r[self.prompt_key].strip() == "" or r[self.completion_key].strip() == "":
                bad += 1
                continue
            cleaned.append(r)

        if bad > 0:
            print(f"[WARN] Dropped {bad} invalid/empty examples from {path}. Kept {len(cleaned)}.")

        if len(cleaned) == 0:
            raise ValueError(f"{path}: no valid examples left after cleaning.")
        self.rows = cleaned

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        r = self.rows[idx]
        return {"prompt": r[self.prompt_key], "completion": r[self.completion_key]}


# -------------------------
# Collator
# -------------------------
@dataclass
class PromptCompletionCollator:
    tokenizer: Any
    max_length: int = 2048

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        for ex in batch:
            prompt = ex.get("prompt", "")
            completion = ex.get("completion", "")

            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            comp_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]

            # ensure at least 1 supervised token
            if len(comp_ids) == 0:
                comp_ids = [eos_id]

            if len(comp_ids) > self.max_length:
                comp_ids = comp_ids[-self.max_length:]

            max_prompt_len = max(0, self.max_length - len(comp_ids))
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + comp_ids
            attention_mask = [1] * len(input_ids)

            labels = input_ids.copy()
            labels[:len(prompt_ids)] = [IGNORE_INDEX] * len(prompt_ids)

            if all(x == IGNORE_INDEX for x in labels):
                labels[-1] = input_ids[-1]

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -------------------------
# Checkpoint helpers
# -------------------------
def _latest_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    ckpts = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            p = os.path.join(output_dir, name)
            if os.path.isdir(p):
                try:
                    step = int(name.split("-")[-1])
                except Exception:
                    step = -1
                ckpts.append((step, p))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def write_save_reason_json(output_dir: str, entry: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "save_reasons.json")

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if not isinstance(data, list):
        data = [data]

    data.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# -------------------------
# Evaluation helpers
# -------------------------
def _norm_span(s: str) -> str:
    # minimal normalization for exact-match span comparison
    return " ".join(s.strip().strip('"').strip("'").split())


def _parse_generated_to_spans(text: str) -> List[str]:
    """
    Robust-ish parser:
    - If model returns JSON list/dict: extract spans from:
        - [{"span": "..."}] or ["..."]
        - {"spans":[...]} or {"Unsupported claim":[...]} etc.
    - Else: treat as lines / bullets / comma-separated
    """
    t = text.strip()

    # strip markdown fences if present
    if t.startswith("```"):
        # remove first fence line and last fence if any
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1].strip()
        else:
            t = t.replace("```", "").strip()

    # attempt JSON
    if t and t[0] in "[{":
        try:
            obj = json.loads(t)
            spans: List[str] = []

            def collect(o: Any):
                nonlocal spans
                if isinstance(o, str):
                    spans.append(o)
                elif isinstance(o, dict):
                    # common fields
                    if "span" in o and isinstance(o["span"], str):
                        spans.append(o["span"])
                    for k, v in o.items():
                        if k.lower() in {"spans", "unsupported claim", "unsupported_claim", "claims", "items"}:
                            collect(v)
                        else:
                            # still recurse: could be nested structure
                            collect(v)
                elif isinstance(o, list):
                    for it in o:
                        collect(it)

            collect(obj)
            spans = [_norm_span(s) for s in spans if _norm_span(s)]
            # de-dup preserve order
            seen = set()
            out = []
            for s in spans:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            if out:
                return out
        except Exception:
            pass

    # fallback: lines/bullets/commas
    # remove leading label prefixes
    t = t.replace("Unsupported claim:", "").replace("Unsupported_claim:", "").strip()
    lines = [ln.strip(" \t-•*").strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) == 1:
        # maybe comma-separated
        if "," in lines[0]:
            chunks = [c.strip() for c in lines[0].split(",")]
            lines = [c for c in chunks if c]
    spans = [_norm_span(s) for s in lines if _norm_span(s)]
    # de-dup preserve order
    seen = set()
    out = []
    for s in spans:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _extract_annotator_spans(example: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Return a dict: annotator_id -> list[str] spans.

    Supports common layouts:
      - {"annotations": {"ann1":[...], "ann2":[...]}}
      - {"annotators": {...}}
      - {"gold_spans":[...]}  (single gold)
      - {"completions":[...]} (single gold list)
      - {"completion": "..."} (single gold string)
    """
    # multi-annotator structures
    for key in ("annotations", "annotators", "labels_by_annotator"):
        if key in example and isinstance(example[key], dict):
            out: Dict[str, List[str]] = {}
            for ann, val in example[key].items():
                if isinstance(val, str):
                    spans = [_norm_span(val)] if _norm_span(val) else []
                elif isinstance(val, list):
                    spans = [_norm_span(x) for x in val if isinstance(x, str) and _norm_span(x)]
                else:
                    spans = []
                out[str(ann)] = spans
            # keep only annotators with at least something (still allow empty though)
            return out

    # single gold variants
    if "gold_spans" in example and isinstance(example["gold_spans"], list):
        spans = [_norm_span(x) for x in example["gold_spans"] if isinstance(x, str) and _norm_span(x)]
        return {"gold": spans}

    if "completions" in example and isinstance(example["completions"], list):
        spans = [_norm_span(x) for x in example["completions"] if isinstance(x, str) and _norm_span(x)]
        return {"gold": spans}

    if "completion" in example and isinstance(example["completion"], str):
        s = _norm_span(example["completion"])
        return {"gold": [s] if s else []}

    return {"gold": []}


def _micro_counts(preds: List[str], golds: List[str]) -> Tuple[int, int, int]:
    ps = set(_norm_span(x) for x in preds if _norm_span(x))
    gs = set(_norm_span(x) for x in golds if _norm_span(x))
    tp = len(ps & gs)
    fp = len(ps - gs)
    fn = len(gs - ps)
    return tp, fp, fn


def _prf_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def pairwise_micro_averaged_f1(ann_spans: Dict[str, List[List[str]]]) -> Dict[str, Any]:
    """
    ann_spans: annotator -> list of examples -> list of spans
      e.g. ann_spans["ann1"][i] = ["span a", "span b"]
    Computes pairwise (directed) counts and micro-averages across all directed pairs.
    """
    annotators = [a for a in ann_spans.keys() if a is not None]
    if len(annotators) < 2:
        return {"pairwise_micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "num_annotators": len(annotators)}

    # ensure same number of examples for all
    n = min(len(ann_spans[a]) for a in annotators)
    for a in annotators:
        ann_spans[a] = ann_spans[a][:n]

    total_tp = total_fp = total_fn = 0
    per_pair: Dict[str, Dict[str, float]] = {}

    for a, b in permutations(annotators, 2):
        tp = fp = fn = 0
        for i in range(n):
            tpi, fpi, fni = _micro_counts(ann_spans[a][i], ann_spans[b][i])
            tp += tpi
            fp += fpi
            fn += fni
        per_pair[f"{a}→{b}"] = _prf_from_counts(tp, fp, fn)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    return {
        "pairwise_micro": _prf_from_counts(total_tp, total_fp, total_fn),
        "num_annotators": len(annotators),
        "num_examples": n,
        "per_pair": per_pair,
    }


@torch.no_grad()
def evaluate_model_on_eval_file(
    model,
    tokenizer,
    eval_path: str,
    prompt_key: str = "prompt",
    max_new_tokens: int = 128,
    gen_temperature: float = 0.0,
    gen_top_p: float = 1.0,
) -> Dict[str, Any]:
    """
    Evaluates model predictions vs gold spans. Also computes pairwise micro-averaged F1 across annotators
    if multiple annotators exist in the eval file.

    Expected eval example schemas supported:
      - {"prompt": "...", "completion": "..."}                     (single gold)
      - {"prompt": "...", "gold_spans": ["...", "..."]}            (single gold list)
      - {"prompt": "...", "annotations": {"a": [...], "b": [...]}} (multi-annotator)
    """
    examples = read_examples(eval_path)
    # filter examples that have prompt
    filtered = [ex for ex in examples if isinstance(ex, dict) and isinstance(ex.get(prompt_key), str) and ex.get(prompt_key).strip()]
    if not filtered:
        return {"error": f"No valid examples with prompt_key='{prompt_key}' in {eval_path}"}

    model.eval()
    device = model.device

    # collect gold spans by annotator (if present)
    # ann_spans[ann][i] = list[str] gold spans for example i
    ann_spans: Dict[str, List[List[str]]] = {}
    # collect model preds per example
    preds: List[List[str]] = []

    for ex in filtered:
        prompt = ex[prompt_key]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(gen_temperature > 0.0),
            temperature=gen_temperature if gen_temperature > 0.0 else None,
            top_p=gen_top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # decode only the newly generated portion
        gen_ids = gen[0].tolist()
        in_len = inputs["input_ids"].shape[1]
        new_ids = gen_ids[in_len:]
        pred_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        pred_spans = _parse_generated_to_spans(pred_text)
        preds.append(pred_spans)

        gold_by_ann = _extract_annotator_spans(ex)
        for ann, spans in gold_by_ann.items():
            ann_spans.setdefault(ann, []).append(spans)

    # If multiple annotators in gold, compute pairwise micro-averaged F1 between annotators
    pairwise = pairwise_micro_averaged_f1(dict(ann_spans)) if len(ann_spans.keys()) >= 2 else None

    # Compute model-vs-gold micro P/R/F1.
    # If multiple annotators exist, we compute model vs EACH annotator as gold and micro-aggregate across annotators.
    total_tp = total_fp = total_fn = 0
    per_gold: Dict[str, Dict[str, float]] = {}

    for ann, gold_lists in ann_spans.items():
        tp = fp = fn = 0
        for i in range(min(len(preds), len(gold_lists))):
            tpi, fpi, fni = _micro_counts(preds[i], gold_lists[i])
            tp += tpi
            fp += fpi
            fn += fni
        per_gold[ann] = _prf_from_counts(tp, fp, fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    model_vs_gold_micro = _prf_from_counts(total_tp, total_fp, total_fn)

    return {
        "eval_path": eval_path,
        "num_examples": len(preds),
        "model_vs_gold_micro": model_vs_gold_micro,
        "model_vs_each_gold": per_gold,
        "pairwise_annotator_micro": pairwise["pairwise_micro"] if pairwise else None,
        "pairwise_details": pairwise,
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--category", required=True, help="Category name from config.json")
    args_cli = parser.parse_args()

    os.environ.setdefault("BF16", "1")
    os.environ.setdefault("FP16", "0")

    cfg = load_config(args_cli.config)

    defaults = cfg.get("defaults", {})
    if not isinstance(defaults, dict):
        raise TypeError(f"'defaults' must be a dict, got {type(defaults)}")

    base_model = cfg.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")

    categories = cfg.get("categories", [])
    if not isinstance(categories, list):
        raise TypeError(f"'categories' must be a list, got {type(categories)}")

    cat_cfg = next((c for c in categories if isinstance(c, dict) and c.get("name") == args_cli.category), None)
    if cat_cfg is None:
        available = [c.get("name") for c in categories if isinstance(c, dict)]
        raise ValueError(f"Category '{args_cli.category}' not found. Available: {available}")

    run_cfg = merge_dicts(defaults, cat_cfg)

    # Keys (allow config override)
    prompt_key = run_cfg.get("prompt_key", "prompt")
    completion_key = run_cfg.get("completion_key", "completion")

    # Paths
    train_path = run_cfg["train_path"]
    dev_path = run_cfg.get("dev_path")
    eval_path = run_cfg.get("eval_path")

    eval_split = run_cfg.get("eval_split", "dev")
    eval_path_for_training = dev_path if eval_split == "dev" else eval_path

    # Output dir: same folder as data (category folder), run_name subdir
    data_dir = Path(train_path).resolve().parent
    run_name = run_cfg.get("run_name", args_cli.category)
    out_dir = str(data_dir / run_name)

    # Model + training params
    model_name = run_cfg.get("model_name", base_model)

    max_length = int(run_cfg.get("max_length", 2048))
    per_device_train_batch_size = int(run_cfg.get("per_device_train_batch_size", 1))
    per_device_eval_batch_size = int(run_cfg.get("per_device_eval_batch_size", per_device_train_batch_size))
    gradient_accumulation_steps = int(run_cfg.get("gradient_accumulation_steps", 16))

    learning_rate = float(run_cfg.get("learning_rate", 2e-4))
    num_train_epochs = float(run_cfg.get("num_train_epochs", 1))
    warmup_ratio = float(run_cfg.get("warmup_ratio", 0.03))

    logging_steps = int(run_cfg.get("logging_steps", 25))
    save_steps = int(run_cfg.get("save_steps", 500))
    save_reason = str(run_cfg.get("save_reason", "config-driven run"))

    bf16 = bool(run_cfg.get("bf16", True))
    fp16 = bool(run_cfg.get("fp16", False))

    lora_r = int(run_cfg.get("lora_r", 16))
    lora_alpha = int(run_cfg.get("lora_alpha", 32))
    lora_dropout = float(run_cfg.get("lora_dropout", 0.05))

    torch_compile = bool(run_cfg.get("torch_compile", False))
    use_flash_attn = bool(run_cfg.get("flash_attn", False))

    num_workers = int(run_cfg.get("dataloader_num_workers", 4))
    max_grad_norm = float(run_cfg.get("max_grad_norm", 1.0))

    # Generation settings for eval
    max_new_tokens = int(run_cfg.get("max_new_tokens", 128))
    gen_temperature = float(run_cfg.get("gen_temperature", 0.0))
    gen_top_p = float(run_cfg.get("gen_top_p", 1.0))

    print("Settings (from config):")
    print(f"  Category: {args_cli.category}")
    print(f"  Model: {model_name}")
    print(f"  Train data: {train_path}")
    print(f"  Dev/Eval used during training: {eval_path_for_training}")
    print(f"  Eval file for post-train scoring: {eval_path}")
    print(f"  Output dir: {out_dir}")
    print(f"  max_length: {max_length}")
    print(f"  train_bs: {per_device_train_batch_size}, eval_bs: {per_device_eval_batch_size}, grad_accum: {gradient_accumulation_steps}")
    print(f"  lr: {learning_rate}, epochs: {num_train_epochs}, warmup_ratio: {warmup_ratio}")
    print(f"  bf16: {bf16}, fp16: {fp16}")
    print(f"  LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"  flash_attn: {use_flash_attn}, torch_compile: {torch_compile}")
    print(f"  num_workers: {num_workers}, max_grad_norm: {max_grad_norm}")
    print(f"  prompt_key: {prompt_key}, completion_key: {completion_key}")
    print(f"  eval generation: max_new_tokens={max_new_tokens}, temperature={gen_temperature}, top_p={gen_top_p}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Tokenizer loaded.")

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if bf16 else (torch.float16 if fp16 else None),
    )
    print("Model loaded.")

    if use_flash_attn:
        try:
            model.config.attn_implementation = "flash_attention_2"
            print("Enabled flash_attention_2")
        except Exception as e:
            print("Could not enable flash_attention_2:", e)

    print("Prepping for kbit training and adding LoRA adapters...")
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if torch_compile:
        try:
            model = torch.compile(model)
            print("Enabled torch.compile")
        except Exception as e:
            print("Could not enable torch.compile:", e)

    print("Loading datasets...")
    train_ds = PromptCompletionDataset(train_path, prompt_key=prompt_key, completion_key=completion_key)
    eval_ds = (
        PromptCompletionDataset(eval_path_for_training, prompt_key=prompt_key, completion_key=completion_key)
        if eval_path_for_training and os.path.exists(eval_path_for_training)
        else None
    )
    collator = PromptCompletionCollator(tokenizer=tokenizer, max_length=max_length)

    # Training args
    train_args = TrainingArguments(
        remove_unused_columns=False,
        output_dir=out_dir,
        overwrite_output_dir=False,

        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        bf16=bf16,
        fp16=fp16,

        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",

        dataloader_num_workers=num_workers,
        gradient_checkpointing=True,
        tf32=True,
        max_grad_norm=max_grad_norm,

        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        save_safetensors=True,
        logging_dir=os.path.join(out_dir, "logs"),

        # transformers older versions use eval_strategy
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=save_steps if eval_ds is not None else None,

        report_to="none",
        group_by_length=False,
    )

    # --- SANITY CHECK ---
    sample = [train_ds[0]]
    batch = collator(sample)
    num_label_tokens = (batch["labels"] != IGNORE_INDEX).sum().item()
    seq_len = batch["input_ids"].shape[1]
    print(f"[SANITY] seq_len={seq_len}, num_label_tokens={num_label_tokens}")
    if num_label_tokens == 0:
        raise RuntimeError("[SANITY FAIL] num_label_tokens == 0 (all labels masked).")

    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            labels=batch["labels"].to(model.device),
        )
    print(f"[SANITY] forward loss={out.loss.item()}")
    model.train()

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        label_names=["labels"],  # important for PEFT Trainer compatibility
    )

    # Resume if possible
    resume_ckpt = _latest_checkpoint(out_dir)
    if resume_ckpt:
        print(f"Resuming from latest checkpoint: {resume_ckpt}")
        train_output = trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        print("No checkpoint found; starting fresh.")
        train_output = trainer.train()

    # Final save
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Post-train evaluation on eval_path (span-level)
    eval_metrics = None
    if eval_path and os.path.exists(eval_path):
        try:
            print(f"Running post-train evaluation on eval_path: {eval_path}")
            eval_metrics = evaluate_model_on_eval_file(
                model=model,
                tokenizer=tokenizer,
                eval_path=eval_path,
                prompt_key=prompt_key,
                max_new_tokens=max_new_tokens,
                gen_temperature=gen_temperature,
                gen_top_p=gen_top_p,
            )
            print("[EVAL] model_vs_gold_micro:", eval_metrics.get("model_vs_gold_micro"))
            if eval_metrics.get("pairwise_annotator_micro") is not None:
                print("[EVAL] pairwise_annotator_micro:", eval_metrics.get("pairwise_annotator_micro"))

            with open(os.path.join(out_dir, "eval_pairwise_f1.json"), "w", encoding="utf-8") as f:
                json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
            print(f"[EVAL] Wrote eval metrics to: {os.path.join(out_dir, 'eval_pairwise_f1.json')}")
        except Exception as e:
            print("[EVAL] Failed to compute eval metrics:", repr(e))

    entry = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "reason": save_reason,
        "category": args_cli.category,
        "model_name": model_name,
        "train_path": train_path,
        "eval_path_training": eval_path_for_training if eval_ds is not None else None,
        "eval_path_post": eval_path if eval_path and os.path.exists(eval_path) else None,
        "out_dir": out_dir,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "a100_tuned": True,
        "final_save": True,
        "global_step": int(getattr(trainer.state, "global_step", -1)),
        "train_runtime_sec": float(getattr(train_output, "metrics", {}).get("train_runtime", -1)),
        "lora": {"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout},
        "quantization": "4bit_nf4",
        "max_length": max_length,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "grad_accum": gradient_accumulation_steps,
        "lr": learning_rate,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": True,
        "tf32": True,
        "flash_attn": use_flash_attn,
        "torch_compile": torch_compile,
        "config_path": args_cli.config,
        "run_cfg": run_cfg,
        "post_eval_metrics": eval_metrics,
    }
    write_save_reason_json(out_dir, entry=entry)

    print(f"Saved (LoRA adapters + tokenizer) to: {out_dir}")
    print(f"Wrote save reasons to: {os.path.join(out_dir, 'save_reasons.json')}")


if __name__ == "__main__":
    main()
