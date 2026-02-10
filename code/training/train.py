import argparse
import json
import os
import socket
import platform
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from itertools import permutations

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
    return " ".join(str(s).strip().strip('"').strip("'").split())


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)


def _tokenize_span(s: str) -> List[str]:
    s = _norm_span(s).lower()
    return _TOKEN_RE.findall(s)


def _span_iou_tokens(a: str, b: str) -> float:
    ta = set(_tokenize_span(a))
    tb = set(_tokenize_span(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def match_spans_overlap_tokens(
    gold_spans: List[str],
    pred_spans: List[str],
    *,
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Greedy 1-1 matching between gold and pred spans using token IoU (Jaccard).
    Returns: (tp, fp, fn)
    """
    gold = [_norm_span(x) for x in (gold_spans or []) if _norm_span(x)]
    pred = [_norm_span(x) for x in (pred_spans or []) if _norm_span(x)]

    if not gold and not pred:
        return 0, 0, 0
    if not gold:
        return 0, len(pred), 0
    if not pred:
        return 0, 0, len(gold)

    candidates: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            iou = _span_iou_tokens(g, p)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    matched_gold = set()
    matched_pred = set()
    tp = 0

    for iou, gi, pi in candidates:
        if gi in matched_gold or pi in matched_pred:
            continue
        matched_gold.add(gi)
        matched_pred.add(pi)
        tp += 1

    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn


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
                    if "span" in o and isinstance(o["span"], str):
                        spans.append(o["span"])
                    for k, v in o.items():
                        if k.lower() in {"spans", "unsupported claim", "unsupported_claim", "claims", "items"}:
                            collect(v)
                        else:
                            collect(v)
                elif isinstance(o, list):
                    for it in o:
                        collect(it)

            collect(obj)
            spans = [_norm_span(s) for s in spans if _norm_span(s)]
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
    t = t.replace("Unsupported claim:", "").replace("Unsupported_claim:", "").strip()
    lines = [ln.strip(" \t-•*").strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) == 1 and "," in lines[0]:
        chunks = [c.strip() for c in lines[0].split(",")]
        lines = [c for c in chunks if c]
    spans = [_norm_span(s) for s in lines if _norm_span(s)]
    seen = set()
    out = []
    for s in spans:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _prf_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def pairwise_micro_averaged_f1_gold_vs_pred(
    examples: List[Dict[str, Any]],
    preds: List[List[str]],
    *,
    category_label: str,
    prompt_key: str = "prompt",
    gold_key: str = "completion",
    use_label: bool = True,
    gold_annotator_name: str = "gold",
    pred_annotator_name: str = "pred",
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Two annotators:
      - gold: spans from eval file (gold_key)
      - pred: spans predicted by the model (preds)
    Compute directed pairwise micro P/R/F1 (gold→pred and pred→gold) with token-IoU overlap
    and micro-average across the two directed pairs.
    """
    def pack(span: str) -> str:
        span = _norm_span(span)
        if not span:
            return ""
        return f"{category_label}|||{span}" if use_label else span

    filtered: List[Dict[str, Any]] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        p = ex.get(prompt_key)
        if not isinstance(p, str) or not p.strip():
            continue
        if gold_key not in ex:
            continue
        filtered.append(ex)

    if not filtered:
        return {
            "pairwise_micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "num_annotators": 2,
            "num_examples": 0,
            "per_pair": {},
            "note": f"No valid examples found with keys prompt_key='{prompt_key}', gold_key='{gold_key}'.",
            "iou_threshold": iou_threshold,
        }

    n = min(len(filtered), len(preds))
    filtered = filtered[:n]
    preds = preds[:n]

    ann_spans: Dict[str, List[List[str]]] = {gold_annotator_name: [], pred_annotator_name: []}

    for ex, pred_spans in zip(filtered, preds):
        gold_val = ex.get(gold_key)

        if isinstance(gold_val, str):
            gold_spans = [gold_val]
        elif isinstance(gold_val, list):
            gold_spans = [x for x in gold_val if isinstance(x, str)]
        else:
            gold_spans = []

        gold_items = [pack(s) for s in gold_spans]
        gold_items = [x for x in gold_items if x]

        pred_items = [pack(s) for s in (pred_spans or [])]
        pred_items = [x for x in pred_items if x]

        ann_spans[gold_annotator_name].append(gold_items)
        ann_spans[pred_annotator_name].append(pred_items)

    annotators = [gold_annotator_name, pred_annotator_name]
    total_tp = total_fp = total_fn = 0
    per_pair: Dict[str, Dict[str, float]] = {}

    for a, b in permutations(annotators, 2):
        tp = fp = fn = 0
        for i in range(n):
            tpi, fpi, fni = match_spans_overlap_tokens(
                ann_spans[a][i],
                ann_spans[b][i],
                iou_threshold=iou_threshold,
            )
            tp += tpi
            fp += fpi
            fn += fni
        per_pair[f"{a}→{b}"] = _prf_from_counts(tp, fp, fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return {
        "pairwise_micro": _prf_from_counts(total_tp, total_fp, total_fn),
        "num_annotators": 2,
        "num_examples": n,
        "per_pair": per_pair,
        "use_label": use_label,
        "category_label": category_label,
        "gold_key": gold_key,
        "prompt_key": prompt_key,
        "iou_threshold": iou_threshold,
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
    *,
    category_label: str,
    gold_key: str = "completion",
    use_label: bool = True,
    debug_k: int = 25,
    debug_chars: int = 600,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluates model predictions vs gold spans using token-overlap IoU matching,
    treating:
      - gold spans (+label) as annotator #1
      - predicted spans (+label) as annotator #2
    and computes pairwise micro-averaged F1 across these two annotators.

    ALSO returns debug examples with:
      - gold span(s)
      - raw generation
      - parsed predicted spans
      - per-example tp/fp/fn under token-IoU matching
    """
    examples = read_examples(eval_path)

    filtered = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        p = ex.get(prompt_key)
        if not isinstance(p, str) or not p.strip():
            continue
        if gold_key not in ex:
            continue
        filtered.append(ex)

    if not filtered:
        return {"error": f"No valid examples with prompt_key='{prompt_key}' and gold_key='{gold_key}' in {eval_path}"}

    model.eval()
    device = model.device

    preds: List[List[str]] = []
    debug_rows: List[Dict[str, Any]] = []

    for ex_i, ex in enumerate(filtered):
        prompt = ex[prompt_key]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
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

        gen_ids = gen[0].tolist()
        in_len = inputs["input_ids"].shape[1]
        new_ids = gen_ids[in_len:]
        pred_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        pred_spans = _parse_generated_to_spans(pred_text)
        preds.append(pred_spans)

        if len(debug_rows) < debug_k:
            gold_val = ex.get(gold_key)
            if isinstance(gold_val, str):
                gold_spans = [gold_val]
            elif isinstance(gold_val, list):
                gold_spans = [x for x in gold_val if isinstance(x, str)]
            else:
                gold_spans = []

            # token-IoU counts for this example (on raw spans; packing happens elsewhere)
            tp_i, fp_i, fn_i = match_spans_overlap_tokens(
                gold_spans=gold_spans,
                pred_spans=pred_spans,
                iou_threshold=iou_threshold,
            )

            # also record best IoU pairs (small + helpful)
            best_matches = []
            for g in gold_spans[:10]:
                best = 0.0
                best_p = None
                for pspan in pred_spans[:20]:
                    v = _span_iou_tokens(g, pspan)
                    if v > best:
                        best = v
                        best_p = pspan
                best_matches.append({"gold": g, "best_pred": best_p, "best_iou": best})

            debug_rows.append({
                "index": ex_i,
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
                "tp_fp_fn": {"tp": tp_i, "fp": fp_i, "fn": fn_i},
                "best_matches": best_matches,
                "raw_generation": pred_text[:debug_chars],
                "prompt_preview": prompt[:debug_chars],
            })

    pairwise_gold_vs_pred = pairwise_micro_averaged_f1_gold_vs_pred(
        examples=filtered,
        preds=preds,
        category_label=category_label,
        prompt_key=prompt_key,
        gold_key=gold_key,
        use_label=use_label,
        gold_annotator_name="gold",
        pred_annotator_name="pred",
        iou_threshold=iou_threshold,
    )

    num_empty_pred = sum(1 for p in preds if len(p) == 0)
    avg_pred_spans = sum(len(p) for p in preds) / max(1, len(preds))

    return {
        "eval_path": eval_path,
        "num_examples": pairwise_gold_vs_pred.get("num_examples", len(preds)),
        "pairwise_gold_vs_pred": pairwise_gold_vs_pred,
        "model_vs_gold_micro": pairwise_gold_vs_pred.get("per_pair", {}).get(
            "pred→gold", {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        ),
        "gold_vs_model_micro": pairwise_gold_vs_pred.get("per_pair", {}).get(
            "gold→pred", {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        ),
        "pairwise_micro": pairwise_gold_vs_pred.get(
            "pairwise_micro", {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        ),
        "debug_summary": {
            "num_empty_pred_examples": num_empty_pred,
            "avg_pred_spans_per_example": avg_pred_spans,
            "debug_k": debug_k,
            "debug_chars": debug_chars,
            "iou_threshold": iou_threshold,
        },
        "debug_examples": debug_rows,
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

    eval_split = run_cfg.get("eval_split", "eval")
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

    # Debug settings for eval
    debug_k = int(run_cfg.get("eval_debug_k", 25))
    debug_chars = int(run_cfg.get("eval_debug_chars", 600))

    # IoU threshold for token overlap matching
    iou_threshold = float(run_cfg.get("iou_threshold", 0.5))

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
    print(f"  eval debug: eval_debug_k={debug_k}, eval_debug_chars={debug_chars}")
    print(f"  eval token-IoU threshold: iou_threshold={iou_threshold}")

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
                category_label=args_cli.category,
                gold_key=completion_key,   # IMPORTANT: use the configured completion key
                use_label=True,
                debug_k=debug_k,
                debug_chars=debug_chars,
                iou_threshold=iou_threshold,
            )

            print("[EVAL] pairwise_micro:", eval_metrics.get("pairwise_micro"))
            print("[EVAL] model_vs_gold_micro:", eval_metrics.get("model_vs_gold_micro"))
            print("[EVAL] debug_summary:", eval_metrics.get("debug_summary"))

            eval_out_path = os.path.join(out_dir, "eval_pairwise_f1.json")
            with open(eval_out_path, "w", encoding="utf-8") as f:
                json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
            print(f"[EVAL] Wrote eval metrics to: {eval_out_path}")

            # Also write a lightweight debug-only file for easy inspection
            debug_out_path = os.path.join(out_dir, "eval_predictions_debug.json")
            with open(debug_out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "eval_path": eval_path,
                        "category": args_cli.category,
                        "debug_summary": eval_metrics.get("debug_summary", {}),
                        "debug_examples": eval_metrics.get("debug_examples", []),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"[EVAL] Wrote predictions debug to: {debug_out_path}")

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
