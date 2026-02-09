import argparse
import json
import os
import socket
import platform
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

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
# Dataset
# -------------------------
class PromptCompletionDataset(Dataset):
    def __init__(self, path: str, prompt_key: str = "prompt", completion_key: str = "completion"):
        self.rows: List[Dict[str, Any]] = []
        self.prompt_key = prompt_key
        self.completion_key = completion_key

        with open(path, "r", encoding="utf-8-sig") as f:
            text = f.read().strip()
        if not text:
            raise ValueError(f"{path} is empty")

        if text[0] in "[{":
            try:
                obj = json.loads(text)
                self.rows = self._normalize_json(obj, path)
            except json.JSONDecodeError:
                self.rows = self._load_jsonl(path)
        else:
            self.rows = self._load_jsonl(path)

        self._validate_rows(path)

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        rows = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    preview = line[:200].replace("\n", "\\n")
                    raise ValueError(f"Bad JSON on line {lineno} in {path}: {e}\nPreview: {preview}") from e
        return rows

    def _normalize_json(self, obj: Any, path: str) -> List[Dict[str, Any]]:
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("data", "rows", "examples", "items"):
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
        raise ValueError(f"{path} JSON must be a list of examples (or dict containing one). Got: {type(obj)}")

    def _validate_rows(self, path: str) -> None:
        cleaned = []
        bad = 0
        for i, r in enumerate(self.rows):
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
        # Always return canonical keys for the collator
        return {"prompt": r[self.prompt_key], "completion": r[self.completion_key]}


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

            # If completion tokenizes to nothing (e.g., whitespace), force EOS so we have supervision
            if len(comp_ids) == 0:
                comp_ids = [eos_id]

            # If completion alone exceeds max_length, keep the tail (or head) so we still train
            if len(comp_ids) > self.max_length:
                comp_ids = comp_ids[-self.max_length:]

            # Truncate prompt to make room for completion
            max_prompt_len = max(0, self.max_length - len(comp_ids))
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + comp_ids
            attention_mask = [1] * len(input_ids)

            labels = input_ids.copy()
            labels[:len(prompt_ids)] = [IGNORE_INDEX] * len(prompt_ids)

            # Final safety: ensure at least one label token
            if all(x == IGNORE_INDEX for x in labels):
                # shouldn't happen, but guard anyway
                labels[-1] = input_ids[-1]

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX
        )

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


def write_save_reason_json(output_dir: str, reason: str, entry: dict) -> None:
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
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--category", required=True, help="Category name from config.json")
    args_cli = parser.parse_args()

    # A100-friendly defaults for env (config still controls actual bf16/fp16 flags passed to TrainingArguments)
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

    # Paths
    train_path = run_cfg["train_path"]
    dev_path = run_cfg.get("dev_path")
    eval_path = run_cfg.get("eval_path")

    eval_split = run_cfg.get("eval_split", "dev")  # "dev" or "eval"
    eval_path_for_training = dev_path if eval_split == "dev" else eval_path

    # Output dir: same folder as data (category folder), run_name subdir
    data_dir = Path(train_path).resolve().parent
    run_name = run_cfg.get("run_name", args_cli.category)
    out_dir = str(data_dir / run_name)

    # Model + training params (from config)
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

    print("Settings (from config):")
    print(f"  Category: {args_cli.category}")
    print(f"  Model: {model_name}")
    print(f"  Train data: {train_path}")
    print(f"  Eval data: {eval_path_for_training}")
    print(f"  Output dir: {out_dir}")
    print(f"  max_length: {max_length}")
    print(f"  train_bs: {per_device_train_batch_size}, eval_bs: {per_device_eval_batch_size}, grad_accum: {gradient_accumulation_steps}")
    print(f"  lr: {learning_rate}, epochs: {num_train_epochs}, warmup_ratio: {warmup_ratio}")
    print(f"  bf16: {bf16}, fp16: {fp16}")
    print(f"  LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"  flash_attn: {use_flash_attn}, torch_compile: {torch_compile}")
    print(f"  num_workers: {num_workers}, max_grad_norm: {max_grad_norm}")

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
    train_ds = PromptCompletionDataset(train_path)
    eval_ds = PromptCompletionDataset(eval_path_for_training) if eval_path_for_training and os.path.exists(eval_path_for_training) else None
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

        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=save_steps if eval_ds is not None else None,

        report_to="none",
        group_by_length=False,
    )

    # --- SANITY CHECK: inspect one batch end-to-end ---
    sample = [train_ds[0]]
    batch = collator(sample)

    num_label_tokens = (batch["labels"] != IGNORE_INDEX).sum().item()
    seq_len = batch["input_ids"].shape[1]
    print(f"[SANITY] seq_len={seq_len}, num_label_tokens={num_label_tokens}")

    # Must have at least 1 supervised token
    if num_label_tokens == 0:
        raise RuntimeError(
            "[SANITY FAIL] num_label_tokens == 0. Your collator is masking everything "
            "(likely because completion is empty or being truncated away)."
        )

    # Run a single forward pass to confirm non-zero loss + that compute is happening
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            labels=batch["labels"].to(model.device),
        )
    print(f"[SANITY] forward loss={out.loss.item()}")
    if out.loss.item() == 0.0:
        raise RuntimeError("[SANITY FAIL] forward loss is 0.0 even though label tokens exist.")
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

    entry = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "reason": save_reason,
        "category": args_cli.category,
        "model_name": model_name,
        "train_path": train_path,
        "eval_path": eval_path_for_training if eval_ds is not None else None,
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
    }
    write_save_reason_json(out_dir, reason=save_reason, entry=entry)

    print(f"Saved (LoRA adapters + tokenizer) to: {out_dir}")
    print(f"Wrote save reasons to: {os.path.join(out_dir, 'save_reasons.json')}")


if __name__ == "__main__":
    main()
