import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    set_seed,
)

from seqeval.metrics import f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# ----------------------------
# Config Loader
# ----------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Load JSON NER data
# Expected format per example:
#   {"tokens": [...], "labels": [...]}
# ----------------------------
def load_ner_json(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


# ----------------------------
# Tokenize and align labels
# Uses BIO-style assumption: label for first wordpiece, -100 for rest
# ----------------------------
def tokenize_and_align_labels(example, tokenizer, label2id):
    max_len = getattr(tokenizer, "model_max_length", 512)
    # Some tokenizers set model_max_length to a huge int; clamp to BERT's real limit if needed.
    if max_len is None or max_len > 512:
        max_len = 512

    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    
    word_ids = tokenized.word_ids()
    labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(label2id[example["labels"][word_idx]])
        else:
            labels.append(-100)
        previous_word_idx = word_idx

    tokenized["labels"] = labels
    return tokenized


# ----------------------------
# Metric Helpers
# ----------------------------
def compute_seqeval_f1(predictions, labels, id2label):
    true_preds = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        preds = []
        labs = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                preds.append(id2label[p])
                labs.append(id2label[l])
        true_preds.append(preds)
        true_labels.append(labs)

    return f1_score(true_labels, true_preds)


def flatten_valid_tokens(predictions, labels):
    flat_preds = []
    flat_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                flat_preds.append(p)
                flat_labels.append(l)

    return np.array(flat_preds), np.array(flat_labels)


def krippendorff_alpha_nominal(rater1: np.ndarray, rater2: np.ndarray, num_labels: int) -> float:
    """
    Krippendorff's alpha for nominal data for 2 raters with complete data.
    rater1/rater2: arrays of label ids in [0, num_labels-1]
    """
    if rater1.shape[0] == 0:
        return 0.0

    o = np.zeros((num_labels, num_labels), dtype=np.float64)

    for a, b in zip(rater1, rater2):
        if a == b:
            o[a, a] += 2.0
        else:
            o[a, b] += 1.0
            o[b, a] += 1.0

    N = o.sum()
    if N <= 1:
        return 0.0

    Do = (o.sum() - np.trace(o)) / N

    n = o.sum(axis=0)
    De = (N * N - np.sum(n * n)) / (N * (N - 1))

    if De <= 0:
        return 1.0 if Do == 0 else 0.0

    return float(1.0 - (Do / De))


# ----------------------------
# Main
# ----------------------------
def main():
    def get_tokenized_lengths(dataset, tokenizer):
        lengths = []
        for example in dataset:
            encoded = tokenizer(
                example["tokens"],
                is_split_into_words=True,
                truncation=False,  # IMPORTANT: don't truncate so we see true length
                padding=False,
            )
            lengths.append(len(encoded["input_ids"]))
        return lengths


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})

    base_model = cfg.get("base_model", "allenai/scibert_scivocab_uncased")

    seed = int(defaults.get("seed", 42))
    set_seed(seed)

    cat_cfg = next(c for c in cfg["categories"] if c["name"] == args.category)

    train_path = cat_cfg["train_path"]
    dev_path = cat_cfg["dev_path"]
    eval_path = cat_cfg["eval_path"]

    output_root = Path(cfg.get("output_root", "./runs"))
    run_name = cat_cfg.get("run_name", args.category)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_ner_json(train_path)
    dev_dataset = load_ner_json(dev_path)
    eval_dataset = load_ner_json(eval_path)
    lengths = [len(example["tokens"]) for example in train_dataset]

    print("Raw token stats:")
    print("  max:", max(lengths))
    print("  mean:", sum(lengths)/len(lengths))
    print("  95th percentile:", np.percentile(lengths, 95))

    
    label_set = set()
    for ex in train_dataset:
        for lab in ex["labels"]:
            label_set.add(lab)
    label_list = sorted(list(label_set))

    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    train_dataset = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=False)
    dev_dataset = dev_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=False)
    eval_dataset = eval_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=False)
    train_lengths = get_tokenized_lengths(train_dataset, tokenizer)

    print("Tokenized length stats (no truncation):")
    print("  max:", max(train_lengths))
    print("  mean:", sum(train_lengths)/len(train_lengths))
    print("  95th percentile:", np.percentile(train_lengths, 95))
    
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    print(f"Using base_model={base_model}")
    print(f"num_labels={len(label_list)} | labels={label_list}")
    print(f"output_dir={output_dir}")

    warmup_ratio = defaults.get("warmup_ratio", None)
    warmup_steps = defaults.get("warmup_steps", 0)
    if warmup_ratio is not None:
        warmup_steps = 0
    else:
        warmup_steps = int(warmup_steps)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy=defaults.get("evaluation_strategy", "epoch"),
        save_strategy=defaults.get("save_strategy", "epoch"),
        logging_steps=int(defaults.get("logging_steps", 50)),
        learning_rate=float(defaults.get("learning_rate", 3e-5)),
        per_device_train_batch_size=int(defaults.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(defaults.get("per_device_eval_batch_size", 16)),
        num_train_epochs=float(defaults.get("num_train_epochs", 3)),
        weight_decay=float(defaults.get("weight_decay", 0.01)),
        warmup_steps=warmup_steps if warmup_ratio is None else 0,
        warmup_ratio=float(warmup_ratio) if warmup_ratio is not None else 0.0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=bool(defaults.get("fp16", False)),
        bf16=bool(defaults.get("bf16", False)),
        seed=seed,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    predictions, labels, _ = trainer.predict(eval_dataset)
    predictions = np.argmax(predictions, axis=2)

    seqeval_f1 = compute_seqeval_f1(predictions, labels, id2label)

    flat_preds, flat_labels = flatten_valid_tokens(predictions, labels)

    cm = confusion_matrix(flat_labels, flat_preds)
    cm_dict = {"labels": label_list, "matrix": cm.tolist()}

    alpha = krippendorff_alpha_nominal(flat_labels, flat_preds, num_labels=len(label_list))

    p1, r1, f1_1, _ = precision_recall_fscore_support(flat_labels, flat_preds, average="micro", zero_division=0)
    p2, r2, f1_2, _ = precision_recall_fscore_support(flat_preds, flat_labels, average="micro", zero_division=0)

    tp = int(np.sum(flat_preds == flat_labels))
    fp = int(np.sum(flat_preds != flat_labels))
    fn = fp

    total_tp = tp * 2
    total_fp = fp * 2
    total_fn = fn * 2

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    results = {
        "base_model": base_model,
        "seqeval_span_f1": float(seqeval_f1),
        "krippendorff_alpha_nominal": float(alpha),
        "pairwise_f1": {
            "gold_to_pred": {"precision": float(p1), "recall": float(r1), "f1": float(f1_1)},
            "pred_to_gold": {"precision": float(p2), "recall": float(r2), "f1": float(f1_2)},
            "pairwise_micro_f1": float(micro_f1),
        },
        "confusion_matrix": cm_dict,
        "seed": seed,
    }

    print("\n-----FINAL EVAL RESULTS-----")
    print(json.dumps(results, indent=2))

    with open(output_dir / "final_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
