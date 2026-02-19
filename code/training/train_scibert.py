import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score


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
                preds.append(id2label[int(p)])
                labs.append(id2label[int(l)])
        true_preds.append(preds)
        true_labels.append(labs)

    return f1_score(true_labels, true_preds)


def flatten_valid_tokens(predictions, labels):
    flat_preds = []
    flat_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                flat_preds.append(int(p))
                flat_labels.append(int(l))

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
# Span helpers (word-level)
# ----------------------------
def _word_level_pred_ids_from_aligned(pred_ids_seq: np.ndarray, aligned_label_ids_seq: np.ndarray) -> List[int]:
    """
    Convert wordpiece-level predictions into word-level predictions by taking the prediction
    at positions where aligned labels != -100 (these correspond to first wordpiece of each word).
    Returns list[int] length == number of words in the original example.
    """
    out = []
    for p, l in zip(pred_ids_seq.tolist(), aligned_label_ids_seq.tolist()):
        if l != -100:
            out.append(int(p))
    return out


def _bio_spans_from_word_labels(tokens: List[str], word_labels: List[str]) -> List[Dict[str, Any]]:
    """
    Extract spans from BIO word-level label strings.
    Returns spans with token indices [start_token, end_token) and text joined by spaces.
    """
    spans = []
    i = 0
    n = min(len(tokens), len(word_labels))

    def normalize(lbl: str) -> str:
        return lbl.strip() if isinstance(lbl, str) else "O"

    while i < n:
        lab = normalize(word_labels[i])
        if lab == "O":
            i += 1
            continue

        if lab.startswith("B-"):
            ent = lab[2:]
            start = i
            i += 1
            while i < n and normalize(word_labels[i]) == f"I-{ent}":
                i += 1
            end = i
            spans.append(
                {
                    "label": ent,
                    "start_token": start,
                    "end_token": end,
                    "text": " ".join(tokens[start:end]),
                }
            )
            continue

        if lab.startswith("I-"):
            ent = lab[2:]
            start = i
            i += 1
            while i < n and normalize(word_labels[i]) == f"I-{ent}":
                i += 1
            end = i
            spans.append(
                {
                    "label": ent,
                    "start_token": start,
                    "end_token": end,
                    "text": " ".join(tokens[start:end]),
                    "note": "started_with_I",
                }
            )
            continue

        i += 1

    return spans


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
    dev_path = cat_cfg["eval_path"]
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
    print("  mean:", sum(lengths) / len(lengths))
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
    print("  mean:", sum(train_lengths) / len(train_lengths))
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
        load_best_model_at_end=False,  # removed best-model selection per your preference
        report_to="none",
        fp16=bool(defaults.get("fp16", False)),
        bf16=bool(defaults.get("bf16", False)),
        seed=seed,
    )

    # Use padding in collator to avoid batching length mismatch errors
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

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

    # -------- All-token metrics (includes 'O') --------
    cm = confusion_matrix(flat_labels, flat_preds)
    cm_dict = {"labels": label_list, "matrix": cm.tolist()}

    alpha = krippendorff_alpha_nominal(flat_labels, flat_preds, num_labels=len(label_list))

    p1, r1, f1_1, _ = precision_recall_fscore_support(flat_labels, flat_preds, average="micro", zero_division=0)
    p2, r2, f1_2, _ = precision_recall_fscore_support(flat_preds, flat_labels, average="micro", zero_division=0)

    if flat_labels.size == 0:
        kappa_all = 0.0
    else:
        kappa_all = float(cohen_kappa_score(flat_labels, flat_preds, labels=list(range(len(label_list)))))

    # Pairwise micro-F1 (kept)
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

    # -------- Entity-only token metrics (ignore gold 'O') --------
    def filter_out_gold_O(flat_preds_arr: np.ndarray, flat_labels_arr: np.ndarray, id2label_map: Dict[int, str]):
        if flat_labels_arr.size == 0:
            return flat_preds_arr, flat_labels_arr
        keep_mask = np.array([id2label_map[int(l)] != "O" for l in flat_labels_arr], dtype=bool)
        return flat_preds_arr[keep_mask], flat_labels_arr[keep_mask]

    flat_preds_ent, flat_labels_ent = filter_out_gold_O(flat_preds, flat_labels, id2label)

    if flat_labels_ent.size == 0:
        # no entity tokens present in gold -> zeros
        cm_ent = np.zeros((len(label_list), len(label_list)), dtype=int)
        cm_ent_dict = {"labels": label_list, "matrix": cm_ent.tolist()}
        alpha_ent = 0.0
        kappa_ent = 0.0
        p1_ent = r1_ent = f1_1_ent = 0.0
        p2_ent = r2_ent = f1_2_ent = 0.0
        micro_f1_ent = 0.0
    else:
        cm_ent = confusion_matrix(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list))))
        cm_ent_dict = {"labels": label_list, "matrix": cm_ent.tolist()}

        alpha_ent = krippendorff_alpha_nominal(flat_labels_ent, flat_preds_ent, num_labels=len(label_list))

        p1_ent, r1_ent, f1_1_ent, _ = precision_recall_fscore_support(flat_labels_ent, flat_preds_ent, average="micro", zero_division=0)
        p2_ent, r2_ent, f1_2_ent, _ = precision_recall_fscore_support(flat_preds_ent, flat_labels_ent, average="micro", zero_division=0)

        if flat_labels_ent.size == 0:
            kappa_ent = 0.0
        else:
            kappa_ent = float(cohen_kappa_score(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list)))))

        tp_ent = int(np.sum(flat_preds_ent == flat_labels_ent))
        fp_ent = int(np.sum(flat_preds_ent != flat_labels_ent))
        fn_ent = fp_ent
        total_tp_ent = tp_ent * 2
        total_fp_ent = fp_ent * 2
        total_fn_ent = fn_ent * 2
        micro_precision_ent = total_tp_ent / (total_tp_ent + total_fp_ent) if (total_tp_ent + total_fp_ent) else 0.0
        micro_recall_ent = total_tp_ent / (total_tp_ent + total_fn_ent) if (total_tp_ent + total_fn_ent) else 0.0
        micro_f1_ent = (
            2 * micro_precision_ent * micro_recall_ent / (micro_precision_ent + micro_recall_ent)
            if (micro_precision_ent + micro_recall_ent)
            else 0.0
        )

    # ---- Build per-example predicted spans for eval set ----
    pred_spans_by_example: List[Dict[str, Any]] = []
    for raw_ex, pred_seq, lab_seq in zip(eval_dataset, predictions, labels):
        # raw_ex here is tokenized example (contains wordpiece fields). We need original tokens/labels.
        # Use eval_dataset's underlying original if available; fall back to tokenized's word-level mapping.
        # We have eval_dataset (tokenized) and eval_dataset was created from raw eval JSON earlier (same order).
        # So use eval_dataset.dataset if available: but simplest: load eval raw earlier and reuse original examples.
        pass  # placeholder

    # To reliably pair tokenized outputs with original raw examples we will reload the raw eval file
    # (the original ordering was preserved).
    eval_raw_path = eval_path
    eval_dataset_raw = load_ner_json(eval_raw_path)

    pred_spans_by_example = []
    for raw_ex, pred_seq, lab_seq in zip(eval_dataset_raw, predictions, labels):
        tokens_words = raw_ex["tokens"]
        gold_word_labels = raw_ex["labels"]

        pred_word_ids = _word_level_pred_ids_from_aligned(pred_seq, lab_seq)
        pred_word_labels = [id2label[i] for i in pred_word_ids]

        pred_spans = _bio_spans_from_word_labels(tokens_words, pred_word_labels)
        gold_spans = _bio_spans_from_word_labels(tokens_words, gold_word_labels)

        pred_spans_by_example.append(
            {
                "tokens": tokens_words,
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
            }
        )

    pred_spans_path = output_dir / "eval_predicted_spans.json"
    with open(pred_spans_path, "w", encoding="utf-8") as f:
        json.dump(pred_spans_by_example, f, indent=2, ensure_ascii=False)

    results = {
        "base_model": base_model,
        "seqeval_span_f1": float(seqeval_f1),
        "krippendorff_alpha_nominal": float(alpha),
        "cohen_kappa_token_level_all_tokens": float(kappa_all),
        "pairwise_f1": {
            "gold_to_pred": {"precision": float(p1), "recall": float(r1), "f1": float(f1_1)},
            "pred_to_gold": {"precision": float(p2), "recall": float(r2), "f1": float(f1_2)},
            "pairwise_micro_f1": float(micro_f1),
        },
        "confusion_matrix": cm_dict,
        "entity_only_metrics": {
            "krippendorff_alpha_nominal": float(alpha_ent),
            "cohen_kappa_token_level": float(kappa_ent),
            "pairwise_f1": {
                "gold_to_pred": {"precision": float(p1_ent), "recall": float(r1_ent), "f1": float(f1_1_ent)},
                "pred_to_gold": {"precision": float(p2_ent), "recall": float(r2_ent), "f1": float(f1_2_ent)},
                "pairwise_micro_f1": float(micro_f1_ent),
            },
            "confusion_matrix": cm_ent_dict,
        },
        "artifacts": {
            "eval_predicted_spans_path": str(pred_spans_path),
        },
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
