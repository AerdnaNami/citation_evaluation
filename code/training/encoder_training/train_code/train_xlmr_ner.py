import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

from seqeval.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ner_json(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def tokenize_and_align_labels(example, tokenizer, label2id):
    # Enforce a safe max_length (important for BERT-like models; also ok for XLM-R)
    max_len = getattr(tokenizer, "model_max_length", 512)
    if max_len is None or max_len > 512:  # clamp to 512 to be safe in most NER setups
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


def compute_seqeval_f1(predictions, labels):
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
                flat_preds.append(int(p))
                flat_labels.append(int(l))

    return np.array(flat_preds), np.array(flat_labels)


def filter_out_gold_O(flat_preds: np.ndarray, flat_labels: np.ndarray, id2label: dict):
    """
    Keep only tokens where the GOLD label is NOT 'O'.
    This evaluates only entity tokens (B-*/I-*), ignoring gold negatives.
    """
    if flat_labels.size == 0:
        return flat_preds, flat_labels

    keep_mask = []
    for l in flat_labels.tolist():
        keep_mask.append(id2label[int(l)] != "O")

    keep_mask = np.array(keep_mask, dtype=bool)
    return flat_preds[keep_mask], flat_labels[keep_mask]


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
# Token-overlap (entity-only) F1
# ----------------------------
def token_overlap_entity_only_from_flat(flat_preds: np.ndarray, flat_labels: np.ndarray, id2label: Dict[int, str]) -> Dict[str, Any]:
    """
    Compute token-overlap style F1 but *only* over gold entity tokens (B-*/I-*).
    Gold 'O' tokens are ignored entirely.

    Returns micro/macro precision/recall/F1 and supports.
    """
    if flat_labels.size == 0:
        return {
            "scored_labels": [],
            "precision_micro": 0.0,
            "recall_micro": 0.0,
            "f1_micro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "num_tokens_scored": 0,
            "support_positive_gold": 0,
            "support_positive_pred": 0,
        }

    # keep mask where gold is not 'O'
    keep_mask = np.array([id2label[int(l)] != "O" for l in flat_labels], dtype=bool)
    if not keep_mask.any():
        return {
            "scored_labels": [],
            "precision_micro": 0.0,
            "recall_micro": 0.0,
            "f1_micro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "num_tokens_scored": 0,
            "support_positive_gold": 0,
            "support_positive_pred": 0,
            "note": "no gold entity (B/I) tokens found",
        }

    gold = flat_labels[keep_mask].astype(np.int64)
    pred = flat_preds[keep_mask].astype(np.int64)

    # entity label ids present in gold (we only score over these)
    entity_label_ids = sorted(list(set(gold.tolist())))

    # micro & macro over entity label ids
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        gold, pred, labels=entity_label_ids, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        gold, pred, labels=entity_label_ids, average="macro", zero_division=0
    )

    return {
        "scored_labels": [id2label[i] for i in entity_label_ids],
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f1_micro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "num_tokens_scored": int(gold.size),
        "support_positive_gold": int((gold != -1).sum()),  # all scored tokens
        "support_positive_pred": int((pred != -1).sum()),
    }


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
            spans.append({
                "label": ent,
                "start_token": start,
                "end_token": end,
                "text": " ".join(tokens[start:end]),
            })
            continue

        # If it starts with I- without a preceding B-, treat as a new span (common model error)
        if lab.startswith("I-"):
            ent = lab[2:]
            start = i
            i += 1
            while i < n and normalize(word_labels[i]) == f"I-{ent}":
                i += 1
            end = i
            spans.append({
                "label": ent,
                "start_token": start,
                "end_token": end,
                "text": " ".join(tokens[start:end]),
                "note": "started_with_I",
            })
            continue

        # unknown label form
        i += 1

    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    defaults = cfg.get("defaults", {})
    base_model = cfg.get("base_model", "xlm-roberta-base")

    cat_cfg = next(c for c in cfg["categories"] if c["name"] == args.category)

    train_path = cat_cfg["train_path"]
    dev_path = cat_cfg["eval_path"]
    eval_path = cat_cfg.get("eval_path", dev_path)

    output_root = Path(cfg.get("output_root", "./runs"))
    run_name = cat_cfg.get("run_name", args.category)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load RAW datasets
    train_dataset_raw = load_ner_json(train_path)
    dev_dataset_raw = load_ner_json(dev_path)
    eval_dataset_raw = load_ner_json(eval_path)

    # Build label set
    label_list = sorted(
        list({label for example in train_dataset_raw for label in example["labels"]})
    )

    global label2id, id2label
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Tokenize datasets
    train_dataset = train_dataset_raw.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=False,
    )
    dev_dataset = dev_dataset_raw.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=False,
    )
    eval_dataset = eval_dataset_raw.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=False,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

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
        warmup_ratio=float(defaults.get("warmup_ratio", 0.1)),
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Label list:", label_list)

    trainer.train()

    # ----------------------------
    # FINAL EVALUATION ON EVAL SET
    # ----------------------------
    predictions, labels, _ = trainer.predict(eval_dataset)
    predictions = np.argmax(predictions, axis=2)

    # Span-based F1 (seqeval)
    seqeval_f1 = compute_seqeval_f1(predictions, labels)

    # Token-level arrays (valid tokens only, ignoring -100)
    flat_preds, flat_labels = flatten_valid_tokens(predictions, labels)

    # -------- Token metrics (ALL TOKENS, includes 'O') --------
    cm_all = confusion_matrix(flat_labels, flat_preds, labels=list(range(len(label_list))))
    cm_all_dict = {"labels": label_list, "matrix": cm_all.tolist()}

    alpha_all = krippendorff_alpha_nominal(flat_labels, flat_preds, num_labels=len(label_list))

    if flat_labels.size == 0:
        kappa_all = 0.0
    else:
        kappa_all = float(cohen_kappa_score(flat_labels, flat_preds, labels=list(range(len(label_list)))))

    p_micro_all, r_micro_all, f1_micro_all, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average="micro", zero_division=0
    )
    p_macro_all, r_macro_all, f1_macro_all, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average="macro", zero_division=0
    )

    # -------- Token metrics (ENTITY ONLY, ignore gold 'O') --------
    flat_preds_ent, flat_labels_ent = filter_out_gold_O(flat_preds, flat_labels, id2label)

    cm_ent = confusion_matrix(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list))))
    cm_ent_dict = {"labels": label_list, "matrix": cm_ent.tolist()}

    alpha_ent = krippendorff_alpha_nominal(flat_labels_ent, flat_preds_ent, num_labels=len(label_list))

    if flat_labels_ent.size == 0:
        kappa_ent = 0.0
    else:
        kappa_ent = float(cohen_kappa_score(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list)))))

    p_micro_ent, r_micro_ent, f1_micro_ent, _ = precision_recall_fscore_support(
        flat_labels_ent, flat_preds_ent, average="micro", zero_division=0
    )
    p_macro_ent, r_macro_ent, f1_macro_ent, _ = precision_recall_fscore_support(
        flat_labels_ent, flat_preds_ent, average="macro", zero_division=0
    )

    # ---- Build per-example predicted spans for eval set ----
    pred_spans_by_example: List[Dict[str, Any]] = []
    for raw_ex, pred_seq, lab_seq in zip(eval_dataset_raw, predictions, labels):
        tokens_words = raw_ex["tokens"]
        gold_word_labels = raw_ex["labels"]

        pred_word_ids = _word_level_pred_ids_from_aligned(pred_seq, lab_seq)
        pred_word_labels = [id2label[i] for i in pred_word_ids]

        pred_spans = _bio_spans_from_word_labels(tokens_words, pred_word_labels)
        gold_spans = _bio_spans_from_word_labels(tokens_words, gold_word_labels)

        pred_spans_by_example.append({
            "tokens": tokens_words,
            "gold_spans": gold_spans,
            "pred_spans": pred_spans,
        })

    pred_spans_path = output_dir / "eval_predicted_spans.json"
    with open(pred_spans_path, "w", encoding="utf-8") as f:
        json.dump(pred_spans_by_example, f, indent=2, ensure_ascii=False)

    # -------- Token-overlap entity-only F1 (NEW) --------
    token_overlap_entity_only = token_overlap_entity_only_from_flat(flat_preds, flat_labels, id2label)

    results = {
        "seqeval_span_f1": float(seqeval_f1),

        # Token overlap (entity-only)
        "token_overlap_entity_only": token_overlap_entity_only,

        # Original token metrics (includes O)
        "token_metrics_all_tokens": {
            "krippendorff_alpha_nominal": float(alpha_all),
            "cohen_kappa_token_level": float(kappa_all),
            "token_prf": {
                "micro": {"precision": float(p_micro_all), "recall": float(r_micro_all), "f1": float(f1_micro_all)},
                "macro": {"precision": float(p_macro_all), "recall": float(r_macro_all), "f1": float(f1_macro_all)},
            },
            "confusion_matrix": cm_all_dict,
            "num_tokens_scored": int(flat_labels.size),
        },

        # Entity-only token metrics (ignores gold O)
        "token_metrics_entity_only": {
            "krippendorff_alpha_nominal": float(alpha_ent),
            "cohen_kappa_token_level": float(kappa_ent),
            "token_prf": {
                "micro": {"precision": float(p_micro_ent), "recall": float(r_micro_ent), "f1": float(f1_micro_ent)},
                "macro": {"precision": float(p_macro_ent), "recall": float(r_macro_ent), "f1": float(f1_macro_ent)},
            },
            "confusion_matrix": cm_ent_dict,
            "num_tokens_scored": int(flat_labels_ent.size),
        },

        "artifacts": {
            "eval_predicted_spans_path": str(pred_spans_path),
        },
    }

    print("\n-----FINAL EVAL RESULTS-----")
    print(json.dumps(results, indent=2))

    with open(output_dir / "final_eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()