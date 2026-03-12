import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score


# ----------------------------
# Config + Data
# ----------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list of examples")

    cleaned = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            continue
        toks = ex.get("tokens")
        labs = ex.get("labels")
        if not isinstance(toks, list) or not isinstance(labs, list):
            raise ValueError(f"{path} example {i}: missing tokens/labels lists")
        if len(toks) != len(labs):
            raise ValueError(
                f"{path} example {i}: len(tokens)={len(toks)} != len(labels)={len(labs)}"
            )

        toks = [str(t) for t in toks]
        labs = [str(l) for l in labs]

        cleaned.append(
            {
                "tokens": toks,
                "labels": labs,
                "text": str(ex.get("text", " ".join(toks))),
                "filename": str(ex.get("filename", "")),
            }
        )

    return Dataset.from_list(cleaned)


# ----------------------------
# BIO helpers
# ----------------------------
def _ent_type(lbl: str) -> str:
    lbl = (lbl or "O").strip()
    if "-" in lbl:
        return lbl.split("-", 1)[1].strip()
    return ""


def bio_fix_sequence(labels: List[str]) -> List[str]:
    """Repair invalid BIO sequences (I-* cannot start or switch type without B-*)."""
    fixed = []
    prev = "O"
    for lbl in labels:
        lbl = (lbl or "O").strip()

        if lbl.startswith("I-"):
            x = _ent_type(lbl)
            if not (prev == f"B-{x}" or prev == f"I-{x}"):
                lbl = f"B-{x}"

        if lbl.startswith("I-") and prev.startswith(("B-", "I-")):
            x = _ent_type(lbl)
            if _ent_type(prev) != x:
                lbl = f"B-{x}"

        fixed.append(lbl)
        prev = lbl
    return fixed


def continuation_label(word_label: str) -> str:
    word_label = (word_label or "O").strip()
    if word_label.startswith("B-"):
        return "I-" + word_label[2:]
    if word_label.startswith("I-"):
        return word_label
    return "O"


def to_binary_entity_labels(word_labels: List[str]) -> List[str]:
    """Convert BIO labels to binary: ENTITY vs O."""
    out = []
    for l in word_labels:
        s = (l or "O").strip()
        out.append("O" if s == "O" else "ENTITY")
    return out


# ----------------------------
# Tokenization + alignment
# ----------------------------
def tokenize_and_align(
    example: Dict[str, Any],
    tokenizer,
    label2id: Dict[str, int],
    label_all_subtokens: bool = True,
) -> Dict[str, Any]:
    max_len = getattr(tokenizer, "model_max_length", 512) or 512
    max_len = min(int(max_len), 512)

    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_len,
        padding=False,
    )

    word_ids = tokenized.word_ids()
    aligned = []
    prev_word = None

    for widx in word_ids:
        if widx is None:
            aligned.append(-100)
            continue

        wlab = example["labels"][widx]  # already binary at word-level
        if widx != prev_word:
            aligned.append(label2id[wlab])
        else:
            if label_all_subtokens:
                # for binary, continuation is same label
                aligned.append(label2id[wlab])
            else:
                aligned.append(-100)

        prev_word = widx

    tokenized["labels"] = aligned
    return tokenized


# ----------------------------
# Eval helpers (binary)
# ----------------------------
def flatten_valid(pred_ids: np.ndarray, label_ids: np.ndarray):
    flat_p, flat_l = [], []
    for p_seq, l_seq in zip(pred_ids, label_ids):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            flat_p.append(int(p))
            flat_l.append(int(l))
    return np.array(flat_p, dtype=np.int64), np.array(flat_l, dtype=np.int64)


def exact_match_seq(pred_ids: np.ndarray, label_ids: np.ndarray) -> float:
    """Sequence exact match over scored positions only (label != -100)."""
    exact = 0
    total = 0
    for p_seq, l_seq in zip(pred_ids, label_ids):
        mask = (l_seq != -100)
        if int(mask.sum()) == 0:
            continue
        total += 1
        if np.all(p_seq[mask] == l_seq[mask]):
            exact += 1
    return float(exact / total) if total else 0.0


def token_overlap_f1_binary_from_ids(flat_pred: np.ndarray, flat_gold: np.ndarray, entity_id: int) -> Dict[str, Any]:
    """
    Binary token overlap F1:
      positive = ENTITY (entity_id)
      negative = O
    """
    if flat_gold.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    gold_pos = (flat_gold == entity_id).astype(np.int64)
    pred_pos = (flat_pred == entity_id).astype(np.int64)

    tp = int(((pred_pos == 1) & (gold_pos == 1)).sum())
    fp = int(((pred_pos == 1) & (gold_pos == 0)).sum())
    fn = int(((pred_pos == 0) & (gold_pos == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn}


def compute_metrics_fn_binary(label2id: Dict[str, int]):
    """
    Trainer-time metrics on dev: binary PRF for ENTITY.
    """
    entity_id = int(label2id["ENTITY"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        flat_p, flat_l = flatten_valid(preds, labels)
        if flat_l.size == 0:
            return {"entity_precision": 0.0, "entity_recall": 0.0, "entity_f1": 0.0, "token_accuracy": 0.0}

        # PRF on positive class ENTITY
        gold_pos = (flat_l == entity_id).astype(np.int64)
        pred_pos = (flat_p == entity_id).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(gold_pos, pred_pos, average="binary", zero_division=0)

        acc = float((flat_p == flat_l).mean())

        return {
            "entity_precision": float(p),
            "entity_recall": float(r),
            "entity_f1": float(f1),
            "token_accuracy": float(acc),
        }

    return compute_metrics


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--run_name", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    base_model = cfg.get("base_model", "xlm-roberta-base")

    # IMPORTANT: select by BOTH (name, run_name) so short/long both work
    cat_cfg = next(
        c
        for c in cfg["categories"]
        if c["name"] == args.category and c.get("run_name") == args.run_name
    )

    train_path = cat_cfg["train_path"]
    eval_path = cat_cfg["eval_path"]
    dev_path = cat_cfg.get("dev_path") or cat_cfg.get("eval_path")

    output_root = Path(cfg.get("output_root", "./runs"))
    output_dir = output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_raw = load_json_dataset(train_path)
    dev_raw = load_json_dataset(dev_path)
    eval_raw = load_json_dataset(eval_path)

    # Repair BIO in gold then convert to binary at WORD level
    def fix_and_binary(ex):
        fixed = bio_fix_sequence(ex["labels"])
        return {**ex, "labels": to_binary_entity_labels(fixed)}

    train_raw = train_raw.map(fix_and_binary)
    dev_raw = dev_raw.map(fix_and_binary)
    eval_raw = eval_raw.map(fix_and_binary)

    # Binary label space
    label_list = ["O", "ENTITY"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    print("Base model:", base_model)
    print("Output dir:", str(output_dir))
    print("Labels:", label_list)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    # As requested: NOT configurable; always True
    label_all_subtokens = True

    train_ds = train_raw.map(lambda ex: tokenize_and_align(ex, tokenizer, label2id, label_all_subtokens), batched=False)
    dev_ds = dev_raw.map(lambda ex: tokenize_and_align(ex, tokenizer, label2id, label_all_subtokens), batched=False)
    eval_ds = eval_raw.map(lambda ex: tokenize_and_align(ex, tokenizer, label2id, label_all_subtokens), batched=False)

    # Remove non-tensor columns so the collator doesn't choke on strings/lists
    keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in train_ds.column_names:
        keep.add("token_type_ids")

    def prune(ds: Dataset, name: str) -> Dataset:
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            print(f"[{name}] removing columns:", drop)
            ds = ds.remove_columns(drop)
        return ds

    train_ds = prune(train_ds, "train")
    dev_ds = prune(dev_ds, "dev")
    eval_ds = prune(eval_ds, "eval")

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
        logging_steps=int(defaults.get("logging_steps", 200)),
        learning_rate=float(defaults.get("learning_rate", 3e-5)),
        per_device_train_batch_size=int(defaults.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(defaults.get("per_device_eval_batch_size", 16)),
        num_train_epochs=float(defaults.get("num_train_epochs", 3)),
        weight_decay=float(defaults.get("weight_decay", 0.01)),
        warmup_ratio=float(defaults.get("warmup_ratio", 0.1)),
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn_binary(label2id),
    )

    trainer.train()

    # ----------------------------
    # FINAL EVALUATION ON EVAL SET (requested metrics)
    # ----------------------------
    pred = trainer.predict(eval_ds)
    logits = pred.predictions
    labels = pred.label_ids
    pred_ids = np.argmax(logits, axis=-1)

    flat_p, flat_l = flatten_valid(pred_ids, labels)

    # Confusion matrix (token-level, scored positions only)
    if flat_l.size:
        cm = confusion_matrix(flat_l, flat_p, labels=list(range(len(label_list))))
        cm_payload = {"labels": label_list, "matrix": cm.tolist()}
    else:
        cm_payload = {"labels": label_list, "matrix": []}

    cm_path = output_dir / "confusion_matrix.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm_payload, f, indent=2, ensure_ascii=False)

    # Token overlap F1 (ENTITY vs O)
    entity_id = int(label2id["ENTITY"])
    overlap = token_overlap_f1_binary_from_ids(flat_p, flat_l, entity_id=entity_id)

    # Exact match
    token_exact_match = float((flat_p == flat_l).mean()) if flat_l.size else 0.0
    seq_exact_match = exact_match_seq(pred_ids, labels)

    # Cohen's kappa (token-level, scored positions only)
    kappa = float(cohen_kappa_score(flat_l, flat_p, labels=list(range(len(label_list))))) if flat_l.size else 0.0

    final_report = {
        "run_name": args.run_name,
        "category": args.category,
        "base_model": base_model,
        "num_eval_sequences": int(labels.shape[0]),
        "num_eval_tokens_scored": int(flat_l.size),
        "token_overlap_f1_entity_vs_O": overlap,
        "exact_match": {
            "token_accuracy": float(token_exact_match),
            "sequence_exact_match": float(seq_exact_match),
        },
        "cohens_kappa_token_level": float(kappa),
        "confusion_matrix_path": str(cm_path),
    }

    report_path = output_dir / "final_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\n-----FINAL EVAL REPORT (EVAL SET)-----")
    print(json.dumps(final_report, indent=2))

    # Save model + tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\nDone. Saved to:", str(output_dir))


if __name__ == "__main__":
    main()