import json
import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score


# ----------------------------
# I/O (BIO JSON: [{"tokens":[...],"labels":[...]}])
# ----------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_bio_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list")

    cleaned = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            continue

        toks = ex.get("tokens")
        labs = ex.get("labels")
        if not isinstance(toks, list) or not isinstance(labs, list):
            raise ValueError(f"{path} example {i} missing tokens/labels lists")
        if len(toks) != len(labs):
            raise ValueError(f"{path} example {i}: len(tokens)={len(toks)} != len(labels)={len(labs)}")

        ex = dict(ex)
        ex["tokens"] = [str(t) for t in toks]
        ex["labels"] = [str(l) for l in labs]
        cleaned.append(ex)

    return Dataset.from_list(cleaned)


# ----------------------------
# BIO repair (kept for coverage stability)
# ----------------------------
def _ent_type(lbl):
    lbl = (lbl or "O").strip()
    if "-" in lbl:
        return lbl.split("-", 1)[1].strip()
    return ""


def bio_fix_sequence(labels):
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


# ----------------------------
# Label from BIO coverage
# ----------------------------
def coverage_label_from_bio(word_labels, threshold=0.5):
    if not word_labels:
        return 0
    n = len(word_labels)
    ent = sum(1 for l in word_labels if (l or "O").strip() != "O")
    cov = ent / n if n else 0.0
    return 1 if cov >= threshold else 0


# ----------------------------
# "Sentences together" as token windows over paragraph tokens
# ----------------------------
def build_token_windows(tokens, labels, window_tokens=160, stride_tokens=80):
    out = []
    n = len(tokens)
    if n == 0:
        return out

    w = int(window_tokens)
    s = int(stride_tokens)
    if w <= 0:
        w = n
    if s <= 0:
        s = w

    for start in range(0, n, s):
        end = min(n, start + w)
        win_toks = tokens[start:end]
        win_labs = labels[start:end]
        if not win_toks:
            continue
        out.append((start, end, win_toks, win_labs))
        if end == n:
            break

    return out


def make_classification_dataset(raw_ds, threshold=0.5, window_tokens=160, stride_tokens=80):
    rows = []

    for ex in raw_ds:
        toks = ex["tokens"]
        labs = bio_fix_sequence(ex["labels"])

        windows = build_token_windows(toks, labs, window_tokens=window_tokens, stride_tokens=stride_tokens)

        for w_idx, (s, e, win_toks, win_labs) in enumerate(windows):
            y = coverage_label_from_bio(win_labs, threshold=threshold)

            row = {
                "text": " ".join(win_toks),  # keep for inspection; we'll drop it from training ds later
                "labels": int(y),            # Trainer expects "labels"
                "win_start_token": int(s),
                "win_end_token": int(e),
                "window_index": int(w_idx),
            }

            # optional metadata pass-through
            if "doc_index" in ex:
                row["doc_index"] = ex["doc_index"]
            if "sample_index" in ex:
                row["sample_index"] = ex["sample_index"]

            rows.append(row)

    return Dataset.from_list(rows)


# ----------------------------
# Tokenization (sequence classification)
# ----------------------------
def tokenize_cls(example, tokenizer):
    max_len = getattr(tokenizer, "model_max_length", 512) or 512
    max_len = min(max_len, 512)
    return tokenizer(example["text"], truncation=True, max_length=max_len)


# ----------------------------
# Metrics (binary)
# ----------------------------
def compute_binary_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1).astype(int)
    labels = labels.astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = float((labels == preds).mean()) if labels.size else 0.0
    kappa = float(cohen_kappa_score(labels, preds, labels=[0, 1])) if labels.size else 0.0

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float(acc),
        "cohen_kappa": float(kappa),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm,
        "num_examples": int(labels.size),
    }


def drop_non_tensor_columns(ds):
    """
    IMPORTANT: remove any string/object columns (e.g. 'text') so the data collator
    doesn't try to convert them into tensors.
    """
    keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in ds.column_names:
        keep.add("token_type_ids")

    cols_to_remove = [c for c in ds.column_names if c not in keep]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--coverage_threshold", type=float, default=0.5)
    parser.add_argument("--window_tokens", type=int, default=160)
    parser.add_argument("--stride_tokens", type=int, default=80)

    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    base_model = cfg.get("base_model", "xlm-roberta-base")

    cat_cfg = next(c for c in cfg["categories"] if c["name"] == args.category)
    train_path = cat_cfg["train_path"]
    dev_path = cat_cfg.get("dev_path", cat_cfg.get("eval_path"))
    eval_path = cat_cfg.get("eval_path", cat_cfg.get("dev_path"))

    output_root = Path(cfg.get("output_root", "./runs"))
    run_name = cat_cfg.get("run_name", args.category)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    thr = float(args.coverage_threshold)
    win = int(args.window_tokens)
    stride = int(args.stride_tokens)

    print("Base model:", base_model)
    print("Category:", args.category)
    print("coverage_threshold:", thr)
    print("window_tokens:", win, "stride_tokens:", stride)

    train_raw = load_bio_json(train_path)
    dev_raw = load_bio_json(dev_path)
    eval_raw = load_bio_json(eval_path)

    train_cls = make_classification_dataset(train_raw, threshold=thr, window_tokens=win, stride_tokens=stride)
    dev_cls = make_classification_dataset(dev_raw, threshold=thr, window_tokens=win, stride_tokens=stride)
    eval_cls = make_classification_dataset(eval_raw, threshold=thr, window_tokens=win, stride_tokens=stride)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Tokenize
    train_tok = train_cls.map(lambda x: tokenize_cls(x, tokenizer), batched=False)
    dev_tok = dev_cls.map(lambda x: tokenize_cls(x, tokenizer), batched=False)
    eval_tok = eval_cls.map(lambda x: tokenize_cls(x, tokenizer), batched=False)

    # Make a copy of eval for inspection (keep text + metadata)
    eval_for_inspect = eval_tok

    # Drop string columns before training (FIXES your crash)
    train_ds = drop_non_tensor_columns(train_tok)
    dev_ds = drop_non_tensor_columns(dev_tok)
    eval_ds = drop_non_tensor_columns(eval_tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label={0: "NOT_CATEGORY", 1: args.category.upper()},
        label2id={"NOT_CATEGORY": 0, args.category.upper(): 1},
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

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_binary_metrics,
    )

    print(f"{args.category} WINDOWED classification training")
    trainer.train()

    # Final eval (and save inspection artifact with text)
    pred = trainer.predict(eval_ds)
    logits = pred.predictions
    y_pred = np.argmax(logits, axis=1).astype(int)
    y_true = np.array(eval_ds["labels"], dtype=int)

    # Build inspection list using eval_for_inspect (which still has text/metadata)
    inspect = []
    for ex, yp in zip(eval_for_inspect, y_pred.tolist()):
        inspect.append(
            {
                "doc_index": ex.get("doc_index"),
                "sample_index": ex.get("sample_index"),
                "window_index": ex.get("window_index"),
                "win_start_token": ex.get("win_start_token"),
                "win_end_token": ex.get("win_end_token"),
                "text": ex.get("text"),
                "gold": int(ex["labels"]),
                "pred": int(yp),
            }
        )

    inspect_path = output_dir / "eval_window_predictions.json"
    with open(inspect_path, "w", encoding="utf-8") as f:
        json.dump(inspect, f, indent=2, ensure_ascii=False)

    results = {
        "task": "windowed_sequence_classification",
        "category": args.category,
        "base_model": base_model,
        "coverage_threshold": thr,
        "window_tokens": win,
        "stride_tokens": stride,
        "final_eval_metrics": compute_binary_metrics((logits, y_true)),
        "artifacts": {"eval_window_predictions_path": str(inspect_path)},
    }

    print("\n-----FINAL EVAL RESULTS (WINDOWED CLASSIFICATION)-----")
    print(json.dumps(results, indent=2))

    with open(output_dir / "final_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()