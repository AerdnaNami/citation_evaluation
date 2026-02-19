import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    cohen_kappa_score,
    accuracy_score,
)


# ----------------------------
# I/O (SPAN FORMAT)
# ----------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_span_json(path: str) -> Dataset:
    """
    Expects list of span samples like:
      {
        "doc_index": 0,
        "paragraph": "...",
        "span_text": "...",
        "start": 564,
        "end": 1117,
        "start_token": 68,
        "end_token": 156,
        "label": "LACKS_SYNTHESIS",
        "sample_index": 0
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list")

    cleaned = []
    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            continue

        para = ex.get("paragraph") or ex.get("text")
        if not isinstance(para, str) or not para.strip():
            raise ValueError(f"{path} example {i} missing paragraph/text")

        # prefer token indices (best for overlap)
        if "start_token" not in ex or "end_token" not in ex:
            raise ValueError(f"{path} example {i} missing start_token/end_token")

        try:
            st_tok = int(ex["start_token"])
            en_tok = int(ex["end_token"])
        except Exception:
            raise ValueError(f"{path} example {i} invalid start_token/end_token")

        label = str(ex.get("label", "CATEGORY")).strip()

        cleaned.append(
            {
                "doc_index": int(ex.get("doc_index", 0)),
                "paragraph": para,
                "span_text": ex.get("span_text", ""),
                "start": int(ex.get("start", -1)),
                "end": int(ex.get("end", -1)),
                "start_token": st_tok,
                "end_token": en_tok,
                "label": label,
                "sample_index": int(ex.get("sample_index", i)),
            }
        )

    return Dataset.from_list(cleaned)


# ----------------------------
# Sentence splitting over TOKENS (must match your start_token indexing)
# ----------------------------
_SENT_END = {".", "?", "!"}


def paragraph_tokens_from_span_text(ex: Dict[str, Any]) -> List[str]:
    """
    Your start_token/end_token were created from a whitespace tokenization of `paragraph`.
    To be consistent, we must tokenize the paragraph in the SAME way.

    This matches your earlier pipeline: splitting on whitespace and keeping punctuation attached.
    """
    return ex["paragraph"].split()


def split_tokens_into_sentences_with_indices(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Returns list of:
      {
        "tokens": [...],
        "start_token": i,
        "end_token": j,   # exclusive
      }
    """
    sents: List[Dict[str, Any]] = []
    cur: List[str] = []
    sent_start = 0

    for idx, t in enumerate(tokens):
        cur.append(t)
        if len(t) > 0 and t[-1] in _SENT_END:
            sents.append(
                {
                    "tokens": cur,
                    "start_token": sent_start,
                    "end_token": idx + 1,
                }
            )
            cur = []
            sent_start = idx + 1

    if cur:
        sents.append(
            {
                "tokens": cur,
                "start_token": sent_start,
                "end_token": len(tokens),
            }
        )

    return sents


def interval_overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """
    Returns number of overlapping tokens between [a_start,a_end) and [b_start,b_end)
    """
    ov_start = max(a_start, b_start)
    ov_end = min(a_end, b_end)
    return max(0, ov_end - ov_start)


def sentence_label_by_span_overlap(
    sent_start: int,
    sent_end: int,
    span_start: int,
    span_end: int,
    min_overlap_tokens: int = 1,
) -> int:
    """
    Positive if the sentence overlaps the gold span by >= min_overlap_tokens.
    """
    return 1 if interval_overlap_len(sent_start, sent_end, span_start, span_end) >= min_overlap_tokens else 0


def build_sentence_window_examples_from_span_sample(
    ex: Dict[str, Any],
    window: int = 1,
    min_overlap_tokens: int = 1,
    sep: str = " </s></s> ",
) -> List[Dict[str, Any]]:
    """
    For a single paragraph+span, create one training example per sentence.

    Label is based on overlap between sentence token range and gold span token range.
    """
    paragraph_tokens = paragraph_tokens_from_span_text(ex)
    sents = split_tokens_into_sentences_with_indices(paragraph_tokens)
    if not sents:
        return []

    # gold span (token indices)
    span_start = int(ex["start_token"])
    span_end = int(ex["end_token"])

    sent_texts = [" ".join(s["tokens"]) for s in sents]
    sent_labels = [
        sentence_label_by_span_overlap(
            s["start_token"],
            s["end_token"],
            span_start,
            span_end,
            min_overlap_tokens=min_overlap_tokens,
        )
        for s in sents
    ]

    out: List[Dict[str, Any]] = []
    n = len(sents)
    for i in range(n):
        left = max(0, i - window)
        right = min(n, i + window + 1)

        window_text = sep.join(sent_texts[left:right])

        out.append(
            {
                "doc_id": int(ex.get("doc_index", 0)),
                "sample_index": int(ex.get("sample_index", -1)),
                "text": window_text,
                "label": int(sent_labels[i]),
                "center_sentence": sent_texts[i],
                "sent_index": int(i),
                "sent_start_token": int(sents[i]["start_token"]),
                "sent_end_token": int(sents[i]["end_token"]),
                "span_start_token": int(span_start),
                "span_end_token": int(span_end),
                "paragraph": ex.get("paragraph", ""),
                "span_text": ex.get("span_text", ""),
                "span_label": ex.get("label", ""),
                "window_left": int(left),
                "window_right": int(right - 1),
            }
        )

    return out


def make_sentence_dataset_from_span_ds(
    ds_span: Dataset,
    window: int,
    min_overlap_tokens: int,
) -> Dataset:
    rows: List[Dict[str, Any]] = []
    for ex in ds_span:
        rows.extend(
            build_sentence_window_examples_from_span_sample(
                ex,
                window=window,
                min_overlap_tokens=min_overlap_tokens,
            )
        )
    return Dataset.from_list(rows)


# ----------------------------
# Token overlap F1 (derived from sentence predictions)
# ----------------------------
def compute_token_overlap_f1_from_span_samples(
    span_ds: Dataset,
    sent_pred_map: Dict[Tuple[int, int, int], int],
) -> Dict[str, float]:
    gold_tokens: List[int] = []
    pred_tokens: List[int] = []

    for ex in span_ds:
        doc_id = int(ex.get("doc_index", 0))
        sample_index = int(ex.get("sample_index", -1))

        paragraph_tokens = paragraph_tokens_from_span_text(ex)
        sents = split_tokens_into_sentences_with_indices(paragraph_tokens)

        span_start = int(ex["start_token"])
        span_end = int(ex["end_token"])

        for sent_index, s in enumerate(sents):
            pred_sent = int(sent_pred_map.get((doc_id, sample_index, sent_index), 0))

            for tok_i in range(s["start_token"], s["end_token"]):
                gold = 1 if (span_start <= tok_i < span_end) else 0
                gold_tokens.append(gold)
                pred_tokens.append(pred_sent)

    if not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_tokens": 0}

    gold_arr = np.array(gold_tokens, dtype=np.int64)
    pred_arr = np.array(pred_tokens, dtype=np.int64)

    p, r, f1, _ = precision_recall_fscore_support(gold_arr, pred_arr, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "num_tokens": int(gold_arr.size)}


# ----------------------------
# Class weighting + Trainer
# ----------------------------
def compute_binary_class_weights(ds: Dataset) -> np.ndarray:
    y = np.array([int(ex["label"]) for ex in ds], dtype=np.int64)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return w


class WeightedSeqClsTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ----------------------------
# Tokenization
# ----------------------------
def tokenize_for_seqcls(example, tokenizer):
    max_len = getattr(tokenizer, "model_max_length", 512)
    if max_len is None or max_len > 512:
        max_len = 512
    return tokenizer(example["text"], truncation=True, max_length=max_len)


# ----------------------------
# Metrics
# ----------------------------
def compute_eval_metrics(logits: np.ndarray, gold: np.ndarray) -> Dict[str, Any]:
    pred = np.argmax(logits, axis=1).astype(np.int64)
    gold = gold.astype(np.int64)

    acc = float(accuracy_score(gold, pred)) if gold.size else 0.0
    kappa = float(cohen_kappa_score(gold, pred, labels=[0, 1])) if gold.size else 0.0
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(gold, pred, average="micro", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(gold, pred, average="macro", zero_division=0)
    cm = confusion_matrix(gold, pred, labels=[0, 1])

    return {
        "accuracy": acc,
        "cohen_kappa": kappa,
        "prf": {
            "micro": {"precision": float(p_micro), "recall": float(r_micro), "f1": float(f1_micro)},
            "macro": {"precision": float(p_macro), "recall": float(r_macro), "f1": float(f1_macro)},
        },
        "confusion_matrix": {"labels": ["NOT_CATEGORY", "CATEGORY"], "matrix": cm.tolist()},
        "num_sentences_scored": int(gold.size),
    }


# ----------------------------
# Main
# ----------------------------
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
    dev_path = cat_cfg.get("dev_path", cat_cfg.get("eval_path"))
    eval_path = cat_cfg.get("eval_path", cat_cfg.get("dev_path"))

    output_root = Path(cfg.get("output_root", "./runs"))
    run_name = cat_cfg.get("run_name", args.category)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load span format
    train_span = load_span_json(train_path)
    dev_span = load_span_json(dev_path)
    eval_span = load_span_json(eval_path)

    # window + overlap params
    window = int(cat_cfg.get("sentence_window", defaults.get("sentence_window", 1)))
    min_overlap_tokens = int(cat_cfg.get("sentence_min_overlap_tokens", defaults.get("sentence_min_overlap_tokens", 1)))

    # Build sentence-level datasets from spans
    train_sent = make_sentence_dataset_from_span_ds(train_span, window=window, min_overlap_tokens=min_overlap_tokens)
    dev_sent = make_sentence_dataset_from_span_ds(dev_span, window=window, min_overlap_tokens=min_overlap_tokens)
    eval_sent = make_sentence_dataset_from_span_ds(eval_span, window=window, min_overlap_tokens=min_overlap_tokens)

    print(f"Sentence dataset sizes: train={len(train_sent)}, dev={len(dev_sent)}, eval={len(eval_sent)}")
    print("Train sentence label distribution:", Counter(train_sent["label"]))
    print("Eval sentence label distribution:", Counter(eval_sent["label"]))

    # Binary labels
    label2id = {"NOT_CATEGORY": 0, "CATEGORY": 1}
    id2label = {0: "NOT_CATEGORY", 1: "CATEGORY"}

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_tok = train_sent.map(lambda x: tokenize_for_seqcls(x, tokenizer), batched=False)
    dev_tok = dev_sent.map(lambda x: tokenize_for_seqcls(x, tokenizer), batched=False)
    eval_tok = eval_sent.map(lambda x: tokenize_for_seqcls(x, tokenizer), batched=False)

    cols_to_keep = {"input_ids", "attention_mask", "token_type_ids", "label"}

    def prune_cols(ds: Dataset, name: str) -> Dataset:
        cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
        if cols_to_remove:
            print(f"[{name}] removing columns: {cols_to_remove}")
            ds = ds.remove_columns(cols_to_remove)
        if "label" in ds.column_names:
            ds = ds.rename_column("label", "labels")
        return ds

    train_tok = prune_cols(train_tok, "train")
    dev_tok = prune_cols(dev_tok, "dev")
    eval_tok = prune_cols(eval_tok, "eval")

    # Class weights
    w = compute_binary_class_weights(train_sent)
    print("Binary class weights [NOT_CATEGORY, CATEGORY]:", w.tolist())
    class_weights_t = torch.tensor(w, dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy=defaults.get("evaluation_strategy", "epoch"),
        save_strategy=defaults.get("save_strategy", "epoch"),
        logging_steps=int(defaults.get("logging_steps", 500)),
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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedSeqClsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        class_weights=class_weights_t,
    )

    print(f"{cat_cfg['name']} sentence-level training (window={window}, min_overlap_tokens={min_overlap_tokens})")
    trainer.train()

    # ----------------------------
    # FINAL EVAL
    # ----------------------------
    out = trainer.predict(eval_tok)
    logits = out.predictions
    gold = out.label_ids

    eval_metrics = compute_eval_metrics(logits, gold)

    pred = np.argmax(logits, axis=1).astype(int).tolist()
    probs = torch.softmax(torch.tensor(logits), dim=1).cpu().numpy().tolist()

    # Map: (doc_id, sample_index, sent_index) -> pred
    sent_pred_map: Dict[Tuple[int, int, int], int] = {}
    for ex, yhat in zip(eval_sent, pred):
        sent_pred_map[(int(ex["doc_id"]), int(ex["sample_index"]), int(ex["sent_index"]))] = int(yhat)

    token_overlap = compute_token_overlap_f1_from_span_samples(
        span_ds=eval_span,
        sent_pred_map=sent_pred_map,
    )

    # Save per-sentence preds
    pred_rows = []
    for ex, yhat, pr in zip(eval_sent, pred, probs):
        pred_rows.append(
            {
                "doc_id": int(ex.get("doc_id", -1)),
                "sample_index": int(ex.get("sample_index", -1)),
                "sent_index": int(ex.get("sent_index", -1)),
                "sent_start_token": int(ex.get("sent_start_token", -1)),
                "sent_end_token": int(ex.get("sent_end_token", -1)),
                "span_start_token": int(ex.get("span_start_token", -1)),
                "span_end_token": int(ex.get("span_end_token", -1)),
                "gold_label": int(ex["label"]),
                "pred_label": int(yhat),
                "prob_not_category": float(pr[0]),
                "prob_category": float(pr[1]),
                "center_sentence": ex.get("center_sentence", ""),
                "window_text": ex.get("text", ""),
                "paragraph": ex.get("paragraph", ""),
                "span_text": ex.get("span_text", ""),
                "span_label": ex.get("span_label", ""),
            }
        )

    preds_path = output_dir / "eval_sentence_predictions.json"
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(pred_rows, f, indent=2, ensure_ascii=False)

    results = {
        "category": args.category,
        "sentence_window": window,
        "sentence_min_overlap_tokens": min_overlap_tokens,
        "eval_metrics_sentence_level": eval_metrics,
        "token_level_overlap_f1": token_overlap,
        "artifacts": {
            "eval_sentence_predictions_path": str(preds_path),
        },
    }

    print("\n-----FINAL EVAL RESULTS (SPAN->SENTENCE OVERLAP)-----")
    print(json.dumps(results, indent=2))

    with open(output_dir / "final_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
