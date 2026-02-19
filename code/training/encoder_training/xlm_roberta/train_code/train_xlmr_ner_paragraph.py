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
# I/O
# ----------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ner_json(path: str) -> Dataset:
    """
    Input format unchanged:
      [{"tokens":[...], "labels":[...], optional: "paragraph"/"text"}]

    labels expected like:
      O, B-COHERENCE, I-COHERENCE
      or O, B-LACKS_SYNTHESIS, I-LACKS_SYNTHESIS
    """
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

        toks = [str(t) for t in toks]
        labs = [str(l) for l in labs]

        paragraph_text = ex.get("paragraph") or ex.get("text") or " ".join(toks)

        cleaned.append(
            {
                "tokens": toks,
                "labels": labs,
                "paragraph": paragraph_text,
            }
        )

    return Dataset.from_list(cleaned)


# ----------------------------
# Sentence splitting + windowing
# ----------------------------
_SENT_END = {".", "?", "!"}


def split_tokens_into_sentences(tokens: List[str], labels: List[str]) -> List[Tuple[List[str], List[str]]]:
    """
    Simple rule-based token sentence split. Works on your existing token list.
    Keeps labels aligned exactly.
    """
    sents: List[Tuple[List[str], List[str]]] = []
    cur_toks: List[str] = []
    cur_labs: List[str] = []

    for t, l in zip(tokens, labels):
        cur_toks.append(t)
        cur_labs.append(l)

        # End sentence when token endswith a sentence-final punctuation.
        if len(t) > 0 and (t[-1] in _SENT_END):
            sents.append((cur_toks, cur_labs))
            cur_toks, cur_labs = [], []

    if cur_toks:
        sents.append((cur_toks, cur_labs))

    return sents


def _entity_type_from_bio(label: str) -> str:
    if not isinstance(label, str):
        return ""
    label = label.strip()
    if label.startswith("B-") or label.startswith("I-"):
        parts = label.split("-", 1)
        return parts[1] if len(parts) == 2 else ""
    return ""


def infer_positive_entity_type(train_dataset_raw: Dataset) -> str:
    """
    Finds the single entity type used for this category from the BIO labels.
    For coherence it should infer 'COHERENCE'; for lacks synthesis, 'LACKS_SYNTHESIS', etc.
    """
    types = set()
    for ex in train_dataset_raw:
        for l in ex["labels"]:
            t = _entity_type_from_bio(l)
            if t:
                types.add(t)
    if not types:
        raise ValueError("Could not infer positive entity type from labels (no B-/I- labels found).")
    if len(types) > 1:
        picked = sorted(types)[0]
        print(f"[WARN] Multiple entity types found in labels: {sorted(types)}. Using: {picked}")
        return picked
    return next(iter(types))


def sentence_label_from_token_labels(
    sent_labels: List[str],
    positive_entity_type: str,
    positive_threshold: float = 0.5,
) -> int:
    """
    Binary label:
      1 = sentence belongs to the category (e.g., COHERENCE / LACKS_SYNTHESIS)
      0 = not-category

    Uses coverage fraction: (# tokens with B/I of positive type) / (sentence tokens)
    """
    if not sent_labels:
        return 0
    pos = 0
    for l in sent_labels:
        et = _entity_type_from_bio(l)
        if et == positive_entity_type:
            pos += 1
    frac = pos / max(len(sent_labels), 1)
    return 1 if frac >= positive_threshold else 0


def build_sentence_window_examples(
    paragraph_tokens: List[str],
    paragraph_labels: List[str],
    paragraph_text: str,
    doc_id: int,
    positive_entity_type: str,
    window: int = 1,
    positive_threshold: float = 0.5,
    sep: str = " </s></s> ",
) -> List[Dict[str, Any]]:
    """
    Produces examples for sentence-level classification with context.

    For each sentence i:
      text = s[i-window] + SEP + ... + SEP + s[i] + SEP + ... + s[i+window]
      label = label(s[i]) computed from token label coverage

    Also stores sentence metadata for debugging/artifacts, including doc_id/sent_index
    so we can map sentence predictions back to tokens.
    """
    sents = split_tokens_into_sentences(paragraph_tokens, paragraph_labels)
    if not sents:
        return []

    sent_texts = [" ".join(toks) for toks, _ in sents]
    sent_labs = [sentence_label_from_token_labels(labs, positive_entity_type, positive_threshold) for _, labs in sents]

    out = []
    n = len(sents)
    for i in range(n):
        left = max(0, i - window)
        right = min(n, i + window + 1)
        window_text = sep.join(sent_texts[left:right])

        out.append(
            {
                "doc_id": int(doc_id),
                "text": window_text,
                "label": int(sent_labs[i]),
                "center_sentence": sent_texts[i],
                "sent_index": int(i),
                "window_left": int(left),
                "window_right": int(right - 1),
                "paragraph": paragraph_text,
            }
        )
    return out


def make_sentence_dataset(
    ds_raw: Dataset,
    positive_entity_type: str,
    window: int,
    positive_threshold: float,
) -> Dataset:
    rows: List[Dict[str, Any]] = []
    for doc_id, ex in enumerate(ds_raw):
        rows.extend(
            build_sentence_window_examples(
                paragraph_tokens=ex["tokens"],
                paragraph_labels=ex["labels"],
                paragraph_text=ex.get("paragraph", " ".join(ex["tokens"])),
                doc_id=doc_id,
                positive_entity_type=positive_entity_type,
                window=window,
                positive_threshold=positive_threshold,
            )
        )
    return Dataset.from_list(rows)


# ----------------------------
# Token-level overlap F1 (token classification derived from sentence preds)
# ----------------------------
def _token_is_positive(label: str, positive_entity_type: str) -> int:
    et = _entity_type_from_bio(label)
    return 1 if et == positive_entity_type else 0


def compute_token_overlap_f1(
    eval_raw: Dataset,
    positive_entity_type: str,
    sent_pred_map: Dict[Tuple[int, int], int],
) -> Dict[str, float]:
    """
    Builds token-level binary arrays:
      gold_token = 1 if BIO label is B/I of positive_entity_type else 0
      pred_token = 1 if its sentence was predicted CATEGORY else 0

    Then computes overlap Precision/Recall/F1 (micro over tokens).
    """
    gold_tokens: List[int] = []
    pred_tokens: List[int] = []

    for doc_id, ex in enumerate(eval_raw):
        tokens = ex["tokens"]
        labels = ex["labels"]

        sents = split_tokens_into_sentences(tokens, labels)

        sent_index = 0
        for sent_toks, sent_labs in sents:
            pred_sent = int(sent_pred_map.get((doc_id, sent_index), 0))

            for l in sent_labs:
                gold_tokens.append(_token_is_positive(l, positive_entity_type))
                pred_tokens.append(pred_sent)

            sent_index += 1

    if not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_tokens": 0}

    gold_arr = np.array(gold_tokens, dtype=np.int64)
    pred_arr = np.array(pred_tokens, dtype=np.int64)

    p, r, f1, _ = precision_recall_fscore_support(gold_arr, pred_arr, average="binary", zero_division=0)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "num_tokens": int(gold_arr.size),
    }


# ----------------------------
# Class weighting + Trainer
# ----------------------------
def compute_binary_class_weights(ds: Dataset) -> np.ndarray:
    """
    weights[0] for label 0, weights[1] for label 1
    Inverse frequency, normalized to mean=1.
    """
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
        logits = outputs.get("logits")  # [batch, num_labels]

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
    parser.add_argument("--category", required=True)  # e.g., "coherence" or "lacks_synthesis"
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

    # Load raw BIO data (format unchanged)
    train_raw = load_ner_json(train_path)
    dev_raw = load_ner_json(dev_path)
    eval_raw = load_ner_json(eval_path)

    # Infer the positive entity type from BIO labels (e.g., COHERENCE or LACKS_SYNTHESIS)
    positive_entity_type = infer_positive_entity_type(train_raw)
    print(f"Positive entity type inferred from BIO labels: {positive_entity_type}")

    # Sentence window parameters (tunable per category via config defaults)
    window = int(cat_cfg.get("sentence_window", defaults.get("sentence_window", 1)))
    positive_threshold = float(cat_cfg.get("sentence_positive_threshold", defaults.get("sentence_positive_threshold", 0.5)))

    # Build sentence-level datasets (now include doc_id + sent_index)
    train_sent = make_sentence_dataset(train_raw, positive_entity_type, window=window, positive_threshold=positive_threshold)
    dev_sent = make_sentence_dataset(dev_raw, positive_entity_type, window=window, positive_threshold=positive_threshold)
    eval_sent = make_sentence_dataset(eval_raw, positive_entity_type, window=window, positive_threshold=positive_threshold)

    print(f"Sentence dataset sizes: train={len(train_sent)}, dev={len(dev_sent)}, eval={len(eval_sent)}")

    label2id = {"NOT_CATEGORY": 0, "CATEGORY": 1}
    id2label = {0: "NOT_CATEGORY", 1: "CATEGORY"}

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_tok = train_sent.map(lambda x: tokenize_for_seqcls(x, tokenizer), batched=False)
    dev_tok = dev_sent.map(lambda x: tokenize_for_seqcls(x, tokenizer), batched=False)
    eval_tok = eval_sent.map(lambda x: tokenize_for_seqcls(x, tokenizer), batched=False)

    # Keep only the fields needed by Trainer + keep label
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

    # Class weights (binary)
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
    print("Train sentence label distribution:", Counter(train_sent["label"]))
    print("Eval sentence label distribution:", Counter(eval_sent["label"]))

    print(f"{cat_cfg['name']} sentence-level training (window={window}, threshold={positive_threshold})")
    trainer.train()

    # ----------------------------
    # FINAL EVALUATION ON EVAL SET
    # ----------------------------
    out = trainer.predict(eval_tok)
    logits = out.predictions
    gold = out.label_ids

    eval_metrics = compute_eval_metrics(logits, gold)

    # Sentence predictions
    pred = np.argmax(logits, axis=1).astype(int).tolist()
    probs = torch.softmax(torch.tensor(logits), dim=1).cpu().numpy().tolist()

    # Build map so we can compute token-level overlap F1
    # (doc_id, sent_index) -> predicted sentence label
    sent_pred_map: Dict[Tuple[int, int], int] = {}
    for ex, yhat in zip(eval_sent, pred):
        sent_pred_map[(int(ex["doc_id"]), int(ex["sent_index"]))] = int(yhat)

    token_overlap = compute_token_overlap_f1(
        eval_raw=eval_raw,
        positive_entity_type=positive_entity_type,
        sent_pred_map=sent_pred_map,
    )

    # Save per-sentence predictions for inspection
    pred_rows = []
    for ex, yhat, pr in zip(eval_sent, pred, probs):
        pred_rows.append(
            {
                "doc_id": int(ex.get("doc_id", -1)),
                "paragraph": ex.get("paragraph", ""),
                "center_sentence": ex.get("center_sentence", ""),
                "sent_index": ex.get("sent_index", -1),
                "window_left": ex.get("window_left", -1),
                "window_right": ex.get("window_right", -1),
                "gold_label": int(ex["label"]),
                "pred_label": int(yhat),
                "prob_not_category": float(pr[0]),
                "prob_category": float(pr[1]),
                "window_text": ex.get("text", ""),
            }
        )

    preds_path = output_dir / "eval_sentence_predictions.json"
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(pred_rows, f, indent=2, ensure_ascii=False)

    results = {
        "category": args.category,
        "positive_entity_type_inferred": positive_entity_type,
        "sentence_window": window,
        "sentence_positive_threshold": positive_threshold,
        "eval_metrics_sentence_level": eval_metrics,
        "token_level_overlap_f1": token_overlap,
        "artifacts": {
            "eval_sentence_predictions_path": str(preds_path),
        },
    }

    print("\n-----FINAL EVAL RESULTS (SENTENCE LEVEL + TOKEN OVERLAP)-----")
    print(json.dumps(results, indent=2))

    with open(output_dir / "final_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
