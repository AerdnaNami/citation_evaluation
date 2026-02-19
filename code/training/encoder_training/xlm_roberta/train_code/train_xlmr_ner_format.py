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
    """
    Expects list of:
      {"tokens":[...], "labels":[...]}   (optional "sentence")
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

        # ensure strings
        toks = [str(t) for t in toks]
        labs = [str(l) for l in labs]

        cleaned.append(
            {
                "tokens": toks,
                "labels": labs,
                # helpful for later span outputs (optional)
                "sentence": ex.get("sentence", " ".join(toks)),
            }
        )

    return Dataset.from_list(cleaned)


def tokenize_and_align_labels(example, tokenizer, label2id):
    max_len = getattr(tokenizer, "model_max_length", 512)
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
    aligned = []
    prev_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            aligned.append(-100)
        elif word_idx != prev_word_idx:
            aligned.append(label2id[example["labels"][word_idx]])
        else:
            aligned.append(-100)
        prev_word_idx = word_idx

    tokenized["labels"] = aligned
    return tokenized


# ----------------------------
# Metrics helpers
# ----------------------------
def compute_seqeval_f1(predictions, labels):
    true_preds, true_labels = [], []

    for pred_seq, label_seq in zip(predictions, labels):
        preds, labs = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                preds.append(id2label[int(p)])
                labs.append(id2label[int(l)])
        true_preds.append(preds)
        true_labels.append(labs)

    return f1_score(true_labels, true_preds)


def flatten_valid_tokens(predictions, labels):
    flat_preds, flat_labels = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                flat_preds.append(int(p))
                flat_labels.append(int(l))
    return np.array(flat_preds), np.array(flat_labels)


def filter_out_gold_O(flat_preds, flat_labels, id2label):
    if flat_labels.size == 0:
        return flat_preds, flat_labels
    keep_mask = np.array([id2label[int(l)] != "O" for l in flat_labels], dtype=bool)
    return flat_preds[keep_mask], flat_labels[keep_mask]


def krippendorff_alpha_nominal(rater1: np.ndarray, rater2: np.ndarray, num_labels: int) -> float:
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
# Token overlap F1 (binary): entity vs non-entity
# ----------------------------
def infer_positive_entity_type_from_label_list(label_list: List[str]) -> str:
    """
    If labels are like O, B-COHERENCE, I-COHERENCE -> returns COHERENCE
    If multiple types exist, chooses one deterministically and warns.
    """
    types = set()
    for lbl in label_list:
        if lbl.startswith("B-") or lbl.startswith("I-"):
            parts = lbl.split("-", 1)
            if len(parts) == 2 and parts[1].strip():
                types.add(parts[1].strip())
    if not types:
        return ""  # fallback: treat any non-O as positive
    if len(types) > 1:
        picked = sorted(types)[0]
        print(f"[WARN] Multiple entity types found in labels: {sorted(types)}. Using: {picked}")
        return picked
    return next(iter(types))


def token_overlap_f1_from_flat(flat_preds: np.ndarray, flat_labels: np.ndarray, id2label: Dict[int, str]) -> Dict[str, Any]:
    """
    Computes token-level overlap F1 for:
      positive = tokens inside the entity (B-*/I-*), negative = O

    Uses binary PRF on tokens (micro for the positive class).
    """
    if flat_labels.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_tokens": 0}

    # Determine which entity type is considered "positive" (usually only one)
    label_list = [id2label[i] for i in range(len(id2label))]
    positive_type = infer_positive_entity_type_from_label_list(label_list)

    def is_pos(lbl_str: str) -> int:
        s = (lbl_str or "O").strip()
        if s == "O":
            return 0
        if positive_type:
            if (s.startswith("B-") or s.startswith("I-")) and s.split("-", 1)[1] == positive_type:
                return 1
            return 0
        # fallback: any non-O is positive
        return 1

    gold_bin = np.array([is_pos(id2label[int(i)]) for i in flat_labels], dtype=np.int64)
    pred_bin = np.array([is_pos(id2label[int(i)]) for i in flat_preds], dtype=np.int64)

    p, r, f1, _ = precision_recall_fscore_support(gold_bin, pred_bin, average="binary", zero_division=0)

    return {
        "positive_entity_type": positive_type if positive_type else "ANY_NON_O",
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "num_tokens": int(gold_bin.size),
        "support_positive_gold": int(gold_bin.sum()),
        "support_positive_pred": int(pred_bin.sum()),
    }


# ----------------------------
# Pred span helpers (word-level)
# ----------------------------
def _word_level_pred_ids_from_aligned(pred_ids_seq, aligned_label_ids_seq):
    out = []
    for p, l in zip(pred_ids_seq.tolist(), aligned_label_ids_seq.tolist()):
        if l != -100:
            out.append(int(p))
    return out


def _bio_spans_from_word_labels(tokens, word_labels):
    spans = []
    i = 0
    n = min(len(tokens), len(word_labels))

    def norm(lbl: str) -> str:
        return lbl.strip() if isinstance(lbl, str) else "O"

    while i < n:
        lab = norm(word_labels[i])
        if lab == "O":
            i += 1
            continue

        if lab.startswith("B-"):
            ent = lab[2:]
            start = i
            i += 1
            while i < n and norm(word_labels[i]) == f"I-{ent}":
                i += 1
            end = i
            spans.append({"label": ent, "start_token": start, "end_token": end, "text": " ".join(tokens[start:end])})
            continue

        if lab.startswith("I-"):
            ent = lab[2:]
            start = i
            i += 1
            while i < n and norm(word_labels[i]) == f"I-{ent}":
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

    # Load datasets (already tokens/labels)
    train_dataset_raw = load_ner_json(train_path)
    dev_dataset_raw = load_ner_json(dev_path)
    eval_dataset_raw = load_ner_json(eval_path)

    # Label set from TRAIN
    label_list = sorted(list({lab for ex in train_dataset_raw for lab in ex["labels"]}))

    global label2id, id2label
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Tokenize + align labels
    train_dataset = train_dataset_raw.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=False)
    dev_dataset = dev_dataset_raw.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=False)
    eval_dataset = eval_dataset_raw.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=False)

    # Remove non-tensor columns after tokenization
    cols_to_keep = {"input_ids", "attention_mask", "labels", "token_type_ids"}

    def remove_non_tensor_cols(ds: Dataset, name: str) -> Dataset:
        cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
        if cols_to_remove:
            print(f"[{name}] removing columns: {cols_to_remove}")
            ds = ds.remove_columns(cols_to_remove)
        return ds

    train_dataset = remove_non_tensor_cols(train_dataset, "train")
    dev_dataset = remove_non_tensor_cols(dev_dataset, "dev")
    eval_dataset = remove_non_tensor_cols(eval_dataset, "eval")

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
    logits, labels, _ = trainer.predict(eval_dataset)
    pred_ids = np.argmax(logits, axis=2)

    seqeval_f1 = compute_seqeval_f1(pred_ids, labels)
    flat_preds, flat_labels = flatten_valid_tokens(pred_ids, labels)

    # Token overlap (binary) on all scored tokens
    token_overlap = token_overlap_f1_from_flat(flat_preds, flat_labels, id2label)

    # ALL TOKENS
    cm_all = confusion_matrix(flat_labels, flat_preds, labels=list(range(len(label_list))))
    cm_all_dict = {"labels": label_list, "matrix": cm_all.tolist()}
    alpha_all = krippendorff_alpha_nominal(flat_labels, flat_preds, num_labels=len(label_list))
    kappa_all = float(cohen_kappa_score(flat_labels, flat_preds, labels=list(range(len(label_list))))) if flat_labels.size else 0.0
    p_micro_all, r_micro_all, f1_micro_all, _ = precision_recall_fscore_support(flat_labels, flat_preds, average="micro", zero_division=0)
    p_macro_all, r_macro_all, f1_macro_all, _ = precision_recall_fscore_support(flat_labels, flat_preds, average="macro", zero_division=0)

    # ENTITY ONLY
    flat_preds_ent, flat_labels_ent = filter_out_gold_O(flat_preds, flat_labels, id2label)
    cm_ent = confusion_matrix(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list))))
    cm_ent_dict = {"labels": label_list, "matrix": cm_ent.tolist()}
    alpha_ent = krippendorff_alpha_nominal(flat_labels_ent, flat_preds_ent, num_labels=len(label_list))
    kappa_ent = float(cohen_kappa_score(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list))))) if flat_labels_ent.size else 0.0
    p_micro_ent, r_micro_ent, f1_micro_ent, _ = precision_recall_fscore_support(flat_labels_ent, flat_preds_ent, average="micro", zero_division=0)
    p_macro_ent, r_macro_ent, f1_macro_ent, _ = precision_recall_fscore_support(flat_labels_ent, flat_preds_ent, average="macro", zero_division=0)

    # Predicted spans per example
    pred_spans_by_example: List[Dict[str, Any]] = []
    for raw_ex, pred_seq, lab_seq in zip(eval_dataset_raw, pred_ids, labels):
        tokens_words = raw_ex["tokens"]
        gold_word_labels = raw_ex["labels"]
        sentence_text = raw_ex.get("sentence", " ".join(tokens_words))

        pred_word_ids = _word_level_pred_ids_from_aligned(pred_seq, lab_seq)
        pred_word_labels = [id2label[i] for i in pred_word_ids]

        pred_spans = _bio_spans_from_word_labels(tokens_words, pred_word_labels)
        gold_spans = _bio_spans_from_word_labels(tokens_words, gold_word_labels)

        pred_spans_by_example.append(
            {
                "sentence": sentence_text,
                "tokens": tokens_words,
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
            }
        )

    pred_spans_path = output_dir / "eval_predicted_spans.json"
    with open(pred_spans_path, "w", encoding="utf-8") as f:
        json.dump(pred_spans_by_example, f, indent=2, ensure_ascii=False)

    results = {
        "seqeval_span_f1": float(seqeval_f1),
        "token_overlap_f1_binary": token_overlap,
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

    with open(output_dir / "final_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
