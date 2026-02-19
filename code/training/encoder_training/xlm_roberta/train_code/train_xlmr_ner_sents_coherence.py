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

        # for span previews later; coherence is paragraph-level text
        paragraph_text = ex.get("paragraph") or ex.get("text") or " ".join(toks)

        cleaned.append(
            {
                "tokens": toks,
                "labels": labs,
                "paragraph": paragraph_text,
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


def _word_level_pred_ids_from_aligned(pred_ids_seq, aligned_label_ids_seq):
    out = []
    for p, l in zip(pred_ids_seq.tolist(), aligned_label_ids_seq.tolist()):
        if l != -100:
            out.append(int(p))
    return out

def bio_fix_sequence(word_labels: List[str]) -> List[str]:
    """
    Enforce BIO validity:
      - If an I-X appears without a preceding B-X or I-X, convert it to B-X.
      - If an I-X follows B-Y/I-Y where X != Y, convert it to B-X.
    Keeps 'O' as is; leaves non-BIO tags unchanged.
    """
    fixed: List[str] = []
    prev = "O"

    def ent_type(lbl: str) -> str:
        return lbl.split("-", 1)[1] if "-" in lbl else ""

    for lbl in word_labels:
        lbl = (lbl or "O").strip()

        if lbl.startswith("I-"):
            x = ent_type(lbl)

            # previous must be B-X or I-X, otherwise start a new span
            if not (prev == f"B-{x}" or prev == f"I-{x}"):
                lbl = f"B-{x}"
            # previous is an entity but different type -> start new span
            else:
                # already correct type handled above; nothing else needed
                pass

        # also handle the "I-X after B-Y/I-Y (Y != X)" case explicitly
        if lbl.startswith("I-"):
            x = ent_type(lbl)
            if prev.startswith(("B-", "I-")) and ent_type(prev) != x:
                lbl = f"B-{x}"

        fixed.append(lbl)
        prev = lbl

    return fixed


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

def apply_bio_fix_to_aligned_pred_ids(pred_ids: np.ndarray, labels: np.ndarray, id2label: Dict[int, str], label2id: Dict[str, int]) -> np.ndarray:
    """
    pred_ids: [batch, seq_len] argmax predictions
    labels:   [batch, seq_len] with -100 for non-word-start/padding
    Returns a new pred_ids with BIO fixed ONLY on positions where labels != -100.
    """
    fixed = pred_ids.copy()

    for i in range(pred_ids.shape[0]):
        # indices of word-level positions (where we have gold labels)
        idxs = np.where(labels[i] != -100)[0]
        if idxs.size == 0:
            continue

        # current word-level label strings from predictions
        word_pred_labels = [id2label[int(pred_ids[i, j])] for j in idxs]

        # fix BIO
        word_pred_labels_fixed = bio_fix_sequence(word_pred_labels)

        # write back to fixed ids
        for j, lbl in zip(idxs, word_pred_labels_fixed):
            fixed[i, j] = int(label2id[lbl])

    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", required=True)  # use "coherence"
    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    base_model = cfg.get("base_model", "xlm-roberta-base")

    # --- COHERENCE CATEGORY ---
    # expects config categories item named "coherence" with NER files
    cat_cfg = next(c for c in cfg["categories"] if c["name"] == args.category)

    train_path = cat_cfg["train_path"]
    dev_path = cat_cfg.get("dev_path", cat_cfg.get("eval_path"))
    eval_path = cat_cfg.get("eval_path", cat_cfg.get("dev_path"))

    output_root = Path(cfg.get("output_root", "./runs"))
    run_name = cat_cfg.get("run_name", args.category)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets (already tokens/labels for COHERENCE)
    train_dataset_raw = load_ner_json(train_path)
    dev_dataset_raw = load_ner_json(dev_path)
    eval_dataset_raw = load_ner_json(eval_path)

    # Label set from TRAIN (for coherence: O/B-COHERENCE/I-COHERENCE)
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

    print("COHERENCE training")
    print("Label list:", label_list)
    trainer.train()

    # ----------------------------
    # FINAL EVALUATION ON EVAL SET
    # ----------------------------
    logits, labels, _ = trainer.predict(eval_dataset)
    pred_ids = np.argmax(logits, axis=2)

    pred_ids_fixed = apply_bio_fix_to_aligned_pred_ids(pred_ids, labels, id2label, label2id)
    seqeval_f1 = compute_seqeval_f1(pred_ids_fixed, labels)
    flat_preds, flat_labels = flatten_valid_tokens(pred_ids_fixed, labels)

    # ALL TOKENS
    cm_all = confusion_matrix(flat_labels, flat_preds, labels=list(range(len(label_list))))
    cm_all_dict = {"labels": label_list, "matrix": cm_all.tolist()}
    alpha_all = krippendorff_alpha_nominal(flat_labels, flat_preds, num_labels=len(label_list))
    kappa_all = (
        float(cohen_kappa_score(flat_labels, flat_preds, labels=list(range(len(label_list)))))
        if flat_labels.size
        else 0.0
    )
    p_micro_all, r_micro_all, f1_micro_all, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average="micro", zero_division=0
    )
    p_macro_all, r_macro_all, f1_macro_all, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average="macro", zero_division=0
    )

    # ENTITY ONLY (ignore gold O)
    flat_preds_ent, flat_labels_ent = filter_out_gold_O(flat_preds, flat_labels, id2label)
    cm_ent = confusion_matrix(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list))))
    cm_ent_dict = {"labels": label_list, "matrix": cm_ent.tolist()}
    alpha_ent = krippendorff_alpha_nominal(flat_labels_ent, flat_preds_ent, num_labels=len(label_list))
    kappa_ent = (
        float(cohen_kappa_score(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list)))))
        if flat_labels_ent.size
        else 0.0
    )
    p_micro_ent, r_micro_ent, f1_micro_ent, _ = precision_recall_fscore_support(
        flat_labels_ent, flat_preds_ent, average="micro", zero_division=0
    )
    p_macro_ent, r_macro_ent, f1_macro_ent, _ = precision_recall_fscore_support(
        flat_labels_ent, flat_preds_ent, average="macro", zero_division=0
    )

    # Predicted spans per example (paragraph-level preview)
    pred_spans_by_example: List[Dict[str, Any]] = []
    for raw_ex, pred_seq, lab_seq in zip(eval_dataset_raw, pred_ids, labels):
        tokens_words = raw_ex["tokens"]
        gold_word_labels = raw_ex["labels"]
        paragraph_text = raw_ex.get("paragraph", " ".join(tokens_words))

        pred_word_ids = _word_level_pred_ids_from_aligned(pred_seq, lab_seq)
        pred_word_labels = [id2label[i] for i in pred_word_ids]
        pred_word_labels_fixed = bio_fix_sequence(pred_word_labels)

        pred_spans = _bio_spans_from_word_labels(tokens_words, pred_word_labels_fixed)
        gold_spans = _bio_spans_from_word_labels(tokens_words, gold_word_labels)

        pred_spans_by_example.append(
            {
                "paragraph": paragraph_text,
                "tokens": tokens_words,
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
            }
        )

    pred_spans_path = output_dir / "eval_predicted_spans.json"
    with open(pred_spans_path, "w", encoding="utf-8") as f:
        json.dump(pred_spans_by_example, f, indent=2, ensure_ascii=False)

    results = {
        "category": args.category,
        "seqeval_span_f1": float(seqeval_f1),
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
