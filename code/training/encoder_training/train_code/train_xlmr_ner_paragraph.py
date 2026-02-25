import json
import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

from seqeval.metrics import f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score


# ----------------------------
# I/O (BIO JSON: [{"tokens":[...],"labels":[...]}])
# ----------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ner_json(path):
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
# BIO repair (force span starts to be B-*)
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


def apply_bio_fix_to_aligned_pred_ids(pred_ids, aligned_gold, id2label, label2id):
    fixed = pred_ids.copy()

    for i in range(pred_ids.shape[0]):
        idxs = np.where(aligned_gold[i] != -100)[0]
        if idxs.size == 0:
            continue

        word_pred_labels = [id2label[int(pred_ids[i, j])] for j in idxs]
        word_pred_labels = bio_fix_sequence(word_pred_labels)

        for j, lbl in zip(idxs, word_pred_labels):
            fixed[i, j] = int(label2id[lbl])

    return fixed


# ----------------------------
# Tokenization + alignment (LABEL ALL SUBTOKENS FIX)
# ----------------------------
def _continuation_label(word_label):
    word_label = (word_label or "O").strip()
    if word_label.startswith("B-"):
        return "I-" + word_label[2:]
    if word_label.startswith("I-"):
        return word_label
    return "O"


def tokenize_and_align_labels(example, tokenizer, label2id, label_all_subtokens=True):
    max_len = getattr(tokenizer, "model_max_length", 512) or 512
    max_len = min(max_len, 512)

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
            continue

        word_label = example["labels"][word_idx]

        if word_idx != prev_word_idx:
            aligned.append(label2id[word_label])
        else:
            if label_all_subtokens:
                cont = _continuation_label(word_label)
                aligned.append(label2id[cont])
            else:
                aligned.append(-100)

        prev_word_idx = word_idx

    tokenized["labels"] = aligned
    return tokenized


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
    keep = np.array([id2label[int(l)] != "O" for l in flat_labels], dtype=bool)
    return flat_preds[keep], flat_labels[keep]


# ----------------------------
# Metrics
# ----------------------------
def compute_seqeval_f1(pred_ids, label_ids, id2label):
    true_preds, true_labels = [], []

    for p_seq, l_seq in zip(pred_ids, label_ids):
        preds, labs = [], []
        for p, l in zip(p_seq, l_seq):
            if l != -100:
                preds.append(id2label[int(p)])
                labs.append(id2label[int(l)])
        true_preds.append(preds)
        true_labels.append(labs)

    return float(f1_score(true_labels, true_preds))


def token_overlap_f1(pred_ids, label_ids, id2label):
    tp = fp = fn = 0

    for p_seq, l_seq in zip(pred_ids, label_ids):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue

            gold_ent = id2label[int(l)] != "O"
            pred_ent = id2label[int(p)] != "O"

            if pred_ent and gold_ent:
                tp += 1
            elif pred_ent and not gold_ent:
                fp += 1
            elif (not pred_ent) and gold_ent:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def krippendorff_alpha_nominal(rater1, rater2, num_labels):
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
# Sentence-level label from BIO "coverage"
# ----------------------------
def sentence_label_from_bio_coverage(word_labels, threshold=0.5):
    if not word_labels:
        return 0

    n = len(word_labels)
    ent = sum(1 for l in word_labels if (l or "O").strip() != "O")
    coverage = ent / n if n else 0.0
    return 1 if coverage >= threshold else 0


# ----------------------------
# Span extraction (BIO -> spans) for artifacts
# ----------------------------
def word_level_pred_ids_from_aligned(pred_ids_seq, aligned_label_ids_seq):
    return [int(p) for p, l in zip(pred_ids_seq.tolist(), aligned_label_ids_seq.tolist()) if l != -100]


def bio_spans_from_word_labels(tokens, word_labels):
    spans = []
    i = 0
    n = min(len(tokens), len(word_labels))

    while i < n:
        lab = (word_labels[i] or "O").strip()
        if lab == "O":
            i += 1
            continue

        if lab.startswith("B-"):
            ent = lab[2:]
            start = i
            i += 1
            while i < n and (word_labels[i] or "O").strip() == f"I-{ent}":
                i += 1
            end = i
            spans.append({"label": ent, "start_token": start, "end_token": end, "text": " ".join(tokens[start:end])})
            continue

        i += 1

    return spans


# ----------------------------
# NEW: Remove predicted spans shorter than N tokens (word-level)
# ----------------------------
def remove_short_predicted_spans_word_labels(word_labels, min_span_len=4):
    """
    Given word-level BIO labels, drop predicted entity spans shorter than min_span_len
    by converting their tokens to "O".

    This operates AFTER bio_fix_sequence, so spans should begin with B-.
    """
    out = [(l or "O").strip() for l in word_labels]
    n = len(out)
    i = 0

    while i < n:
        lab = out[i]
        if lab == "O":
            i += 1
            continue

        if lab.startswith("B-"):
            ent = lab[2:]
            start = i
            i += 1
            while i < n and out[i] == f"I-{ent}":
                i += 1
            end = i
            span_len = end - start

            if span_len < int(min_span_len):
                for k in range(start, end):
                    out[k] = "O"
            continue

        # If any weird I-* sneaks in, just move on (bio_fix should prevent this)
        i += 1

    return out


def apply_remove_short_spans_to_aligned_pred_ids(pred_ids, aligned_gold, id2label, label2id, min_span_len=4):
    """
    Apply short-span removal at WORD positions only (where aligned_gold != -100),
    then write the updated labels back into pred_ids.

    This keeps your existing alignment scheme and all metrics.
    """
    fixed = pred_ids.copy()

    for i in range(pred_ids.shape[0]):
        idxs = np.where(aligned_gold[i] != -100)[0]
        if idxs.size == 0:
            continue

        word_pred_labels = [id2label[int(pred_ids[i, j])] for j in idxs]
        word_pred_labels = bio_fix_sequence(word_pred_labels)
        word_pred_labels = remove_short_predicted_spans_word_labels(word_pred_labels, min_span_len=min_span_len)

        for j, lbl in zip(idxs, word_pred_labels):
            fixed[i, j] = int(label2id[lbl])

    return fixed


# ----------------------------
# Fix 2: weighted loss to penalize fragmentation
# ----------------------------
def build_label_weights(label_list, o_weight=1.0, b_weight=1.5, i_weight=4.0):
    weights = []
    for lab in label_list:
        if lab == "O":
            weights.append(float(o_weight))
        elif lab.startswith("B-"):
            weights.append(float(b_weight))
        elif lab.startswith("I-"):
            weights.append(float(i_weight))
        else:
            weights.append(1.0)
    return weights


class WeightedTokenTrainer(Trainer):
    def __init__(self, *args, label_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight = self._label_weights
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)

        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--coverage_threshold", type=float, default=0.5)

    parser.add_argument("--label_all_subtokens", action="store_true")
    parser.add_argument("--no_label_all_subtokens", action="store_true")

    parser.add_argument("--o_weight", type=float, default=1.0)
    parser.add_argument("--b_weight", type=float, default=1.5)
    parser.add_argument("--i_weight", type=float, default=3.0)

    # NEW: drop predicted spans shorter than this many WORD tokens (default removes 1-3)
    parser.add_argument("--min_pred_span_len", type=int, default=4)

    args = parser.parse_args()

    label_all_subtokens = True
    if args.no_label_all_subtokens:
        label_all_subtokens = False
    if args.label_all_subtokens:
        label_all_subtokens = True

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

    train_raw = load_ner_json(train_path)
    dev_raw = load_ner_json(dev_path)
    eval_raw = load_ner_json(eval_path)

    print("Applying BIO-fix to GOLD labels (train/dev/eval).")
    train_raw = train_raw.map(lambda ex: {**ex, "labels": bio_fix_sequence(ex["labels"])}, batched=False)
    dev_raw = dev_raw.map(lambda ex: {**ex, "labels": bio_fix_sequence(ex["labels"])}, batched=False)
    eval_raw = eval_raw.map(lambda ex: {**ex, "labels": bio_fix_sequence(ex["labels"])}, batched=False)

    label_set = {lab for ex in train_raw for lab in ex["labels"]}
    label_set.add("O")
    label_list = sorted(label_set)

    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    print("Base model:", base_model)
    print("Label list:", label_list)
    print("label_all_subtokens:", label_all_subtokens)
    print("loss weights:", {"O": args.o_weight, "B-*": args.b_weight, "I-*": args.i_weight})
    print("min_pred_span_len:", int(args.min_pred_span_len))

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_ds = train_raw.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id, label_all_subtokens), batched=False)
    dev_ds = dev_raw.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id, label_all_subtokens), batched=False)
    eval_ds = eval_raw.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id, label_all_subtokens), batched=False)

    def prune(ds, name):
        keep = {"input_ids", "attention_mask", "labels"}
        if "token_type_ids" in ds.column_names:
            keep.add("token_type_ids")
        cols_to_remove = [c for c in ds.column_names if c not in keep]
        if cols_to_remove:
            print(f"[{name}] removing columns: {cols_to_remove}")
            ds = ds.remove_columns(cols_to_remove)
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

    weights = build_label_weights(
        label_list,
        o_weight=args.o_weight,
        b_weight=args.b_weight,
        i_weight=args.i_weight,
    )
    label_weights = torch.tensor(weights, dtype=torch.float32)

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        label_weights=label_weights,
    )

    print(f"{cat_cfg['name']} BIO token-classification training (weighted loss)")
    trainer.train()

    logits, aligned_gold, _ = trainer.predict(eval_ds)
    pred_ids = np.argmax(logits, axis=2)

    print("Applying BIO-fix to PREDICTIONS (forces B-* at span starts).")
    pred_ids = apply_bio_fix_to_aligned_pred_ids(pred_ids, aligned_gold, id2label, label2id)

    # NEW: drop short predicted spans (1-3 words by default)
    if int(args.min_pred_span_len) > 1:
        print(f"Removing predicted spans shorter than {int(args.min_pred_span_len)} word-tokens (set to O).")
        pred_ids = apply_remove_short_spans_to_aligned_pred_ids(
            pred_ids,
            aligned_gold,
            id2label,
            label2id,
            min_span_len=int(args.min_pred_span_len),
        )

    seqeval_span_f1 = compute_seqeval_f1(pred_ids, aligned_gold, id2label)
    overlap = token_overlap_f1(pred_ids, aligned_gold, id2label)

    thr = float(args.coverage_threshold)
    gold_sent_y = []
    pred_sent_y = []

    for raw_ex, pred_seq, gold_seq in zip(eval_raw, pred_ids, aligned_gold):
        gold_word_labels = raw_ex["labels"]

        pred_word_ids = word_level_pred_ids_from_aligned(pred_seq, gold_seq)
        pred_word_labels = [id2label[i] for i in pred_word_ids]
        pred_word_labels = bio_fix_sequence(pred_word_labels)
        pred_word_labels = remove_short_predicted_spans_word_labels(pred_word_labels, min_span_len=int(args.min_pred_span_len))

        gold_sent_y.append(sentence_label_from_bio_coverage(gold_word_labels, threshold=thr))
        pred_sent_y.append(sentence_label_from_bio_coverage(pred_word_labels, threshold=thr))

    gold_sent_y = np.array(gold_sent_y, dtype=int)
    pred_sent_y = np.array(pred_sent_y, dtype=int)

    sent_cm = confusion_matrix(gold_sent_y, pred_sent_y, labels=[0, 1]).tolist()
    sent_p, sent_r, sent_f1, _ = precision_recall_fscore_support(
        gold_sent_y, pred_sent_y, average="binary", zero_division=0
    )
    sent_acc = float((gold_sent_y == pred_sent_y).mean()) if gold_sent_y.size else 0.0

    flat_preds, flat_labels = flatten_valid_tokens(pred_ids, aligned_gold)

    cm_all = confusion_matrix(flat_labels, flat_preds, labels=list(range(len(label_list))))
    cm_all_dict = {"labels": label_list, "matrix": cm_all.tolist()}
    alpha_all = krippendorff_alpha_nominal(flat_labels, flat_preds, num_labels=len(label_list))
    kappa_all = float(cohen_kappa_score(flat_labels, flat_preds, labels=list(range(len(label_list))))) if flat_labels.size else 0.0
    p_micro_all, r_micro_all, f1_micro_all, _ = precision_recall_fscore_support(flat_labels, flat_preds, average="micro", zero_division=0)
    p_macro_all, r_macro_all, f1_macro_all, _ = precision_recall_fscore_support(flat_labels, flat_preds, average="macro", zero_division=0)

    flat_preds_ent, flat_labels_ent = filter_out_gold_O(flat_preds, flat_labels, id2label)
    cm_ent = confusion_matrix(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list))))
    cm_ent_dict = {"labels": label_list, "matrix": cm_ent.tolist()}
    alpha_ent = krippendorff_alpha_nominal(flat_labels_ent, flat_preds_ent, num_labels=len(label_list))
    kappa_ent = float(cohen_kappa_score(flat_labels_ent, flat_preds_ent, labels=list(range(len(label_list))))) if flat_labels_ent.size else 0.0
    p_micro_ent, r_micro_ent, f1_micro_ent, _ = precision_recall_fscore_support(flat_labels_ent, flat_preds_ent, average="micro", zero_division=0)
    p_macro_ent, r_macro_ent, f1_macro_ent, _ = precision_recall_fscore_support(flat_labels_ent, flat_preds_ent, average="macro", zero_division=0)

    pred_spans_by_example = []
    for raw_ex, pred_seq, gold_seq in zip(eval_raw, pred_ids, aligned_gold):
        tokens_words = raw_ex["tokens"]
        gold_word_labels = raw_ex["labels"]

        pred_word_ids = word_level_pred_ids_from_aligned(pred_seq, gold_seq)
        pred_word_labels = [id2label[i] for i in pred_word_ids]
        pred_word_labels = bio_fix_sequence(pred_word_labels)
        pred_word_labels = remove_short_predicted_spans_word_labels(pred_word_labels, min_span_len=int(args.min_pred_span_len))

        pred_spans = bio_spans_from_word_labels(tokens_words, pred_word_labels)
        gold_spans = bio_spans_from_word_labels(tokens_words, gold_word_labels)

        pred_spans_by_example.append(
            {
                "doc_index": raw_ex.get("doc_index"),
                "sample_index": raw_ex.get("sample_index"),
                "text": raw_ex.get("text", " ".join(tokens_words)),
                "gold_spans": gold_spans,
                "pred_spans": pred_spans,
                "coverage_threshold": thr,
                "min_pred_span_len": int(args.min_pred_span_len),
                "gold_sentence_label": int(sentence_label_from_bio_coverage(gold_word_labels, threshold=thr)),
                "pred_sentence_label": int(sentence_label_from_bio_coverage(pred_word_labels, threshold=thr)),
            }
        )

    pred_spans_path = output_dir / "eval_predicted_spans.json"
    with open(pred_spans_path, "w", encoding="utf-8") as f:
        json.dump(pred_spans_by_example, f, indent=2, ensure_ascii=False)

    results = {
        "category": args.category,
        "base_model": base_model,
        "coverage_threshold": thr,
        "min_pred_span_len": int(args.min_pred_span_len),
        "label_all_subtokens": bool(label_all_subtokens),
        "loss_weights": {"o_weight": args.o_weight, "b_weight": args.b_weight, "i_weight": args.i_weight},
        "seqeval_span_f1": float(seqeval_span_f1),
        "token_overlap_f1_entity_vs_O": overlap,
        "sentence_label_from_bio_coverage": {
            "confusion_matrix_labels": [0, 1],
            "confusion_matrix": sent_cm,
            "precision": float(sent_p),
            "recall": float(sent_r),
            "f1": float(sent_f1),
            "accuracy": float(sent_acc),
            "num_sentences": int(gold_sent_y.size),
        },
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
        "artifacts": {"eval_predicted_spans_path": str(pred_spans_path)},
    }

    print("\n-----FINAL EVAL RESULTS-----")
    print(json.dumps(results, indent=2))

    with open(output_dir / "final_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()