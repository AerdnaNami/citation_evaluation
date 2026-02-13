import json
import argparse
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

from seqeval.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# ----------------------------
# Config Loader
# ----------------------------
def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


# ----------------------------
# Load JSON NER data
# ----------------------------
def load_ner_json(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


# ----------------------------
# Tokenize and align labels
# ----------------------------
def tokenize_and_align_labels(example, tokenizer, label2id):
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
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
                flat_preds.append(p)
                flat_labels.append(l)

    return np.array(flat_preds), np.array(flat_labels)


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
    dev_path = cat_cfg["dev_path"]
    eval_path = cat_cfg["eval_path"]

    output_root = Path(cfg.get("output_root", "./runs"))
    run_name = cat_cfg.get("run_name", args.category)
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_ner_json(train_path)
    dev_dataset = load_ner_json(dev_path)
    eval_dataset = load_ner_json(eval_path)

    # Build label set
    label_list = sorted(
        list({label for example in train_dataset for label in example["labels"]})
    )

    global label2id, id2label
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=False,
    )
    dev_dataset = dev_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=False,
    )
    eval_dataset = eval_dataset.map(
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
        logging_steps=defaults.get("logging_steps", 50),
        learning_rate=defaults.get("learning_rate", 3e-5),
        per_device_train_batch_size=defaults.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=defaults.get("per_device_eval_batch_size", 16),
        num_train_epochs=defaults.get("num_train_epochs", 3),
        weight_decay=defaults.get("weight_decay", 0.01),
        warmup_steps=defaults.get("warmup_steps", 0.1),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # ----------------------------
    # FINAL EVALUATION ON EVAL SET
    # ----------------------------
    predictions, labels, _ = trainer.predict(eval_dataset)
    predictions = np.argmax(predictions, axis=2)

    seqeval_f1 = compute_seqeval_f1(predictions, labels)

    flat_preds, flat_labels = flatten_valid_tokens(predictions, labels)

    cm = confusion_matrix(flat_labels, flat_preds)
    cm_dict = {
        "labels": label_list,
        "matrix": cm.tolist()
    }

    kappa = cohen_kappa_score(flat_labels, flat_preds)

    # Pairwise F1
    p1, r1, f1_1, _ = precision_recall_fscore_support(
        flat_labels, flat_preds, average="micro"
    )
    p2, r2, f1_2, _ = precision_recall_fscore_support(
        flat_preds, flat_labels, average="micro"
    )

    tp = np.sum(flat_preds == flat_labels)
    fp = np.sum(flat_preds != flat_labels)
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
        "seqeval_span_f1": float(seqeval_f1),
        "cohen_kappa": float(kappa),
        "pairwise_f1": {
            "gold_to_pred": {"precision": float(p1), "recall": float(r1), "f1": float(f1_1)},
            "pred_to_gold": {"precision": float(p2), "recall": float(r2), "f1": float(f1_2)},
            "pairwise_micro_f1": float(micro_f1),
        },
        "confusion_matrix": cm_dict,
    }

    print("\nFINAL EVAL RESULTS")
    print(json.dumps(results, indent=2))

    with open(output_dir / "final_eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
