from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pathlib import Path
import spacy
import re
from typing import List, Dict, Any, Optional
import argparse
import json 
import numpy as np
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm 

unsupp_dir = "code/training/encoder_training/scibert/runs/4_unsupported_short_scibert/checkpoint-21"
tokenizer = AutoTokenizer.from_pretrained(unsupp_dir, local_files_only=True, model_max_length=256)
model = AutoModelForTokenClassification.from_pretrained(unsupp_dir, local_files_only=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def predict_subwords(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    # ✅ ensure inputs are on same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs)

    pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()
    input_ids = inputs["input_ids"][0].tolist()

    toks = tokenizer.convert_ids_to_tokens(input_ids)
    labs = [model.config.id2label[int(i)] for i in pred_ids]

    out = []
    for tok, lab in zip(toks, labs):
        if tok in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token):
            continue
        out.append((tok, lab))
    return out


def merge_wordpiece_span(tokens):
    """
    Merge a list of WordPiece tokens into a single string.
    Example: ['conv', '##ai'] -> 'convai'
    """
    s = ""
    for t in tokens:
        if t.startswith("##"):
            s += t[2:]
        else:
            # add a space only if this isn't the first token AND the previous merge ended a word
            if s != "":
                # if you prefer no spaces between separate words in a span, remove this block
                s += " "
            s += t
    return s


citation_pattern = re.compile(
r"""
(
    # (Chen et al., 2021b) OR (Chen, 2021b) OR (Chen & Smith, 2021b) OR (Chen and Smith, 2021b)
    \(
        [A-Z][A-Za-z\-]+
        (?:
            (?:\s+et\ al\.)                       # allow "et al." alone
            |
            (?:\s+(?:&|and)\s+[A-Z][A-Za-z\-]+)   # or "& Smith" / "and Smith"
        )?
        \,?\s+\d{4}[a-z]?
    \)
    |
    # Chen et al. (2021b) OR Chen & Smith (2021b) OR Chen (2021b)
    [A-Z][A-Za-z\-]+
    (?:
        (?:\s+et\ al\.) 
        |
        (?:\s+(?:&|and)\s+[A-Z][A-Za-z\-]+)
    )?
    \s+\(\d{4}[a-z]?\)
    |
    # [12] or [12, 15-18]
    \[(\d+([,\-\u2013]\s*\d+)*)\]
)
""",
re.VERBOSE,
)


def citation_after_mention(
    sentence: str,
    span: str,
    *,
    window_chars: int = 200,          # window AFTER the mention
    pre_window_chars: int = 50,       # window BEFORE the mention
    case_insensitive: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Returns None if span not in sentence.

    Citation is considered present if it appears:
      - immediately after the mention, OR
      - anywhere within window_chars AFTER, OR
      - anywhere within pre_window_chars BEFORE.
    """
    if not span:
        return None

    flags = re.IGNORECASE if case_insensitive else 0
    m = re.search(re.escape(span), sentence, flags)
    if not m:
        return None

    span_start, span_end = m.start(), m.end()

    # --- AFTER window (covers "immediately after") ---
    after = sentence[span_end : span_end + max(0, int(window_chars))]
    cit_after = citation_pattern.search(after)
    if cit_after:
        where = "immediately_after" if cit_after.start() == 0 else "after"
        return {
            "span": span,
            "sentence": sentence,
            "sentence_index": None,
            "mention_start": span_start,
            "mention_end": span_end,
            "citation_found": True,
            "citation_where": where,
            "citation_match": cit_after.group(0),
        }

    # --- BEFORE window (anywhere near, not comma-dependent) ---
    pre_w = max(0, int(pre_window_chars))
    before_start = max(0, span_start - pre_w)
    before = sentence[before_start:span_start]
    cit_before = None
    for mm in citation_pattern.finditer(before):
        cit_before = mm  # keep the closest (last) one

    if cit_before:
        return {
            "span": span,
            "sentence": sentence,
            "sentence_index": None,
            "mention_start": span_start,
            "mention_end": span_end,
            "citation_found": True,
            "citation_where": "before",
            "citation_match": cit_before.group(0),
        }

    # --- no citation near mention ---
    return {
        "span": span,
        "sentence": sentence,
        "sentence_index": None,
        "mention_start": span_start,
        "mention_end": span_end,
        "citation_found": False,
        "citation_where": None,
        "citation_match": None,
    }


def find_first_mentions_missing_citation(
    sentences: List[str],
    spans: List[str],
    *,
    window_chars: int = 200,
    pre_window_chars: int = 80,
    case_insensitive: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each span, locate its FIRST mention across sentences (by order),
    then check whether a citation appears after the mention OR before-with-comma.
    Returns ONLY the ones where citation is missing.
    """
    missing = []

    for span in spans:
        first_result = None
        first_sent_idx = None

        for i, sent in enumerate(sentences):
            res = citation_after_mention(
                sent,
                span,
                window_chars=window_chars,
                pre_window_chars=pre_window_chars,
                case_insensitive=case_insensitive,
            )
            if res is not None:
                first_result = res
                first_sent_idx = i
                break

        if first_result is None:
            continue

        first_result["sentence_index"] = first_sent_idx

        if not first_result["citation_found"]:
            missing.append(first_result)

    return missing

def extract_gold_spans_from_bio(tokens, labels):
    spans = []
    i = 0
    n = len(labels)

    while i < n:
        label = labels[i]

        if label.startswith("B-"):
            entity_type = label[2:]
            start = i
            i += 1

            # collect I-* continuation
            while i < n and labels[i] == f"I-{entity_type}":
                i += 1

            end = i  # exclusive
            span_text = " ".join(tokens[start:end])

            spans.append({
                "label": entity_type,
                "start_token": start,
                "end_token": end,
                "text": span_text,
            })
        else:
            i += 1

    return spans

# ----------------------------
# Gold span extraction (BIO -> spans)
# ----------------------------
def extract_gold_spans_from_bio(tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
    spans = []
    i = 0
    n = len(labels)

    while i < n:
        lab = (labels[i] or "O").strip()
        if lab.startswith("B-"):
            ent = lab[2:]
            start = i
            i += 1
            while i < n and (labels[i] or "O").strip() == f"I-{ent}":
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
        else:
            i += 1
    return spans


# ----------------------------
# Token offsets from " ".join(tokens)
# ----------------------------
def token_char_offsets_from_space_join(tokens):
    """
    If you form text = " ".join(tokens), compute each token's (start,end) char offsets in that text.
    """
    offsets = []
    pos = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            pos += 1  # the space
        start = pos
        end = start + len(tok)
        offsets.append((start, end))
        pos = end
    return offsets


# ----------------------------
# Predict word-level binary ENTITY/O from a token-classification model
# ----------------------------
@torch.inference_mode()
def predict_word_binary(
    tokens,
    tokenizer,
    model,
    max_length: int = 256,
):
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    outputs = model(**{k: v.to(model.device) for k, v in enc.items()})
    pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()

    word_ids = enc.word_ids(batch_index=0)

    # For each word index, take the first subtoken prediction
    word_pred_label_ids = [None] * len(tokens)
    seen = set()

    for tok_i, widx in enumerate(word_ids):
        if widx is None:
            continue
        if widx in seen:
            continue
        seen.add(widx)
        word_pred_label_ids[widx] = int(pred_ids[tok_i])

    # Words that got truncated may remain None -> treat as O (0)
    out = []
    for widx in range(len(tokens)):
        lid = word_pred_label_ids[widx]
        if lid is None:
            out.append(0)
        else:
            lab = model.config.id2label[int(lid)]
            out.append(0 if lab == "O" else 1)
    return out


# ----------------------------
# Convert binary labels to spans (contiguous 1s)
# ----------------------------
def spans_from_binary(tokens, binary_labels):
    spans = []
    i = 0
    n = min(len(tokens), len(binary_labels))

    while i < n:
        if binary_labels[i] != 1:
            i += 1
            continue
        start = i
        i += 1
        while i < n and binary_labels[i] == 1:
            i += 1
        end = i
        spans.append(
            {
                "label": "ENTITY",
                "start_token": start,
                "end_token": end,
                "text": " ".join(tokens[start:end]),
            }
        )
    return spans


def sentence_for_span(doc, span_char_start, span_char_end):
    for sent in doc.sents:
        if sent.start_char < span_char_end and sent.end_char > span_char_start:
            return sent.text.strip()
    return doc.text.strip()


# ----------------------------
# Metrics: token overlap F1 (binary) + Cohen kappa (binary)
# ----------------------------
def token_overlap_f1_binary(y_pred, y_true):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn}


def prune_isolated_ones(binary_labels: List[int]) -> List[int]:
    """
    Keep a 1 only if it has a neighboring 1 (left or right).
    """
    n = len(binary_labels)
    out = binary_labels[:]  # copy
    for i in range(n):
        if out[i] != 1:
            continue
        left = (i > 0 and out[i - 1] == 1)
        right = (i < n - 1 and out[i + 1] == 1)
        if not (left or right):
            out[i] = 0
    return out


def dynamic_windows(
    sentence: str,
    *,
    after_frac: float = 0.25,   # 25% of sentence length after the mention
    before_frac: float = 0.12,  # 12% of sentence length before the mention
    min_after: int = 40,
    max_after: int = 140,
    min_before: int = 20,
    max_before: int = 80,
):
    n = len(sentence)
    after = int(n * after_frac)
    before = int(n * before_frac)

    after = max(min_after, min(after, max_after))
    before = max(min_before, min(before, max_before))
    return after, before

@torch.inference_mode()
def predict_word_probs(
    tokens: List[str],
    tokenizer,
    model,
    max_length: int = 256,
):
    """
    Returns:
      - p_entity_per_word: List[float] length=len(tokens), probability of ENTITY for each word
      - pred_label_id_per_word: argmax label id for each word (0/1), mainly for debugging

    Uses first-subtoken for each word (same alignment style as your current code).
    """
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    enc_dev = {k: v.to(model.device) for k, v in enc.items()}
    outputs = model(**enc_dev)  # logits: [1, seq_len, num_labels]

    logits = outputs.logits[0]  # [seq_len, 2]
    probs = torch.softmax(logits, dim=-1)  # [seq_len, 2]
    p_entity_tok = probs[:, 1].detach().cpu().numpy()  # prob of label id 1 (ENTITY)

    pred_ids = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
    word_ids = enc.word_ids(batch_index=0)

    p_entity_per_word = [0.0] * len(tokens)
    pred_id_per_word = [0] * len(tokens)
    seen = set()

    for tok_i, widx in enumerate(word_ids):
        if widx is None or widx in seen:
            continue
        seen.add(widx)
        p_entity_per_word[widx] = float(p_entity_tok[tok_i])
        pred_id_per_word[widx] = int(pred_ids[tok_i])

    return p_entity_per_word, pred_id_per_word

from typing import List

def binarize_with_threshold(p_entity: List[float], thresh: float) -> List[int]:
    return [1 if p >= thresh else 0 for p in p_entity]

def prune_isolated_with_exception(
    binary: List[int],
    p_entity: List[float],
    *,
    singleton_keep_thresh: float = 0.95,
) -> List[int]:
    """
    If binary[i]==1 but has no ENTITY neighbor, set to 0 unless p_entity[i] >= singleton_keep_thresh.
    """
    n = len(binary)
    out = binary[:]
    for i in range(n):
        if out[i] != 1:
            continue
        left = (i > 0 and out[i - 1] == 1)
        right = (i < n - 1 and out[i + 1] == 1)
        if not (left or right):
            if p_entity[i] < singleton_keep_thresh:
                out[i] = 0
    return out


def main():
    eval_path = "code/training/data/unsupported_claim/unsupported_claim_ner_eval_short_1_5w.json"
    out_dir = Path("code/training/encoder_training/scibert/runs/4_unsupported_short_scibert")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- thresholds to tune ----
    THRESH = 0.55                 # base threshold for ENTITY
    SINGLETON_KEEP_THRESH = 0.95  # keep isolated ENTITY only if very confident

    # ---- citation window (you can keep dynamic windows too, but here fixed) ----
    WINDOW_CHARS = 120
    PRE_WINDOW_CHARS = 60

    nlp = spacy.load("en_core_web_sm")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    all_gold_bin = []
    all_final_pred_bin = []
    analysis_rows = []

    for ex_idx, example in tqdm(enumerate(eval_data), total=len(eval_data), desc="Evaluating"):
        tokens = example["tokens"]
        gold_labels = example["labels"]

        sent_text = " ".join(tokens)
        _ = nlp(sent_text)  # you wanted sentence-based usage; eval item is already a sentence

        gold_bin = [0 if (str(l).strip() == "O") else 1 for l in gold_labels]

        # --- 1) get probabilities per word ---
        p_entity, _pred_id_dbg = predict_word_probs(tokens, tokenizer, model, max_length=256)

        # --- 2) threshold to binary ---
        raw_bin = binarize_with_threshold(p_entity, THRESH)

        # align lengths (truncation safety)
        L = min(len(tokens), len(gold_bin), len(raw_bin), len(p_entity))
        tokens_L = tokens[:L]
        gold_bin_L = gold_bin[:L]
        raw_bin_L = raw_bin[:L]
        p_entity_L = p_entity[:L]

        # --- 3) prune isolated with high-confidence exception ---
        pruned_bin = prune_isolated_with_exception(
            raw_bin_L,
            p_entity_L,
            singleton_keep_thresh=SINGLETON_KEEP_THRESH,
        )

        # --- 4) spans from pruned predictions ---
        raw_pred_spans = spans_from_binary(tokens_L, pruned_bin)

        # --- 5) citation filtering: discard spans with citation near mention ---
        kept_spans = []
        removed_spans = []

        for sp in raw_pred_spans:
            span_text = sp["text"]
            win_after, win_before = dynamic_windows(sent_text)
            cit_res = citation_after_mention(
                sent_text,
                span_text,
                window_chars=win_after,
                pre_window_chars=win_before,
                case_insensitive=True,
            )

            # If we can't find the mention, discard (prevents accidental keeps)
            if cit_res is None:
                removed_spans.append(
                    {
                        **sp,
                        "citation_found": None,
                        "citation_where": "span_not_found_in_sentence_text",
                        "citation_match": None,
                    }
                )
                continue

            if cit_res["citation_found"]:
                removed_spans.append(
                    {
                        **sp,
                        "citation_found": True,
                        "citation_where": cit_res["citation_where"],
                        "citation_match": cit_res["citation_match"],
                        "mention_start": cit_res["mention_start"],
                        "mention_end": cit_res["mention_end"],
                    }
                )
            else:
                kept_spans.append(
                    {
                        **sp,
                        "citation_found": False,
                        "citation_where": None,
                        "citation_match": None,
                        "mention_start": cit_res["mention_start"],
                        "mention_end": cit_res["mention_end"],
                    }
                )

        # --- 6) final token predictions from kept spans ---
        final_pred = [0] * L
        for sp in kept_spans:
            s = max(0, int(sp["start_token"]))
            e = min(L, int(sp["end_token"]))
            for i in range(s, e):
                final_pred[i] = 1

        all_gold_bin.extend(gold_bin_L)
        all_final_pred_bin.extend(final_pred)

        analysis_rows.append(
            {
                "example_index": ex_idx,
                "sentence": sent_text,
                "tokens": tokens_L,
                "gold_labels": gold_labels[:L],
                "gold_binary": gold_bin_L,
                "p_entity": p_entity_L,
                "raw_pred_binary_thresh": raw_bin_L,
                "pred_binary_after_neighbor_prune": pruned_bin,
                "final_pred_binary": final_pred,
                "raw_pred_spans_after_prune": raw_pred_spans,
                "kept_spans_no_citation": kept_spans,
                "removed_spans_with_citation": removed_spans,
                "params": {
                    "THRESH": THRESH,
                    "SINGLETON_KEEP_THRESH": SINGLETON_KEEP_THRESH,
                    "WINDOW_CHARS": WINDOW_CHARS,
                    "PRE_WINDOW_CHARS": PRE_WINDOW_CHARS,
                },
            }
        )

    y_true = np.array(all_gold_bin, dtype=np.int64)
    y_pred = np.array(all_final_pred_bin, dtype=np.int64)

    overlap = token_overlap_f1_binary(y_pred, y_true)
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=[0, 1])) if y_true.size else 0.0

    report = {
        "num_examples": int(len(eval_data)),
        "num_tokens_scored": int(y_true.size),
        "token_overlap_f1_binary_FINAL": overlap,
        "cohens_kappa_binary_FINAL": float(kappa),
        "thresholds": {
            "THRESH": THRESH,
            "SINGLETON_KEEP_THRESH": SINGLETON_KEEP_THRESH,
        },
        "citation_windows": {
            "WINDOW_CHARS": WINDOW_CHARS,
            "PRE_WINDOW_CHARS": PRE_WINDOW_CHARS,
        },
    }

    with open(out_dir / "eval_final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(out_dir / "eval_final_predictions_with_gold_and_sentence.json", "w", encoding="utf-8") as f:
        json.dump(analysis_rows, f, indent=2, ensure_ascii=False)

    print("\n--- FINAL EVAL REPORT ---")
    print(json.dumps(report, indent=2))
    print(f"\nSaved:\n  - {out_dir / 'eval_final_report.json'}\n  - {out_dir / 'eval_final_predictions_with_gold_and_sentence.json'}")


if __name__ == "__main__":
    main()