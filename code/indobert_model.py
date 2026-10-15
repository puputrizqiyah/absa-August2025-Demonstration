"""Fine-tuned encoder (IndoBERT) under the identical folds of the benchmark.

The script reads the fold assignment written by `run_benchmark.py`, so the encoder
is trained and evaluated on exactly the same splits as every lexical model, which
is what makes the paired contrasts of Table 9 legitimate.

Configuration (Section 3): indobenchmark/indobert-base-p1, max length 64, batch 32,
2 epochs, AdamW with weight decay 0.01, peak learning rate 3e-5, one-cycle schedule
with 10% warm-up, gradient clipping 1.0, seed 42.

    python code/run_benchmark.py --tasks relevance --schemes stratified   # writes the folds
    python code/indobert_model.py --tasks relevance aspect polarity
    python code/indobert_model.py --tasks polarity --fold 1               # smoke test

Requires the optional extra: pip install -r requirements-indobert.txt
On the machine that produced the paper this took about 2.2 hours for three tasks
(Apple MPS, ~1.16 s per optimisation step at batch 32); expect a similar order of
magnitude on other hardware without CUDA.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from config import INDOBERT, LABELS, PREDICTIONS_DIR, RESULTS_DIR, SEED
from data_io import load_gold_standard, task_frame
from metrics import pooled_scores, per_class


def read_folds(task, scheme="stratified"):
    path = RESULTS_DIR / "folds" / f"{scheme}_{task}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found; run run_benchmark.py first so that the "
                         f"encoder uses the identical folds")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def set_seed(seed=SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode(tokenizer, texts, max_len):
    return tokenizer(list(texts), padding=True, truncation=True, max_length=max_len,
                     return_tensors="pt")


def train_fold(train_texts, train_labels, test_texts, test_labels, device, num_labels, fold=None):
    """One fine-tuning run; returns (predictions, accuracy, macro-F1)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    set_seed(INDOBERT["seed"])
    tokenizer = AutoTokenizer.from_pretrained(INDOBERT["model"])
    enc = encode(tokenizer, train_texts, INDOBERT["max_len"])
    label_ids = torch.tensor([LABELS_INDEX[num_labels][l] for l in train_labels])
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], label_ids)
    loader = DataLoader(dataset, batch_size=INDOBERT["batch_size"], shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        INDOBERT["model"], num_labels=num_labels).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=INDOBERT["lr"],
                                  weight_decay=INDOBERT["weight_decay"])
    steps = len(loader) * INDOBERT["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimiser, int(INDOBERT["warmup_ratio"] * steps), steps)

    model.train()
    for epoch in range(INDOBERT["epochs"]):
        for input_ids, attention_mask, y in loader:
            input_ids, attention_mask, y = (input_ids.to(device), attention_mask.to(device), y.to(device))
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), INDOBERT["grad_clip"])
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

    model.eval()
    test_enc = encode(tokenizer, test_texts, INDOBERT["max_len"])
    with torch.no_grad():
        logits = model(input_ids=test_enc["input_ids"].to(device),
                       attention_mask=test_enc["attention_mask"].to(device)).logits
    pred_ids = logits.argmax(-1).cpu().tolist()
    index_to_label = {v: k for k, v in LABELS_INDEX[num_labels].items()}
    preds = [index_to_label[i] for i in pred_ids]
    score = pooled_scores(list(test_labels), preds)
    return preds, score


LABELS_INDEX = {}


def build_label_index(task):
    labels = LABELS[task]
    LABELS_INDEX[len(labels)] = {label: i for i, label in enumerate(labels)}
    return len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["relevance", "aspect", "polarity"])
    ap.add_argument("--scheme", default="stratified")
    ap.add_argument("--fold", type=int, default=None, help="only this fold (smoke test)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "indobert_results.jsonl"))
    args = ap.parse_args()

    import torch
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    records = load_gold_standard()
    out_rows = []
    for task in args.tasks:
        subset, gold = task_frame(records, task)
        texts = [r["text"] for r in subset]
        folds = read_folds(task, args.scheme)
        n_labels = build_label_index(task)
        if len(folds) != len(gold):
            raise SystemExit(f"fold file has {len(folds)} rows for {len(gold)} records")
        bad = [i for i, r in enumerate(folds)
               if r.get("gold") is not None
               and str(r["gold"]).strip().lower() != str(gold[i]).strip().lower()]
        if bad:
            raise SystemExit(f"fold file does not line up with the task frame ({len(bad)} rows); "
                             f"rerun run_benchmark.py so the encoder uses the identical folds")
        fold_ids = sorted({int(r["fold"]) for r in folds})
        if args.fold is not None:
            fold_ids = [args.fold]

        oof = np.empty(len(gold), dtype=object)
        for f in fold_ids:
            test_idx = [i for i, r in enumerate(folds) if int(r["fold"]) == f]
            train_idx = [i for i, r in enumerate(folds) if int(r["fold"]) != f]
            preds, score = train_fold([texts[i] for i in train_idx], [gold[i] for i in train_idx],
                                      [texts[i] for i in test_idx], [gold[i] for i in test_idx],
                                      device, n_labels, fold=f)
            for i, p in zip(test_idx, preds):
                oof[i] = p
            print(f"   {task} fold {f}: n={len(test_idx)} acc={score['accuracy']:.4f} "
                  f"macro-F1={score['macro_f1']:.4f}")
            out_rows.append(dict(task=task, model=INDOBERT["model"], fold=f,
                                 n_fold=len(test_idx), acc=score["accuracy"],
                                 macro_f1=score["macro_f1"], seed=INDOBERT["seed"],
                                 epochs=INDOBERT["epochs"], maxlen=INDOBERT["max_len"],
                                 batch=INDOBERT["batch_size"]))

        if args.fold is not None:
            print(f"   {task}: smoke test on fold {args.fold} only; pooled metrics and the "
                  f"prediction file are written only by the full run")
            continue
        pooled = pooled_scores(list(gold), list(oof))
        out_rows.append(dict(task=task, model=INDOBERT["model"], n=len(gold),
                             acc=pooled["accuracy"], macro_f1=pooled["macro_f1"],
                             classes=per_class(list(gold), list(oof), LABELS[task]),
                             seed=INDOBERT["seed"], epochs=INDOBERT["epochs"],
                             maxlen=INDOBERT["max_len"], batch=INDOBERT["batch_size"],
                             folds=len(fold_ids)))
        path = PREDICTIONS_DIR / f"{args.scheme}_{task}_indobert.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "index", "gold", "predicted", "correct", "fold"])
            for i, r in enumerate(subset):
                f = int(folds[i]["fold"])
                w.writerow([i, r["index"], gold[i], oof[i], int(oof[i] == gold[i]), f])
        print(f"   {task}: pooled macro-F1 {pooled['macro_f1']:.4f} -> {path.name}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
