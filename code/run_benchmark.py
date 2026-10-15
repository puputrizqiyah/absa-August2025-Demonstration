"""Run the benchmark and save the out-of-fold predictions behind every table.

This is the single entry point that produces the artifacts demanded by the
Data/Code Availability statements:

    results/predictions/<scheme>_<task>_<model>.csv   per-record out-of-fold
                                                      predictions (item #10)
    results/folds/<scheme>_<task>.csv                 the fold each record was
                                                      held out in
    results/tables/table3_benchmark.csv               macro-F1, fold SD, accuracy
    results/tables/table4_per_class.csv               per-class precision/recall/F1
    results/tables/table10_robustness.csv             stratified / grouped / temporal
    results/run_manifest.json                         environment and configuration

Examples
--------
    python code/run_benchmark.py                                  # everything lexical
    python code/run_benchmark.py --tasks relevance --schemes stratified
    python code/run_benchmark.py --models tfidf_lr char_lr --no-tables
    python code/run_benchmark.py --report-vocabulary
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from config import (DATA_DIR, DERIVED_DIR, LABELS, PREDICTIONS_DIR, RESULTS_DIR,
                    TABLES_DIR, TASK_SCOPE, N_FOLDS)
from data_io import load_gold_standard, task_frame, task_indices
from folds import day_held_out_folds, grouped_folds, stratified_folds
from lexical_models import MODEL_FACTORIES, MODEL_NAMES, vocabulary_report
from metrics import fold_stats, per_class, pooled_scores

SCHEMES = ("stratified", "grouped", "day_held_out")


def load_groups(records):
    """Near-duplicate cluster ids, aligned with the record list."""
    path = DERIVED_DIR / "near_duplicate_clusters.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found; run: python code/neardup.py")
    mapping = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mapping[int(row["index"])] = int(row["cluster_id"])
    return [mapping[r["index"]] for r in records]


def load_days(records):
    """Collection day (WIB) per record, aligned with the record list."""
    path = DERIVED_DIR / "day_partition.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found; run: python code/day_partition.py")
    mapping = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mapping[int(row["index"])] = row["day"]
    return [mapping[r["index"]] for r in records]


def make_folds(scheme, y, records, subset_idx):
    """Return (folds, fold_labels) where fold_labels[i] is the fold of record i."""
    if scheme == "stratified":
        folds = stratified_folds(y)
        names = [f"fold{k}" for k in range(len(folds))]
    elif scheme == "grouped":
        groups = load_groups(records)
        folds = grouped_folds(y, [groups[i] for i in subset_idx])
        names = [f"fold{k}" for k in range(len(folds))]
    elif scheme == "day_held_out":
        days = load_days(records)
        folds = day_held_out_folds([days[i] for i in subset_idx])
        names = [f[2] for f in folds]
        folds = [(tr, te) for tr, te, _ in folds]
    else:
        raise ValueError(scheme)

    fold_of = np.full(len(y), -1, dtype=int)
    for k, (_, test_idx) in enumerate(folds):
        fold_of[test_idx] = k
    if (fold_of < 0).any():
        raise AssertionError("some records were never held out")
    return folds, fold_of, names


def run_one(model_key, task, texts, y, folds, fold_of):
    """Fit the model once per fold; return per-record predictions and fold scores."""
    factory = MODEL_FACTORIES[model_key]
    preds = np.empty(len(y), dtype=object)
    fold_scores, fold_acc = [], []
    for train_idx, test_idx in folds:
        model = factory()
        model.fit([texts[i] for i in train_idx], [y[i] for i in train_idx])
        p = model.predict([texts[i] for i in test_idx])
        y_true = [y[i] for i in test_idx]
        for i, label in zip(test_idx, p):
            preds[i] = label
        scored = pooled_scores(y_true, list(p))
        fold_scores.append(scored["macro_f1"])
        fold_acc.append(scored["accuracy"])
    macro = fold_stats(fold_scores)
    macro["pooled"] = pooled_scores(list(y), list(preds))["macro_f1"]
    acc = fold_stats(fold_acc)
    acc["pooled"] = pooled_scores(list(y), list(preds))["accuracy"]
    return dict(preds=preds, macro_f1=macro, accuracy=acc,
                classes=per_class(list(y), list(preds), LABELS[task]))


def write_predictions(path, records, subset_idx, y, fold_of, preds, task):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "index", "gold", "predicted", "correct", "fold"])
        for pos, i in enumerate(subset_idx):
            w.writerow([pos, records[i]["index"], y[pos], preds[pos],
                        int(preds[pos] == y[pos]), int(fold_of[pos])])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=["relevance", "aspect", "polarity"])
    ap.add_argument("--schemes", nargs="*", default=["stratified"])
    ap.add_argument("--models", nargs="*", default=list(MODEL_FACTORIES))
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--report-vocabulary", action="store_true",
                    help="print the feature counts of Section 3 and exit")
    args = ap.parse_args()

    records = load_gold_standard()
    print(f"gold standard: {len(records)} records, "
          f"{sum(1 for r in records if r['relevant_gold'] is True)} relevant")

    if args.report_vocabulary:
        all_texts = [r["text"] for r in records]
        rel_texts = [r["text"] for r in records if r["relevant_gold"] is True]
        for key, row in vocabulary_report(all_texts, rel_texts).items():
            mark = "ok" if row["matches"] else "MISMATCH"
            print(f"  {key:22s} observed={row['observed']:6d} expected={row['expected']} {mark}")
        return

    manifest = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"),
                     python=sys.version.split()[0], platform=platform.platform(),
                     records=len(records), n_folds=N_FOLDS,
                     tasks={t: dict(scope=TASK_SCOPE[t], n=len(task_frame(records, t)[0]))
                            for t in args.tasks},
                     schemes=args.schemes, models=args.models,
                     results=[])
    rows_bench, rows_class = [], []

    for scheme in args.schemes:
        for task in args.tasks:
            subset, y = task_frame(records, task)
            subset_idx = task_indices(records, task)
            texts = [r["text"] for r in subset]
            folds, fold_of, fold_names = make_folds(scheme, y, records, subset_idx)
            fold_path = RESULTS_DIR / "folds" / f"{scheme}_{task}.csv"
            fold_path.parent.mkdir(parents=True, exist_ok=True)
            with fold_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["row", "index", "fold", "fold_name", "gold"])
                for pos, i in enumerate(subset_idx):
                    w.writerow([pos, records[i]["index"], int(fold_of[pos]),
                                fold_names[int(fold_of[pos])], y[pos]])
            print(f"\n[{scheme}/{task}] n={len(y)} folds={len(folds)} -> {fold_path.name}")

            for model_key in args.models:
                t0 = time.time()
                res = run_one(model_key, task, texts, y, folds, fold_of)
                pred_path = PREDICTIONS_DIR / f"{scheme}_{task}_{model_key}.csv"
                write_predictions(pred_path, records, subset_idx, y, fold_of,
                                  res["preds"], task)
                print(f"   {MODEL_NAMES[model_key]:32s} macro-F1 {res['macro_f1']['mean']:.3f} "
                      f"± {res['macro_f1']['sd']:.3f}  pooled {res['macro_f1']['pooled']:.3f}  "
                      f"acc {res['accuracy']['mean']:.4f}  ({time.time() - t0:.1f}s)")
                rows_bench.append(dict(
                    scheme=scheme, task=task, model=model_key,
                    model_label=MODEL_NAMES[model_key],
                    macro_f1_mean=res["macro_f1"]["mean"], macro_f1_sd=res["macro_f1"]["sd"],
                    macro_f1_pooled=res["macro_f1"]["pooled"],
                    accuracy_mean=res["accuracy"]["mean"], accuracy_sd=res["accuracy"]["sd"],
                    accuracy_pooled=res["accuracy"]["pooled"],
                    fold_values=";".join(f"{v:.4f}" for v in res["macro_f1"]["values"]),
                    n=len(y)))
                for row in res["classes"]:
                    rows_class.append(dict(scheme=scheme, task=task, model=model_key,
                                           **row))
                manifest["results"].append(dict(
                    scheme=scheme, task=task, model=model_key,
                    macro_f1=res["macro_f1"], accuracy=res["accuracy"],
                    seconds=round(time.time() - t0, 2), predictions=str(pred_path)))

    if args.no_tables:
        return
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with (TABLES_DIR / "table3_benchmark.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_bench[0]))
        w.writeheader()
        w.writerows(rows_bench)
    with (TABLES_DIR / "table4_per_class.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_class[0]))
        w.writeheader()
        w.writerows(rows_class)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "run_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {TABLES_DIR / 'table3_benchmark.csv'}")
    print(f"wrote {TABLES_DIR / 'table4_per_class.csv'}")
    print(f"wrote {RESULTS_DIR / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
