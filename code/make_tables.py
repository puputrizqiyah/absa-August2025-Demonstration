"""Rebuild every table of the paper from the saved out-of-fold predictions.

Inputs (all produced by the other scripts):

    results/predictions/<scheme>_<task>_<model>.csv   lexical models
    results/indobert_results.jsonl                    fine-tuned encoder, per fold
    data/adjudication_analysis_results.xlsx           zero-shot comparator labels
    data/derived/table8_feature_counts.csv            feature rates

Outputs:

    results/tables/table3_benchmark.csv    macro-F1 (mean +- fold SD) and accuracy
    results/tables/table5_confusion.csv    comparator polarity confusion matrix
    results/tables/table6_coverage.csv     conditional vs coverage-aware scoring
    results/tables/table9_paired.csv       the 15 paired contrasts + Holm-Bonferroni
    results/tables/table10_robustness.csv  stratified / grouped / temporal

Usage:
    python code/make_tables.py
    python code/make_tables.py --scheme stratified
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from config import (DATA_DIR, LABELS, PREDICTIONS_DIR, RESULTS_DIR, TABLES_DIR)
from data_io import load_gold_standard, load_zero_shot_output, task_frame
from lexical_models import MODEL_NAMES
from metrics import conditional_scores, coverage_aware_scores, pooled_scores
from stats_tests import (exact_mcnemar, holm_bonferroni, nadeau_bengio_ttest,
                         paired_bootstrap_ci)

# The 15 contrasts of Table 9. "instance" pairs use the out-of-fold prediction
# vectors (bootstrap + exact McNemar); "fold" pairs use the per-fold scores with
# the corrected resampled t-test, because IndoBERT is reported fold by fold.
CONTRASTS = {
    "relevance": [("tfidf_lr_balanced", "tfidf_lr", "instance"),
                  ("char_lr", "complement_nb", "instance"),
                  ("indobert", "tfidf_lr", "fold"),
                  ("indobert", "tfidf_lr_balanced", "fold"),
                  ("indobert", "char_lr", "fold")],
    "aspect": [("tfidf_lr_balanced", "tfidf_lr", "instance"),
               ("char_lr", "tfidf_lr", "instance"),
               ("indobert", "tfidf_lr", "fold"),
               ("indobert", "tfidf_lr_balanced", "fold"),
               ("indobert", "char_lr", "fold")],
    "polarity": [("tfidf_lr_balanced", "tfidf_lr", "instance"),
                 ("tfidf_lr", "complement_nb", "instance"),
                 ("indobert", "tfidf_lr", "fold"),
                 ("indobert", "tfidf_lr_balanced", "fold"),
                 ("indobert", "char_lr", "fold")],
}

PUBLISHED_TABLE3 = {
    ("relevance", "majority"): (0.460, 85.19), ("relevance", "tfidf_lr"): (0.552, 86.44),
    ("relevance", "tfidf_lr_balanced"): (0.719, 84.85), ("relevance", "complement_nb"): (0.595, 86.85),
    ("relevance", "char_lr"): (0.622, 87.46), ("relevance", "indobert"): (0.757, 89.17),
    ("aspect", "majority"): (0.177, 36.08), ("aspect", "tfidf_lr"): (0.749, 75.11),
    ("aspect", "tfidf_lr_balanced"): (0.752, 75.26), ("aspect", "complement_nb"): (0.710, 71.59),
    ("aspect", "char_lr"): (0.769, 77.09), ("aspect", "indobert"): (0.772, 77.48),
    ("polarity", "majority"): (0.271, 68.30), ("polarity", "tfidf_lr"): (0.550, 75.80),
    ("polarity", "tfidf_lr_balanced"): (0.617, 72.00), ("polarity", "complement_nb"): (0.556, 75.24),
    ("polarity", "char_lr"): (0.571, 76.23), ("polarity", "indobert"): (0.665, 78.67),
}
PUBLISHED_COVERAGE_AWARE = {"relevance": 0.698, "aspect": 0.711, "polarity": 0.647}
PUBLISHED_COVERAGE = {"relevance": 1.0, "aspect": 0.7885, "polarity": 0.8418}


def read_predictions(scheme, task, model):
    path = PREDICTIONS_DIR / f"{scheme}_{task}_{model}.csv"
    if not path.exists():
        return None
    rows = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return dict(gold=[r["gold"] for r in rows], pred=[r["predicted"] for r in rows],
                fold=[int(r["fold"]) for r in rows], path=path)


def read_indobert(path=None):
    """Per-fold scores and pooled per-class results of the fine-tuned encoder."""
    path = path or RESULTS_DIR / "indobert_results.jsonl"
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task = {"sentiment": "polarity"}.get(rec["task"], rec["task"])
            entry = out.setdefault(task, {})
            if "fold" in rec:
                entry.setdefault("folds", []).append(rec["macro_f1"])
                entry.setdefault("fold_acc", []).append(rec["acc"])
            else:
                entry["pooled"] = rec["macro_f1"]
                entry["accuracy"] = rec["acc"]
                entry["report"] = rec.get("report", "")
    return out


def fold_scores_by_model(scheme, task, model):
    """Per-fold macro-F1 of a lexical model, recomputed from its predictions."""
    data = read_predictions(scheme, task, model)
    if not data:
        return None
    scores = {}
    for f in sorted(set(data["fold"])):
        idx = [i for i, x in enumerate(data["fold"]) if x == f]
        scores[f] = pooled_scores([data["gold"][i] for i in idx], [data["pred"][i] for i in idx])["macro_f1"]
    return [scores[f] for f in sorted(scores)]


def table3_and_10(schemes, tasks, models):
    bench, robustness = [], []
    for scheme in schemes:
        for task in tasks:
            for model in models:
                data = read_predictions(scheme, task, model)
                if not data:
                    continue
                pooled = pooled_scores(data["gold"], data["pred"])
                per_fold = fold_scores_by_model(scheme, task, model)
                import numpy as np
                sd = float(np.std(per_fold, ddof=1)) if per_fold and len(per_fold) > 1 else 0.0
                acc_by_fold = []
                for f in sorted(set(data["fold"])):
                    idx = [i for i, x in enumerate(data["fold"]) if x == f]
                    acc_by_fold.append(pooled_scores([data["gold"][i] for i in idx],
                                                     [data["pred"][i] for i in idx])["accuracy"])
                bench.append(dict(scheme=scheme, task=task, model=model,
                                  model_label=MODEL_NAMES.get(model, model),
                                  macro_f1=pooled["macro_f1"], macro_f1_sd=sd,
                                  accuracy=float(np.mean(acc_by_fold)), n=pooled["n"],
                                  fold_values=";".join(f"{v:.4f}" for v in per_fold)))
                robustness.append(dict(scheme=scheme, task=task, model=model,
                                       macro_f1=pooled["macro_f1"]))
    return bench, robustness


def table6_coverage(tasks, records):
    zero = load_zero_shot_output()
    rows = []
    for task in tasks:
        subset, gold = task_frame(records, task)
        preds = [zero.get(r["index"], {}).get(task) for r in subset]
        cond = conditional_scores(gold, preds)
        cov = coverage_aware_scores(gold, preds, LABELS[task])
        rows.append(dict(task=task, evaluation_instances=len(gold),
                         valid_outputs=cond["n_answered"], coverage_pct=100 * cond["coverage"],
                         conditional_macro_f1=cond["macro_f1"], conditional_accuracy=cond["accuracy"],
                         coverage_aware_macro_f1=cov["macro_f1"],
                         coverage_aware_accuracy=cov["accuracy"]))
    return rows


def table5_confusion(records):
    zero = load_zero_shot_output()
    subset, gold = task_frame(records, "polarity")
    preds = [zero.get(r["index"], {}).get("polarity") for r in subset]
    labels = LABELS["polarity"]
    matrix = {g: {p: 0 for p in labels} for g in labels}
    missing = 0
    for g, p in zip(gold, preds):
        if p is None:
            missing += 1
            continue
        matrix[g][p] += 1
    rows = []
    for g in labels:
        row = dict(gold_label=g, **{f"predicted_{p}": matrix[g][p] for p in labels})
        row["total"] = sum(matrix[g][p] for p in labels)
        rows.append(row)
    rows.append(dict(gold_label="Total",
                     **{f"predicted_{p}": sum(matrix[g][p] for g in labels) for p in labels},
                     total=sum(matrix[g][p] for g in labels for p in labels)))
    return rows, missing


def table9_paired(scheme, tasks, models, indobert):
    """The 15 contrasts with unadjusted p-values and the Holm-Bonferroni outcome."""
    rows = []
    for task in tasks:
        for a, b, kind in CONTRASTS[task]:
            if kind == "instance":
                pa, pb = read_predictions(scheme, task, a), read_predictions(scheme, task, b)
                if not pa or not pb:
                    continue
                boot = paired_bootstrap_ci(pa["gold"], pa["pred"], pb["pred"])
                mcn = exact_mcnemar(pa["gold"], pa["pred"], pb["pred"])
                rows.append(dict(task=task, comparison=f"{MODEL_NAMES.get(a, a)} vs {MODEL_NAMES.get(b, b)}",
                                 kind="instance", unit="out-of-fold instance",
                                 delta=boot["delta"], ci_low=boot["ci_low"], ci_high=boot["ci_high"],
                                 test="paired bootstrap (10,000) + exact McNemar",
                                 p_value=mcn["p_value"], p_bootstrap=boot["p_value"],
                                 stat_b=mcn["b"], stat_c=mcn["c"]))
            else:
                fa_a, fa_b = None, fold_scores_by_model(scheme, task, b)
                if a == "indobert":
                    fa_a = indobert.get(task, {}).get("folds")
                else:
                    fa_a = fold_scores_by_model(scheme, task, a)
                if not fa_a or not fa_b or len(fa_a) != len(fa_b):
                    continue
                t = nadeau_bengio_ttest(fa_a, fa_b)
                rows.append(dict(task=task, comparison=f"{MODEL_NAMES.get(a, a)} vs {MODEL_NAMES.get(b, b)}",
                                 kind="fold", unit="fold",
                                 delta=t["delta"], ci_low="", ci_high="",
                                 test=f"Nadeau-Bengio corrected resampled t (df={t['df']})",
                                 p_value=t["p_value"], p_bootstrap="", stat_b="", stat_c=""))
    adjusted = holm_bonferroni([r["p_value"] for r in rows])
    for row, adj in zip(rows, adjusted):
        row["p_holm"] = adj["p_adjusted"]
        row["survives_holm"] = int(adj["survives"])
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"   skipped {path.name} (no rows)")
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"   wrote {path.name} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="stratified")
    ap.add_argument("--schemes", nargs="*", default=["stratified", "grouped", "day_held_out"])
    ap.add_argument("--tasks", nargs="*", default=["relevance", "aspect", "polarity"])
    ap.add_argument("--models", nargs="*", default=["majority", "tfidf_lr", "tfidf_lr_balanced",
                                                    "complement_nb", "char_lr"])
    args = ap.parse_args()

    records = load_gold_standard()
    indobert = read_indobert()
    bench, robustness = table3_and_10(args.schemes, args.tasks, args.models)
    write_csv(TABLES_DIR / "table3_benchmark.csv", [r for r in bench if r["scheme"] == args.scheme])
    write_csv(TABLES_DIR / "table10_robustness.csv", robustness)

    coverage = table6_coverage(args.tasks, records)
    write_csv(TABLES_DIR / "table6_coverage.csv", coverage)

    confusion, missing = table5_confusion(records)
    write_csv(TABLES_DIR / "table5_confusion.csv", confusion)

    paired = table9_paired(args.scheme, args.tasks, args.models, indobert)
    write_csv(TABLES_DIR / "table9_paired.csv", paired)

    # ------------------------------------------------------------------ self-check
    print("\nself-check against the published table (macro-F1 / accuracy):")
    print(f"   {'task/model':38s} {'published':>16s} {'recomputed':>16s}")
    for row in bench:
        if row["scheme"] != args.scheme:
            continue
        key = (row["task"], row["model"])
        if key not in PUBLISHED_TABLE3:
            continue
        pub = PUBLISHED_TABLE3[key]
        print(f"   {row['task'] + '/' + row['model']:38s} "
              f"{pub[0]:.3f} / {pub[1]:5.2f}   {row['macro_f1']:.3f} / {100 * row['accuracy']:5.2f}")
    print(f"\n   {'task':10s} {'published cov':>13s} {'recomputed':>11s} "
          f"{'published c-a':>14s} {'recomputed':>11s}")
    for row in coverage:
        task = row["task"]
        print(f"   {task:10s} {PUBLISHED_COVERAGE[task]:13.2%} "
              f"{row['coverage_pct'] / 100:11.2%} "
              f"{PUBLISHED_COVERAGE_AWARE[task]:14.3f} {row['coverage_aware_macro_f1']:11.3f}")
    survivors = [r for r in paired if r["survives_holm"]]
    print(f"\n   contrasts surviving Holm-Bonferroni at 0.05: {len(survivors)}/{len(paired)} "
          f"(the paper reports 8/15)")


if __name__ == "__main__":
    main()
