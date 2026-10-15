"""Metrics used in the paper: macro-F1 (primary), accuracy, per-class F1 and the
conditional / coverage-aware distinction for the zero-shot comparator.

`coverage_aware` treats a missing label as an error: the missing records are added
to the denominator of accuracy and to every class denominator, so a model cannot
improve its score by declining to answer.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def per_class(y_true, y_pred, labels):
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return [dict(label=str(lab), precision=float(pi), recall=float(ri),
                 f1=float(fi), support=int(s)) for lab, pi, ri, fi, s in zip(labels, p, r, f1, support)]


def pooled_scores(y_true, y_pred) -> dict:
    return dict(macro_f1=macro_f1(y_true, y_pred), accuracy=accuracy(y_true, y_pred),
                n=len(y_true))


def conditional_scores(y_true, y_pred_masked) -> dict:
    """Score only the records the model actually labelled.

    y_pred_masked contains None where the model produced no usable label.
    """
    keep = [i for i, p in enumerate(y_pred_masked) if p is not None]
    if not keep:
        return dict(macro_f1=0.0, accuracy=0.0, coverage=0.0, n_answered=0, n_total=len(y_true))
    yt = [y_true[i] for i in keep]
    yp = [y_pred_masked[i] for i in keep]
    out = pooled_scores(yt, yp)
    out.update(coverage=len(keep) / len(y_true), n_answered=len(keep), n_total=len(y_true))
    return out


def coverage_aware_scores(y_true, y_pred_masked, labels, missing_token="__MISSING__") -> dict:
    """Missing-as-error: unanswered records count as wrong on every label.

    Labels are compared as strings so that a boolean task (relevance) and a
    categorical task (aspect, polarity) run through the same code path.
    """
    yt = [str(v) for v in y_true]
    yp = [missing_token if p is None else str(p) for p in y_pred_masked]
    label_str = [str(l) for l in labels]
    macro = float(f1_score(yt, yp, labels=label_str, average="macro", zero_division=0))
    acc = float(sum(1 for a, b in zip(yt, yp) if a == b) / len(yt))
    return dict(macro_f1=macro, accuracy=acc,
                coverage=sum(1 for p in y_pred_masked if p is not None) / len(yt),
                n_answered=sum(1 for p in y_pred_masked if p is not None), n_total=len(yt))


def fold_stats(fold_scores) -> dict:
    """Mean, sample standard deviation (ddof=1) and range over the five folds."""
    arr = np.asarray(fold_scores, dtype=float)
    return dict(mean=float(arr.mean()), sd=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                min=float(arr.min()), max=float(arr.max()), n_folds=int(arr.size),
                values=[float(v) for v in arr])


def majority_class_baseline(y_true, labels) -> dict:
    """Macro-F1 of always predicting the training-set majority class."""
    values, counts = np.unique(np.asarray(y_true, dtype=object), return_counts=True)
    majority = values[int(np.argmax(counts))]
    return pooled_scores(list(y_true), [majority] * len(y_true))
