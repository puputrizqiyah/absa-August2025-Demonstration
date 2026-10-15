"""Fold builders for the three validation designs of Table 10.

* `stratified_folds`  - primary protocol: 5-fold stratified cross-validation, seed 42.
* `grouped_folds`     - near-duplicate aware: every cluster of similar tweets stays
                        inside one fold, so no near-duplicate can leak across the split.
* `day_held_out_folds`- temporal: train on all remaining collection days and hold
                        out the records of one day at a time.

Each builder yields (train_idx, test_idx) pairs over the task's record list, so the
same code drives relevance (all records) and aspect/polarity (relevant records).
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from config import N_FOLDS, SEED


def stratified_folds(y, n_splits: int = N_FOLDS, seed: int = SEED):
    y = list(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def grouped_folds(y, groups, n_splits: int = N_FOLDS, seed: int = SEED):
    """Stratified group folds: group assignment is the near-duplicate cluster id.

    StratifiedGroupKFold keeps the class proportions as close to the corpus as the
    grouping allows and never places two members of a group in different folds.
    """
    y = list(y)
    groups = list(groups)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y, groups))


def day_held_out_folds(days):
    """One fold per collection day: test = that day, train = every other day."""
    days = list(days)
    unique = sorted({d for d in days if d is not None})
    folds = []
    for day in unique:
        test_idx = [i for i, d in enumerate(days) if d == day]
        train_idx = [i for i, d in enumerate(days) if d != day]
        if not test_idx or not train_idx:
            continue
        folds.append((np.asarray(train_idx), np.asarray(test_idx), day))
    return folds


def fold_summary(folds):
    return [dict(train=int(len(tr)), test=int(len(te))) for tr, te in folds]
