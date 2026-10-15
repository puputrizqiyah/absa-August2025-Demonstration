"""The three paired procedures behind Table 9.

1. `paired_bootstrap_ci`  - instance-level: 95% percentile interval for the
   difference in macro-F1, resampling out-of-fold instances 10,000 times.
3. `exact_mcnemar`        - instance-level: exact two-sided McNemar test on the
   paired correctness vectors.
2. `nadeau_bengio_ttest`  - fold-level: corrected resampled t-test. The variance
   of the mean difference is inflated by (1/k + 1/(k-1)) and the statistic is
   referred to a t distribution with k-1 degrees of freedom.

Nothing here adjusts the p-values: the paper reports the unadjusted values and
applies Holm-Bonferroni afterwards with `holm_bonferroni`.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _encode_labels(*arrays):
    """Map labels (bool, str or int) onto integer codes; returns codes and K."""
    values = []
    seen = set()
    for arr in arrays:
        for v in arr:
            key = str(v)
            if key not in seen:
                seen.add(key)
                values.append(key)
    code = {v: i for i, v in enumerate(values)}
    return [[code[str(x)] for x in arr] for arr in arrays], len(values)


def _macro_f1_codes(y_true, y_pred, k) -> float:
    """Macro-F1 from integer codes, without touching sklearn inside the loop."""
    conf = np.bincount(np.asarray(y_true, dtype=np.int64) * k +
                       np.asarray(y_pred, dtype=np.int64), minlength=k * k).reshape(k, k)
    tp = np.diag(conf).astype(float)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def paired_bootstrap_ci(y_true, y_pred_a, y_pred_b, n_boot=10000, ci=0.95, seed=42):
    """Difference in macro-F1 (A - B) with a percentile interval and a p-value.

    The bootstrap resamples instances, so it respects the pairing. The reported
    p-value is the two-sided proportion of resamples whose difference changes
    sign, which is the bootstrap analogue of the paired test.
    """
    y_true, y_pred_a, y_pred_b = list(y_true), list(y_pred_a), list(y_pred_b)
    n = len(y_true)
    (yt, pa, pb), k = _encode_labels(y_true, y_pred_a, y_pred_b)
    yt, pa, pb = np.asarray(yt), np.asarray(pa), np.asarray(pb)
    observed = _macro_f1_codes(yt, pa, k) - _macro_f1_codes(yt, pb, k)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = _macro_f1_codes(yt[idx], pa[idx], k) - _macro_f1_codes(yt[idx], pb[idx], k)
    lo, hi = np.percentile(diffs, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return dict(delta=float(observed), ci_low=float(lo), ci_high=float(hi),
                p_value=float(min(1.0, p)), n_boot=int(n_boot))


def exact_mcnemar(y_true, y_pred_a, y_pred_b):
    """Exact two-sided McNemar test on the discordant pairs.

    b = A correct and B wrong, c = A wrong and B correct;
    p = 2 * P(X <= min(b, c)) for X ~ Binomial(b + c, 0.5), capped at 1.
    """
    y_true = list(y_true)
    a_ok = [pa == yt for pa, yt in zip(y_pred_a, y_true)]
    b_ok = [pb == yt for pb, yt in zip(y_pred_b, y_true)]
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n = b + c
    if n == 0:
        return dict(b=b, c=c, p_value=1.0, statistic=0.0)
    p = 2 * stats.binom.cdf(min(b, c), n, 0.5)
    return dict(b=b, c=c, p_value=float(min(1.0, p)), statistic=float(b))


def nadeau_bengio_ttest(fold_scores_a, fold_scores_b):
    """Corrected resampled t-test on per-fold scores of two models.

    t = mean(d) / sqrt((1/k + 1/(k-1)) * var(d)),  df = k - 1
    """
    d = np.asarray(fold_scores_a, dtype=float) - np.asarray(fold_scores_b, dtype=float)
    k = d.size
    if k < 2:
        raise ValueError("need at least two folds")
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))
    if var_d == 0:
        return dict(delta=mean_d, t_stat=float("inf") if mean_d else 0.0, df=k - 1, p_value=0.0)
    t = mean_d / np.sqrt((1.0 / k + 1.0 / (k - 1)) * var_d)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return dict(delta=mean_d, t_stat=float(t), df=k - 1, p_value=float(p))


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down correction.

    Returns a list of dicts with the raw p, the adjusted p and whether the
    contrast survives at 0.05. Adjusted p-values are the running maximum of
    (m / rank) * p, capped at 1.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    out = [dict(index=i, p_value=float(p_values[i]), p_adjusted=1.0, survives=False)
           for i in range(m)]
    running = 0.0
    for rank, i in enumerate(order, start=1):
        adjusted = min(1.0, (m - rank + 1) * p_values[i])
        running = max(running, adjusted)
        out[i]["p_adjusted"] = float(running)
        out[i]["survives"] = bool(running <= 0.05)
    return out
