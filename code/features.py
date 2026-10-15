"""Table 8: the seven textual features, their counts and the comparator's error rate.

Population: the tweets on which the zero-shot comparator produced a polarity label
(5,114 records, 784 errors, 15.3%). For each feature the script reports

    Tweets          - records carrying the feature
    Errors          - of those, records where the comparator disagrees with gold
    Error rate (%)  - Errors / Tweets
    Complement rate - the same rate over the records without the feature
    Chi-square p    - independence of the feature and the error, from a 2x2 table
                      (Yates-corrected, as reported in the paper)

Definitions are the printed ones from the Table 8 note.

Usage:
    python code/features.py
    python code/features.py --out data/derived/table8_feature_counts.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter

from config import (DATA_DIR, DERIVED_DIR, ENGLISH_FUNCTION_WORDS, GPT_OUTPUT_FILE,
                    INFORMAL_MARKERS, NEGATION_MARKERS, NEWS_OUTLETS,
                    NEWS_REPORT_MARKERS, SARCASM_MARKERS)
from data_io import load_gold_standard, load_zero_shot_output
from scipy.stats import chi2_contingency

WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _has_marker(text: str, markers) -> bool:
    """Word-boundary match for single words, substring match for phrases."""
    padded = f" {' '.join(_tokens(text))} "
    for m in markers:
        if " " in m:
            if m in padded:
                return True
        elif f" {m} " in padded:
            return True
    return False


# --------------------------------------------------------------------------- detectors
def f_emoji(text):
    """Any character with a code point above 127."""
    return any(ord(ch) > 127 for ch in text)


def f_news_outlet(text):
    """Quoted/reported news template: an outlet is named or a report reproduced."""
    low = text.lower()
    return (any(outlet in low for outlet in NEWS_OUTLETS)
            or any(marker in low for marker in NEWS_REPORT_MARKERS))


def f_informal(text):
    """Any of the informal register markers listed in the paper."""
    return _has_marker(text, INFORMAL_MARKERS)


def f_code_mixing(text):
    """At least two distinct English function words."""
    hits = {w for w in _tokens(text) if w in ENGLISH_FUNCTION_WORDS}
    return len(hits) >= 2


def f_negation(text):
    return _has_marker(text, NEGATION_MARKERS)


def f_question(text):
    """Presence of the question mark."""
    return "?" in text


def f_sarcasm(text):
    return _has_marker(text, SARCASM_MARKERS)


# the counts printed in the accepted version, kept as a self-check
PUBLISHED_TABLE8 = {
    "Emoji or non-ASCII characters": (237, 50, 21.1),
    "Quoted/reported news template (news outlet named)": (230, 48, 20.9),
    "Informal register markers": (2264, 290, 12.8),
    "English code-mixing (>=2 distinct function words)": (67, 5, 7.5),
    "Negation": (1860, 269, 14.5),
    "Question form (?)": (792, 83, 10.5),
    "Sarcasm/irony lexical markers": (272, 35, 12.9),
    "All tweets (population)": (5114, 784, 15.3),
}

FEATURES = [
    ("Emoji or non-ASCII characters", f_emoji),
    ("Quoted/reported news template (news outlet named)", f_news_outlet),
    ("Informal register markers", f_informal),
    ("English code-mixing (>=2 distinct function words)", f_code_mixing),
    ("Negation", f_negation),
    ("Question form (?)", f_question),
    ("Sarcasm/irony lexical markers", f_sarcasm),
]


def build_rows(records, zero_shot, task="polarity"):
    """Rows of the Table 8 population: comparator-labelled polarity records."""
    rows = []
    for rec in records:
        if rec["relevant_gold"] is not True or rec["polarity_gold"] is None:
            continue
        pred = zero_shot.get(rec["index"], {}).get(task)
        if pred is None:
            continue
        rows.append(dict(index=rec["index"], text=rec["text"],
                         gold=rec["polarity_gold"], pred=pred,
                         error=int(pred != rec["polarity_gold"])))
    return rows


def feature_table(rows):
    total, errors = len(rows), sum(r["error"] for r in rows)
    out = []
    for name, fn in FEATURES:
        flags = [fn(r["text"]) for r in rows]
        n_with = sum(flags)
        n_err_with = sum(r["error"] for r, f in zip(rows, flags) if f)
        n_without = total - n_with
        n_err_without = errors - n_err_with
        table = [[n_err_with, n_with - n_err_with], [n_err_without, n_without - n_err_without]]
        try:
            p = float(chi2_contingency(table, correction=True)[1])
        except ValueError:
            p = float("nan")
        out.append(dict(feature=name, tweets=n_with, errors=n_err_with,
                        error_rate=100 * n_err_with / n_with if n_with else float("nan"),
                        complement_rate=100 * n_err_without / n_without if n_without else float("nan"),
                        chi2_p=p))
    out.append(dict(feature="All tweets (population)", tweets=total, errors=errors,
                    error_rate=100 * errors / total if total else float("nan"),
                    complement_rate=float("nan"), chi2_p=float("nan")))
    for row in out:
        published = PUBLISHED_TABLE8.get(row["feature"])
        row["published_tweets"], row["published_errors"], row["published_rate"] = (
            published if published else ("", "", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DERIVED_DIR / "table8_feature_counts.csv"))
    ap.add_argument("--gpt", default=None, help="raw zero-shot workbook")
    args = ap.parse_args()

    records = load_gold_standard()
    zero_shot = load_zero_shot_output(args.gpt or DATA_DIR / GPT_OUTPUT_FILE)
    rows = build_rows(records, zero_shot)
    table = feature_table(rows)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Textual feature", "Tweets", "Errors", "Error rate (%)",
                    "Complement rate (%)", "Chi-square p",
                    "Published tweets", "Published errors", "Published error rate (%)"])
        for row in table:
            w.writerow([row["feature"], row["tweets"], row["errors"],
                        f"{row['error_rate']:.1f}",
                        "" if row["complement_rate"] != row["complement_rate"] else f"{row['complement_rate']:.1f}",
                        "" if row["chi2_p"] != row["chi2_p"] else ("<0.001" if row["chi2_p"] < 0.001 else f"{row['chi2_p']:.3f}"),
                        row["published_tweets"], row["published_errors"], row["published_rate"]])

    print(f"population: {len(rows)} comparator-labelled polarity records, "
          f"{sum(r['error'] for r in rows)} errors "
          f"({100 * sum(r['error'] for r in rows) / len(rows):.1f}%)")
    print(f"{'feature':52s} {'tweets':>7s} {'errors':>7s} {'rate':>6s} {'compl':>6s} {'p':>8s}")
    for row in table:
        p = "" if row["chi2_p"] != row["chi2_p"] else ("<0.001" if row["chi2_p"] < 0.001 else f"{row['chi2_p']:.3f}")
        comp = "" if row["complement_rate"] != row["complement_rate"] else f"{row['complement_rate']:.1f}"
        print(f"{row['feature']:52s} {row['tweets']:7d} {row['errors']:7d} "
              f"{row['error_rate']:6.1f} {comp:>6s} {p:>8s}")
    print(f"\nwrote {args.out}")
    print("Self-check against Table 8: 237/50/21.1, 230/48/20.9, 2264/290/12.8, 67/5/7.5,")
    print("                            1860/269/14.5, 792/83/10.5, 272/35/12.9, population 5114/784/15.3")


if __name__ == "__main__":
    main()
