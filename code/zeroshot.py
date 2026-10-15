"""Score the zero-shot comparator, and optionally re-run it.

Scoring (the default) uses the comparator labels that were carried into the study,
so the coverage figures of Table 6 and the confusion matrix of Table 5 can be
recomputed without any API call:

    python code/zeroshot.py

Re-running the comparator is optional and needs OPENAI_API_KEY:

    python code/zeroshot.py --query --model gpt-4o-mini-2024-07-18 --out data/derived/zeroshot_rerun.jsonl

The prompt and the decoding configuration are those of Appendix A; the instruction
text lives in docs/prompt_gpt_zero_shot.md and is loaded from there rather than
duplicated here.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from config import DERIVED_DIR, LABELS, DOCS_DIR, RESULTS_DIR
from data_io import load_gold_standard, load_zero_shot_output, task_frame
from metrics import conditional_scores, coverage_aware_scores, per_class, pooled_scores

PROMPT_FILE = DOCS_DIR / "prompt_gpt_zero_shot.md"
DECODING = dict(model="gpt-4o-mini-2024-07-18", temperature=0, max_output_tokens=4096,
                batch_size=10, retries=0)


def load_prompt():
    """The system instruction and the user-instruction template of Appendix A."""
    text = PROMPT_FILE.read_text(encoding="utf-8")
    system = text.split("## System instruction", 1)[1].split("## User instruction", 1)[0].strip()
    user = text.split("## User instruction", 1)[1].strip()
    return system, user


def _per_class_answered(gold, preds, labels):
    keep = [i for i, p in enumerate(preds) if p is not None]
    if not keep:
        return []
    return per_class([gold[i] for i in keep], [preds[i] for i in keep], labels)


def evaluate(records):
    zero = load_zero_shot_output()
    rows = []
    for task in ("relevance", "aspect", "polarity"):
        subset, gold = task_frame(records, task)
        preds = [zero.get(r["index"], {}).get(task) for r in subset]
        cond = conditional_scores(gold, preds)
        cov = coverage_aware_scores(gold, preds, LABELS[task])
        rows.append(dict(task=task, n=len(gold), valid=cond["n_answered"],
                         coverage_pct=round(100 * cond["coverage"], 2),
                         conditional_macro_f1=round(cond["macro_f1"], 4),
                         conditional_accuracy=round(cond["accuracy"], 4),
                         coverage_aware_macro_f1=round(cov["macro_f1"], 4),
                         coverage_aware_accuracy=round(cov["accuracy"], 4),
                         per_class=_per_class_answered(gold, preds, LABELS[task])))
    return rows

def query(records, model, out_path, limit=None, api_key=None):
    """Re-run the comparator exactly as configured in Appendix A."""
    from openai import OpenAI   # imported lazily: scoring needs no API client
    system, user_template = load_prompt()
    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    subset = records[:limit] if limit else records

    results = []
    batch = DECODING["batch_size"]
    for start in range(0, len(subset), batch):
        chunk = subset[start:start + batch]
        lines = [f"[{r['index']}] Text: {r['text']}" for r in chunk]
        response = client.chat.completions.create(
            model=model or DECODING["model"],
            temperature=DECODING["temperature"],
            max_tokens=DECODING["max_output_tokens"],
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_template + "\n\n" + "\n".join(lines)}],
        )
        payload = response.choices[0].message.content
        try:
            parsed = json.loads(payload)
            parsed = parsed if isinstance(parsed, list) else parsed.get("items", [])
        except json.JSONDecodeError:
            parsed = []          # an unparsed response is a missing label, not a repair
        by_index = {int(item["index"]): item for item in parsed if "index" in item}
        for r in chunk:
            item = by_index.get(r["index"], {})
            row = dict(index=r["index"], text=r["text"],
                       relevant=item.get("relevant"),
                       aspect=item.get("aspect") or None,
                       label=item.get("label") or None)
            # the guideline leaves aspect and polarity empty when relevance is false
            if row["relevant"] is False:
                row["aspect"] = None
                row["label"] = None
            results.append(row)
        print(f"   batch {start // batch + 1}: {len(chunk)} records, "
              f"{len(by_index)} usable objects")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(results)} records)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", action="store_true", help="call the API instead of scoring saved labels")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(DERIVED_DIR / "zeroshot_rerun.jsonl"))
    args = ap.parse_args()

    records = load_gold_standard()
    if args.query:
        query(records, args.model, args.out, limit=args.limit)
        return

    rows = evaluate(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "tables" / "table6_coverage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "evaluation_instances", "valid_model_outputs", "coverage_pct",
                    "conditional_macro_f1", "conditional_accuracy",
                    "coverage_aware_macro_f1", "coverage_aware_accuracy"])
        for row in rows:
            w.writerow([row["task"], row["n"], row["valid"], row["coverage_pct"],
                        row["conditional_macro_f1"], row["conditional_accuracy"],
                        row["coverage_aware_macro_f1"], row["coverage_aware_accuracy"]])
            if row["per_class"]:
                for cls in row["per_class"]:
                    print(f"   {row['task']:10s} {cls['label']:12s} P={cls['precision']:.3f} "
                          f"R={cls['recall']:.3f} F1={cls['f1']:.3f} n={cls['support']}")
    print(f"\nwrote {out}")
    print("published: relevance coverage 100% / 0.698 coverage-aware; "
          "aspect 78.85% / 0.711; polarity 84.18% / 0.647")


if __name__ == "__main__":
    main()
