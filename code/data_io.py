"""Reading the gold standard and the raw zero-shot output.

Both files are plain spreadsheets; nothing here depends on pandas so that the
deposit stays light (numpy + scikit-learn + scipy + openpyxl).
"""
from __future__ import annotations

import openpyxl

from config import (DATA_DIR, EXPECTED_N_RECORDS, GOLD_COLUMNS, GOLD_STANDARD_FILE,
                    GOLD_STANDARD_SHEET, GPT_COLUMNS, GPT_OUTPUT_FILE, GPT_OUTPUT_SHEET,
                    RAW_GPT_COLUMNS, RAW_GPT_FILE, RAW_GPT_SHEET)

TRUE_TOKENS = {"true", "ya", "yes", "1", "relevan", "relevant"}
FALSE_TOKENS = {"false", "tidak", "no", "0", "tidak relevan", "not relevant"}


class Record(dict):
    """One corpus record: index, text, gold labels and (optionally) model output."""

    @property
    def is_relevant(self) -> bool:
        return self["relevant_gold"] is True


def _load_rows(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet in wb.sheetnames else wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    return header, rows[1:]


def _parse_relevance(value):
    """Gold relevance is written as TRUE/FALSE (sometimes in Indonesian)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False
    return None


def load_gold_standard(path=None, allow_partial=False):
    """Return the 7,131 adjudicated records, ordered by their original index.

    `allow_partial` keeps records whose relevance label is missing instead of
    failing, which is useful while inspecting a partially labelled file.
    """
    path = path or DATA_DIR / GOLD_STANDARD_FILE
    header, rows = _load_rows(path, GOLD_STANDARD_SHEET)
    missing = [c for c in GOLD_COLUMNS.values() if c not in header]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {header}")

    records, skipped = [], 0
    for row in rows:
        cells = dict(zip(header, row))
        index = cells[GOLD_COLUMNS["index"]]
        text = cells[GOLD_COLUMNS["text"]]
        if index is None or text is None:
            skipped += 1
            continue
        relevance = _parse_relevance(cells[GOLD_COLUMNS["relevant"]])
        if relevance is None and not allow_partial:
            skipped += 1
            continue
        aspect = _clean_label(cells[GOLD_COLUMNS["aspect"]])
        polarity = _clean_label(cells[GOLD_COLUMNS["polarity"]])
        records.append(Record(
            index=int(index),
            text=" ".join(str(text).split()),   # collapse whitespace/newlines
            relevant_gold=relevance,
            aspect_gold=aspect,
            polarity_gold=polarity,
            created_at=cells.get("created_at_validator") or cells.get("created_at"),
        ))
    records.sort(key=lambda r: r["index"])
    if len(records) != EXPECTED_N_RECORDS and not allow_partial:
        raise AssertionError(
            f"expected {EXPECTED_N_RECORDS} records, read {len(records)} "
            f"(skipped {skipped}); wrong file or a partially labelled version?")
    return records


def _clean_label(value):
    if value is None:
        return None
    label = str(value).strip()
    if label.lower() in ("none", "nan", ""):
        return None
    return label


def load_zero_shot_output(path=None, task_aware=False):
    """The comparator's labels, keyed by record index.

    Returns {index: {"relevant": ..., "aspect": ..., "polarity": ...}}. A value of
    None means the model produced no usable label for that record; those records
    are counted as errors in the coverage-aware evaluation (Section 4, Table 6) and
    are excluded from the conditional one.

    By default the labels are read from the adjudication workbook, which is where
    the comparator's output was carried into the study (TitleCase labels, explicit
    blanks). Pass `path=<raw workbook>` with `task_aware=True` to read the raw
    pre-deduplication responses instead; "tidak sesuai konteks" and "null" are then
    treated as missing and the lowercase labels are normalised.
    """
    path = path or DATA_DIR / GPT_OUTPUT_FILE
    raw = task_aware or str(path).endswith(RAW_GPT_FILE)
    sheet = RAW_GPT_SHEET if raw else GPT_OUTPUT_SHEET
    columns = RAW_GPT_COLUMNS if raw else GPT_COLUMNS

    header, rows = _load_rows(path, sheet)
    if raw and columns["index"] not in header:
        columns = dict(columns)

    missing_tokens = {"", "none", "null", "nan", "tidak sesuai konteks",
                      "tidak relevan", "-"}
    out = {}
    for row in rows:
        cells = dict(zip(header, row))
        index = cells.get(columns["index"])
        if index is None or str(index).strip() == "":
            continue
        out[int(index)] = dict(
            relevance=_parse_relevance(cells.get(columns["relevant"])),
            aspect=_normalise_label(cells.get(columns["aspect"]), missing_tokens),
            polarity=_normalise_label(cells.get(columns["polarity"]), missing_tokens),
        )
    return out


def _normalise_label(value, missing_tokens):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in missing_tokens:
        return None
    return text[:1].upper() + text[1:].lower()


def task_indices(records, task):
    """Positions in `records` of the records a task is evaluated on."""
    from config import TASK_SCOPE
    if TASK_SCOPE[task] == "relevant":
        return [i for i, r in enumerate(records) if r["relevant_gold"] is True]
    return list(range(len(records)))


def task_frame(records, task):
    """Records and gold labels that a task is evaluated on.

    relevance: all records; aspect and polarity: the relevant records only.
    """
    key = {"relevance": "relevant_gold", "aspect": "aspect_gold", "polarity": "polarity_gold"}[task]
    subset = [records[i] for i in task_indices(records, task)]
    return subset, [r[key] for r in subset]

