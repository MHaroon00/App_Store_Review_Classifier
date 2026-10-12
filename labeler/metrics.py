"""Label presence statistics and agreement with an existing (human) label column."""

from __future__ import annotations

import re

import pandas as pd

from .engine import RowResult
from .prompts import NONE_LABEL, _norm

YES = {"1", "yes", "y", "true", "t", "present", "x"}
NO = {"0", "no", "n", "false", "f", "absent", "", "nan", "none"}


def presence_table(results: dict[int, RowResult], names: list[str], include_none: bool = False) -> pd.DataFrame:
    """How many labeled reviews contain each label."""
    labeled = [r for r in results.values() if not r.error]
    n = len(labeled)
    rows = []
    for name in names + ([NONE_LABEL] if include_none else []):
        count = sum(1 for r in labeled if name in r.labels)
        rows.append({"label": name, "reviews": count, "share": (count / n) if n else 0.0, "present": count > 0})
    return pd.DataFrame(rows)


def none_count(results: dict[int, RowResult]) -> int:
    return sum(1 for r in results.values() if not r.error and (not r.labels or r.labels == [NONE_LABEL]))


def is_binary_column(series: pd.Series) -> bool:
    vals = {str(v).strip().lower() for v in series.dropna().unique()}
    return bool(vals) and vals <= (YES | NO)


def parse_existing(series: pd.Series, names: list[str], binary_label: str | None = None) -> pd.Series:
    """Turn an existing column into sets of labels.

    * binary column (yes/no, 1/0, true/false) → {binary_label} or {}
    * text column with label names ("Usability", "Usability; Performance") → matching labels
    """
    canonical = {_norm(n): n for n in names}

    def convert(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        if binary_label:
            low = s.lower()
            if low in YES:
                return {binary_label}
            if low in NO:
                return set()
            return None
        found = set()
        for part in re.split(r"[;,|/]", s):
            key = _norm(part)
            if key in canonical:
                found.add(canonical[key])
        return found

    return series.map(convert)


def agreement(existing: pd.Series, results: dict[int, RowResult], names: list[str]) -> tuple[pd.DataFrame, dict]:
    """Per-label precision / recall / F1 / Cohen's kappa of the model vs. existing labels."""
    pairs = []
    for idx, truth in existing.items():
        r = results.get(idx)
        if truth is None or r is None or r.error:
            continue
        pred = {l for l in r.labels if l != NONE_LABEL}
        pairs.append((truth, pred))

    rows = []
    for name in names:
        tp = sum(1 for t, p in pairs if name in t and name in p)
        fp = sum(1 for t, p in pairs if name not in t and name in p)
        fn = sum(1 for t, p in pairs if name in t and name not in p)
        tn = len(pairs) - tp - fp - fn
        if tp + fp + fn == 0 and not any(name in t for t, _ in pairs):
            # label never appears in the existing column — skip unless model used it
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "label": name, "in_existing": tp + fn, "in_model": tp + fp, "both": tp,
            "precision": precision, "recall": recall, "f1": f1,
            "kappa": cohen_kappa(tp, fp, fn, tn), "agreement": (tp + tn) / len(pairs) if pairs else 0.0,
        })
    overall = {
        "compared": len(pairs),
        "exact_match": (sum(1 for t, p in pairs if t == p) / len(pairs)) if pairs else 0.0,
    }
    return pd.DataFrame(rows), overall


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def disagreements(existing: pd.Series, results: dict[int, RowResult]) -> list[int]:
    out = []
    for idx, truth in existing.items():
        r = results.get(idx)
        if truth is None or r is None or r.error:
            continue
        if truth != {l for l in r.labels if l != NONE_LABEL}:
            out.append(idx)
    return out
