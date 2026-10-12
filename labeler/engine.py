"""Batch-label a table of reviews. Resumable: pass in the results you already have."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .llm import AuthError, ModelError, QuotaError
from .prompts import MULTI, NONE_LABEL, LabelSet, build_system_prompt, build_user_prompt, parse_results


@dataclass
class RowResult:
    labels: list[str] = field(default_factory=list)
    confidence: str = ""
    evidence: str = ""
    warning: str = ""
    error: str = ""


@dataclass
class RunReport:
    status: str = "done"          # done | paused_quota | auth_error | stopped
    message: str = ""
    labeled: int = 0
    errors: int = 0


ProgressFn = Callable[[int, int], None]   # (completed, total)


def label_dataframe(
    df: pd.DataFrame,
    text_col: str,
    labelset: LabelSet,
    client,
    results: dict[int, RowResult] | None = None,
    batch_size: int = 15,
    row_indices: list[int] | None = None,
    progress: ProgressFn | None = None,
    on_batch: Callable[[dict[int, RowResult]], None] | None = None,
) -> tuple[dict[int, RowResult], RunReport]:
    """Label rows of ``df`` (by index) that are not yet in ``results``.

    Results are written into the dict as each batch finishes, so if the run is
    interrupted (quota, closed tab, rerun) nothing already labeled is lost.
    """
    results = results if results is not None else {}
    report = RunReport()
    indices = row_indices if row_indices is not None else list(df.index)
    total = len(indices)

    pending = []
    for idx in indices:
        if idx in results and not results[idx].error:
            continue
        text = df.at[idx, text_col]
        if _is_blank(text):
            results[idx] = RowResult(labels=[] if labelset.mode == MULTI else [NONE_LABEL], warning="empty review text")
            continue
        pending.append(idx)

    system = build_system_prompt(labelset)
    done = total - len(pending)
    if progress:
        progress(done, total)

    queue = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    while queue:
        batch = queue.pop(0)
        local = list(range(1, len(batch) + 1))
        payload = [
            (lid, str(df.at[idx, text_col]), {c: df.at[idx, c] for c in labelset.context_columns if c in df.columns})
            for lid, idx in zip(local, batch)
        ]
        try:
            data = client.chat_json(system, build_user_prompt(payload))
            parsed = parse_results(data, local, labelset)
        except QuotaError as exc:
            report.status, report.message = "paused_quota", str(exc)
            break
        except AuthError as exc:
            report.status, report.message = "auth_error", str(exc)
            break
        except ModelError as exc:
            if len(batch) > 1:          # retry the rows one by one
                queue = [[i] for i in batch] + queue
                continue
            results[batch[0]] = RowResult(error=str(exc))
            done += 1
            report.errors += 1
            if progress:
                progress(done, total)
            continue

        missing = []
        new_rows = {}
        for lid, idx in zip(local, batch):
            row = parsed.get(lid)
            if row is None:
                missing.append(idx)
                continue
            new_rows[idx] = RowResult(row.labels, row.confidence, row.evidence, row.warning)
        if missing and len(batch) > 1:
            queue = [[i] for i in missing] + queue           # ask again individually
        elif missing:
            new_rows[missing[0]] = RowResult(error="model returned no result for this review")
            report.errors += 1

        results.update(new_rows)
        done += len(new_rows)
        if on_batch:
            on_batch(new_rows)
        if progress:
            progress(done, total)

    report.labeled = sum(1 for i in indices if i in results and not results[i].error)
    report.errors = sum(1 for i in indices if i in results and results[i].error)
    if report.status == "done" and report.labeled + report.errors < total:
        report.status = "stopped"
    return results, report


def results_to_frame(df: pd.DataFrame, results: dict[int, RowResult], labelset: LabelSet) -> pd.DataFrame:
    """Original columns + label columns. Unprocessed rows stay blank."""
    out = df.copy()
    names = labelset.names()
    status = []
    if labelset.mode == MULTI:
        for name in names:
            out[label_column(name)] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["labels"] = ""
        out["label_count"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out["label"] = ""
    out["confidence"] = ""
    out["evidence"] = ""
    out["warning"] = ""
    out["error"] = ""

    for idx in out.index:
        r = results.get(idx)
        if r is None:
            status.append("not labeled")
            continue
        if r.error:
            out.at[idx, "error"] = r.error
            status.append("error")
            continue
        status.append("labeled")
        if labelset.mode == MULTI:
            for name in names:
                out.at[idx, label_column(name)] = int(name in r.labels)
            out.at[idx, "labels"] = "; ".join(r.labels)
            out.at[idx, "label_count"] = len(r.labels)
        else:
            out.at[idx, "label"] = r.labels[0] if r.labels else NONE_LABEL
        out.at[idx, "confidence"] = r.confidence
        out.at[idx, "evidence"] = r.evidence
        out.at[idx, "warning"] = r.warning
    out["status"] = status
    if not labelset.include_evidence:
        out = out.drop(columns=["evidence"])
    return out


def label_column(name: str) -> str:
    slug = re.sub(r"[^\w]+", "_", name.strip().lower(), flags=re.UNICODE).strip("_")
    return f"has_{slug or 'label'}"


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or not str(v).strip() or str(v).strip().lower() == "nan"
