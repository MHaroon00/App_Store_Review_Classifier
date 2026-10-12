"""Loading CSV/Excel files and sanity-checking them."""

from __future__ import annotations

import io

import pandas as pd

TEXT_COLUMN_HINTS = ("content", "review", "review_text", "text", "comment", "body", "feedback", "message")


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """Read CSV (any common encoding / delimiter) or Excel into a DataFrame."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data))
    else:
        last_exc = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
                break
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                last_exc = exc
        else:
            raise ValueError(f"Could not read the file: {last_exc}")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise ValueError("The file has no rows.")
    return df


def guess_text_column(df: pd.DataFrame) -> str:
    text_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
    if not text_cols:
        return df.columns[0]
    for hint in TEXT_COLUMN_HINTS:
        for c in text_cols:
            if c.lower() == hint:
                return c
    lengths = {c: df[c].dropna().astype(str).str.len().mean() for c in text_cols}
    return max(lengths, key=lambda c: lengths[c] if pd.notna(lengths[c]) else 0)


def data_check(df: pd.DataFrame, text_col: str) -> dict:
    text = df[text_col].fillna("").astype(str).str.strip()
    words = text.str.split().str.len()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "empty": int((text == "").sum()),
        "duplicates": int(text[text != ""].duplicated().sum()),
        "avg_words": round(float(words.mean()), 1) if len(words) else 0.0,
        "very_short": int(((words > 0) & (words < 3)).sum()),
    }


def candidate_label_columns(df: pd.DataFrame, text_col: str, max_unique: int = 30) -> list[str]:
    """Columns that look like existing labels (few distinct values) — for optional comparison."""
    cands = []
    for c in df.columns:
        if c == text_col:
            continue
        nun = df[c].nunique(dropna=True)
        if 1 <= nun <= max_unique and nun < max(len(df) * 0.5, 2):
            cands.append(c)
    return cands
