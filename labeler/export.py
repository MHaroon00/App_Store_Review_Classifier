"""CSV / Excel / JSON downloads."""

from __future__ import annotations

import io
import json
import re

import pandas as pd

_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")   # BOM → Excel shows Urdu/emoji correctly


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for name, frame in sheets.items():
            clean = frame.copy()
            for col in clean.columns:
                clean[col] = clean[col].map(lambda v: _ILLEGAL.sub("", v) if isinstance(v, str) else v)
            clean.to_excel(xl, sheet_name=name[:31], index=False)
            ws = xl.sheets[name[:31]]
            for i, col in enumerate(clean.columns, 1):
                width = min(70, max(10, int(clean[col].astype(str).str.len().quantile(0.9)) if len(clean) else 10, len(str(col)) + 2))
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
            ws.freeze_panes = "A2"
    return buf.getvalue()


def to_json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
