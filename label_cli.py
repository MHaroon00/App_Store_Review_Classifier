#!/usr/bin/env python3
"""Label a review CSV from the terminal (same engine as the web app).

Good for long runs: progress is checkpointed, so if you hit a free-tier limit
just run the same command again later and it continues where it stopped.

Example
-------
  export GROQ_API_KEY="gsk_..."
  python label_cli.py data/reviews.csv sample_data/labels_general.json \
      --provider groq --model openai/gpt-oss-120b --out data/reviews_labeled.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from labeler import metrics
from labeler.data import guess_text_column, read_table
from labeler.engine import RowResult, label_dataframe, results_to_frame
from labeler.export import to_csv_bytes
from labeler.llm import LLMClient
from labeler.prompts import LabelSet
from labeler.providers import PROVIDERS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Label reviews with your own labels using a free LLM.")
    p.add_argument("csv", help="Input CSV/XLSX with a review-text column")
    p.add_argument("labels", help="Label set JSON (save one from the web app, or see sample_data/)")
    p.add_argument("--provider", choices=list(PROVIDERS), default="groq")
    p.add_argument("--model", default=None, help="Model id (default: provider's first suggestion)")
    p.add_argument("--base-url", default=None, help="Override the provider URL")
    p.add_argument("--text-col", default=None, help="Column with the review text (auto-detected)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--rpm", type=int, default=None, help="Max requests per minute")
    p.add_argument("--limit", type=int, default=None, help="Only label the first N rows (testing)")
    p.add_argument("--out", default=None, help="Output CSV (default: <input>_labeled.csv)")
    args = p.parse_args(argv)

    prov = PROVIDERS[args.provider]
    key = os.environ.get(prov.secret_name, "")
    if prov.needs_key and not key:
        print(f"Set your free key first:  export {prov.secret_name}=...   (get one at {prov.key_url})")
        return 2

    src = Path(args.csv)
    df = read_table(src.read_bytes(), src.name)
    ls = LabelSet.from_dict(json.loads(Path(args.labels).read_text(encoding="utf-8")))
    if err := ls.validate():
        print(f"Label set problem: {err}")
        return 2
    text_col = args.text_col or guess_text_column(df)
    out_path = Path(args.out or src.with_name(src.stem + "_labeled.csv"))
    ckpt = out_path.with_suffix(".checkpoint.json")

    results: dict[int, RowResult] = {}
    if ckpt.exists():
        saved = json.loads(ckpt.read_text(encoding="utf-8"))
        results = {int(k): RowResult(**v) for k, v in saved.items()}
        print(f"Resuming: {len(results)} rows already labeled.")

    def save(_new=None):
        ckpt.write_text(json.dumps({k: vars(v) for k, v in results.items()}, ensure_ascii=False), encoding="utf-8")

    def show(done, total):
        sys.stdout.write(f"\r  {done}/{total} reviews labeled")
        sys.stdout.flush()

    client = LLMClient(args.base_url or prov.base_url, key, args.model or (prov.models[0] if prov.models else ""),
                       requests_per_minute=args.rpm if args.rpm is not None else prov.requests_per_minute)
    print(f"Labeling '{text_col}' in {src.name} with {args.provider}:{client.model} · labels: {', '.join(ls.names())}")
    rows = list(df.index[: args.limit]) if args.limit else None
    results, report = label_dataframe(df, text_col, ls, client, results=results,
                                      batch_size=args.batch_size or prov.batch_size, row_indices=rows,
                                      progress=show, on_batch=save)
    save()
    print()
    out_path.write_bytes(to_csv_bytes(results_to_frame(df, results, ls)))
    print(f"Saved → {out_path}  ({report.labeled} labeled, {report.errors} errors, {client.usage.requests} requests)")
    if report.status != "done":
        print(f"Paused: {report.message}\nRun the same command again later to continue.")
        return 1

    print("\nLabel presence:")
    for r in metrics.presence_table(results, ls.names()).itertuples():
        print(f"  {'✓' if r.present else '✗'} {r.label:<28} {r.reviews:>6}  ({r.share:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
