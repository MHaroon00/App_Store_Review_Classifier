"""Review Labeler — label any review CSV with YOUR labels using a free LLM.

Run locally:   streamlit run app.py
Deploy:        push to GitHub → share.streamlit.io → main file = app.py
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from labeler import metrics  # noqa: E402
from labeler.data import candidate_label_columns, data_check, guess_text_column, read_table  # noqa: E402
from labeler.engine import label_column, label_dataframe, results_to_frame  # noqa: E402
from labeler.export import to_csv_bytes, to_excel_bytes, to_json_bytes  # noqa: E402
from labeler.llm import LLMClient  # noqa: E402
from labeler.prompts import MULTI, NONE_LABEL, SINGLE, LabelSet, LabelSpec, build_system_prompt, parse_label_text  # noqa: E402
from labeler.providers import PROVIDERS  # noqa: E402

ROOT = Path(__file__).parent
SAMPLES = ROOT / "sample_data"
ACCENT = "#1faa7c"
TEXT = "#e9eef4"

st.set_page_config(page_title="Review Labeler", page_icon=":material/label:", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

ss = st.session_state
for k, v in {
    "df": None, "file_name": "", "editor_ver": 0, "ls_mode": MULTI, "ls_instructions": "",
    "labels_df": pd.DataFrame([{"name": "", "definition": "", "examples": ""}]),
    "run": None, "fetched_models": {}, "conn": {}, "scope": "test", "last_notice": None,
}.items():
    ss.setdefault(k, v)


# ═════════════════════════════ helpers ══════════════════════════════════════

def esc(v) -> str:
    return html.escape(str(v if v is not None else ""))


def ic(name: str, cls: str = "") -> str:
    return f"<span class='ms {cls}'>{name}</span>"


def html_block(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def card_head(icon: str, kicker: str, title: str, desc: str, right: str = "") -> None:
    html_block(f"""<div class="card-head"><div class="ico">{ic(icon)}</div>
        <div class="meta"><div class="k">{esc(kicker)}</div><h3>{esc(title)}</h3><p>{desc}</p></div>
        {f'<div class="right">{right}</div>' if right else ''}</div>""")


def subhead(icon: str, text: str, small: str = "") -> None:
    html_block(f"<div class='subhead'>{ic(icon)}{esc(text)}{f'<small>· {esc(small)}</small>' if small else ''}</div>")


def notice(kind: str, icon: str, text: str) -> None:
    html_block(f"<div class='notice {kind}'>{ic(icon, 'fill')}<p>{text}</p></div>")


def secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "")


# ═════════════════════════════ callbacks ════════════════════════════════════

def load_labelset_into_editor(ls: LabelSet) -> None:
    ss.labels_df = pd.DataFrame([{"name": l.name, "definition": l.definition, "examples": l.examples} for l in ls.labels]
                                or [{"name": "", "definition": "", "examples": ""}])
    ss.ls_mode = ls.mode if ls.mode in (MULTI, SINGLE) else MULTI
    ss.ls_instructions = ls.instructions
    ss.editor_ver += 1


def load_sample_data() -> None:
    path = SAMPLES / "reviews_sample.csv"
    ss.df = read_table(path.read_bytes(), path.name)
    ss.file_name = path.name


def load_sample_labels(fname: str) -> None:
    load_labelset_into_editor(LabelSet.from_dict(json.loads((SAMPLES / fname).read_text(encoding="utf-8"))))
    st.toast("Example labels loaded", icon=":material/category:")


def apply_pasted_labels() -> None:
    specs = parse_label_text(ss.get("paste_text", ""))
    if specs:
        load_labelset_into_editor(LabelSet(specs, ss.ls_mode, ss.ls_instructions))
        st.toast(f"{len(specs)} labels added", icon=":material/content_paste:")


def import_label_json() -> None:
    f = ss.get("labels_json")
    if f is None:
        return
    try:
        load_labelset_into_editor(LabelSet.from_dict(json.loads(f.getvalue())))
        st.toast("Label set imported", icon=":material/upload:")
    except Exception as exc:
        st.toast(f"Not a valid label set: {exc}", icon=":material/error:")


def clear_results() -> None:
    ss.run = None
    ss.last_notice = None


# ═════════════════════════════ SIDEBAR ══════════════════════════════════════

with st.sidebar:
    html_block(f"""<div class="brand"><div class="logo">{ic('label', 'fill')}</div>
        <div><div class="name">Review Labeler</div><div class="sub">Free LLM · your labels</div></div></div>""")

    pkey = st.selectbox("AI provider", list(PROVIDERS), format_func=lambda k: PROVIDERS[k].label, key="provider")
    prov = PROVIDERS[pkey]
    link = ""
    if prov.key_url:
        link = f" <a href='{prov.key_url}' target='_blank'>{'Get a free key' if prov.needs_key else 'Download'} →</a>"
    html_block(f"<div class='side-note'>{ic('info')}{esc(prov.note)}{link}</div>")

    base_url = prov.base_url
    if pkey in ("custom", "ollama"):
        base_url = st.text_input("Base URL", value=prov.base_url, placeholder="https://…/v1", key=f"base_{pkey}")

    stored_key = secret(prov.secret_name)
    api_key = ""
    needs_key = prov.needs_key or pkey == "custom"
    if needs_key:
        typed = st.text_input("API key", type="password", key=f"key_{pkey}", placeholder="Paste your free key here",
                              help="Used only in this browser session. It is never saved or logged.")
        api_key = typed.strip() or stored_key
        key_pill = ("<span class='pill ok'>" + ic("check_circle", "fill") + "Ready</span>") if api_key else \
                   ("<span class='pill warn'>" + ic("key") + "Needed</span>")
        html_block(f"<div class='side-h' style='margin-top:-.4rem'><span class='l' style='font-weight:500;color:var(--muted)'>"
                   f"Key status</span>{key_pill}</div>")

    fetched = ss.fetched_models.get(pkey, [])
    options = list(dict.fromkeys(list(prov.models) + fetched)) + ["Other…"]
    choice = st.selectbox("Model", options, key=f"model_{pkey}")
    model = st.text_input("Model id", key=f"model_custom_{pkey}", placeholder="exact model id") if choice == "Other…" else choice

    b1, b2 = st.columns(2)
    if b1.button("Models", icon=":material/sync:", width="stretch", help="Fetch the live list of models"):
        try:
            ids = LLMClient(base_url, api_key, model or "x", max_retries=1).list_models()
            if pkey == "openrouter":
                ids = [m for m in ids if m.endswith(":free")]
            ss.fetched_models[pkey] = ids
            st.toast(f"Found {len(ids)} models", icon=":material/check_circle:")
            st.rerun()
        except Exception as exc:
            ss.conn[pkey] = ("bad", f"Couldn't fetch models: {exc}")
    if b2.button("Test", icon=":material/bolt:", width="stretch", help="Send one tiny request to check key + model"):
        try:
            t0 = time.time()
            LLMClient(base_url, api_key, model, max_retries=1).chat_json("Reply with JSON only.", 'Return {"ok": true}')
            ss.conn[pkey] = ("ok", f"Connected to {model} in {time.time() - t0:.1f}s")
        except Exception as exc:
            ss.conn[pkey] = ("bad", str(exc))
    if pkey in ss.conn:
        kind, msg = ss.conn[pkey]
        html_block(f"<div class='notice {kind}' style='margin-top:.6rem'>{ic('check_circle' if kind == 'ok' else 'error', 'fill')}"
                   f"<p style='font-size:.8rem'>{esc(msg)}</p></div>")

    with st.expander("Speed & limits", icon=":material/tune:"):
        batch_size = st.slider("Reviews per request", 1, 40, prov.batch_size, key=f"batch_{pkey}",
                               help="Bigger batches = fewer requests (good for free daily limits).")
        rpm = st.number_input("Max requests per minute", 0, 600, prov.requests_per_minute, key=f"rpm_{pkey}",
                              help="Keeps you under the free-tier limit. 0 = no limit.")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1, help="0 = most consistent labels (recommended).")

    html_block(f"<div class='side-note'>{ic('shield_lock')}Your key stays in this browser session only and is "
               "cleared when you close the tab.</div>")


# ═════════════════════════════ HERO ═════════════════════════════════════════

hero_l, hero_r = st.columns([1.45, 1], gap="large", vertical_alignment="center")
with hero_l:
    html_block(f"""
    <div class="hero">
      <span class="eyebrow"><span class="dot"></span>Free LLM · your labels · your data</span>
      <h1>Review <em>Labeler</em></h1>
      <p class="lead">Upload reviews as plain text, describe the labels you care about, and a free AI model checks
      which labels are present in every review — then labels the whole dataset for your research.</p>
      <div class="chips">
        <span class="chip">{ic('table_view')}No label column needed</span>
        <span class="chip">{ic('edit_note')}Your own definitions</span>
        <span class="chip">{ic('savings')}Free models</span>
        <span class="chip">{ic('balance')}Agreement (κ)</span>
      </div>
    </div>""")
session_slot = hero_r.empty()
stepper_slot = st.empty()


# ═════════════════════════════ 01 UPLOAD ════════════════════════════════════

with st.container(key="card_upload"):
    df: pd.DataFrame | None = ss.df
    chip = f"<span class='file-chip'>{ic('description')}{esc(ss.file_name)}</span>" if df is not None else ""
    card_head("upload_file", "STEP 01", "Upload your reviews",
              "Any CSV or Excel file works — plain review text is enough, no label column is needed. "
              "The review column is detected automatically and the data is checked for empty and duplicate rows.", chip)

    u1, u2 = st.columns([4, 1.1], vertical_alignment="center", gap="medium")
    uploaded = u1.file_uploader("Drop a CSV or Excel file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    u2.button("Use sample data", icon=":material/dataset:", on_click=load_sample_data, width="stretch", type="secondary")

    if uploaded is not None and uploaded.name != ss.file_name:
        try:
            ss.df = read_table(uploaded.getvalue(), uploaded.name)
            ss.file_name = uploaded.name
            st.rerun()
        except Exception as exc:
            notice("bad", "error", f"Couldn't read that file: {esc(exc)}")

    df = ss.df
    text_col, ctx_cols = None, []
    if df is not None:
        cols = list(df.columns)
        g = guess_text_column(df)
        c1, c2 = st.columns(2, gap="medium")
        text_col = c1.selectbox("Review text column", cols, index=cols.index(g), key=f"textcol_{ss.file_name}")
        ctx_cols = c2.multiselect("Extra context for the model (optional)", [c for c in cols if c != text_col],
                                  key=f"ctx_{ss.file_name}", help="e.g. app name or star rating")
        chk = data_check(df, text_col)
        html_block(f"""<div class="kpis">
          <div class="kpi"><div class="top"><span class="label">Reviews</span><span class="ico">{ic('table_rows')}</span></div>
            <div class="value">{chk['rows']:,}</div><div class="note">rows loaded</div></div>
          <div class="kpi"><div class="top"><span class="label">Columns</span><span class="ico">{ic('view_column')}</span></div>
            <div class="value">{chk['columns']}</div><div class="note">in the file</div></div>
          <div class="kpi"><div class="top"><span class="label">Empty</span><span class="ico {'warn' if chk['empty'] else ''}">{ic('block')}</span></div>
            <div class="value">{chk['empty']:,}</div><div class="note">skipped automatically</div></div>
          <div class="kpi"><div class="top"><span class="label">Duplicates</span><span class="ico {'warn' if chk['duplicates'] else ''}">{ic('content_copy')}</span></div>
            <div class="value">{chk['duplicates']:,}</div><div class="note">identical texts</div></div>
          <div class="kpi"><div class="top"><span class="label">Avg length</span><span class="ico">{ic('short_text')}</span></div>
            <div class="value">{chk['avg_words']:.0f}</div><div class="note">words · {chk['very_short']} very short</div></div>
        </div>""")
        with st.expander(f"Preview data · first {min(len(df), 50)} of {len(df):,} rows", icon=":material/table_view:"):
            st.dataframe(df.head(50), width="stretch", hide_index=True)


# ═════════════════════════════ 02 LABELS ════════════════════════════════════

with st.container(key="card_labels"):
    card_head("sell", "STEP 02", "Define your labels",
              "Write each label with a clear definition — the model reads exactly what you write here. "
              "Start from an example, paste a list, or import a label set you saved earlier.")

    with st.container(horizontal=True, gap="small", vertical_alignment="center", key="label_tools"):
        st.button("General example", icon=":material/category:", type="tertiary",
                  on_click=load_sample_labels, args=("labels_general.json",))
        st.button("Usability (Nielsen)", icon=":material/accessibility_new:", type="tertiary",
                  on_click=load_sample_labels, args=("labels_usability_nielsen.json",))
        with st.popover("Paste list", icon=":material/content_paste:"):
            st.text_area("One label per line — Label: definition", key="paste_text", height=150,
                         placeholder="Usability: ease of use, navigation, confusing design\nPerformance: crashes, lag, battery")
            st.button("Use these labels", icon=":material/done_all:", type="primary", on_click=apply_pasted_labels, width="stretch")
        with st.popover("Import .json", icon=":material/upload:"):
            st.file_uploader("Saved label set", type=["json"], key="labels_json", on_change=import_label_json)

    edited = st.data_editor(
        ss.labels_df, key=f"label_editor_{ss.editor_ver}", num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Label", required=True, width="small"),
            "definition": st.column_config.TextColumn("Definition — what counts as this label?", width="large"),
            "examples": st.column_config.TextColumn("Example phrase (optional)", width="medium"),
        },
    )

    m1, m2 = st.columns(2, gap="large")
    with m1:
        subhead("tune", "Labeling mode")
        mode = st.segmented_control(
            "Labeling mode", [MULTI, SINGLE], key="ls_mode", required=True, label_visibility="collapsed",
            format_func=lambda m: ":material/checklist: Which labels are present" if m == MULTI else ":material/radio_button_checked: One best label",
        ) or MULTI
        html_block("<p class='just' style='color:var(--muted);font-size:.85rem;margin:.5rem 0 .6rem'>"
                   + ("Each review can receive zero, one or several labels — ideal for checking whether each label is present."
                      if mode == MULTI else "Each review receives exactly one label, or NONE when no label fits.") + "</p>")
        include_evidence = st.toggle("Ask for a short evidence quote", value=True,
                                     help="Helps you verify each label. Turn off to save tokens.")
    with m2:
        subhead("edit_note", "Your prompt", "optional")
        instructions = st.text_area("Extra instructions", key="ls_instructions", height=132, label_visibility="collapsed",
                                    placeholder="e.g. These are fitness app reviews. Only mark Usability when the user talks "
                                                "about the interface itself, not about bugs.")

    labelset = LabelSet(
        labels=[LabelSpec(str(r.get("name") or "").strip(), str(r.get("definition") or "").strip(), str(r.get("examples") or "").strip())
                for r in edited.to_dict("records") if str(r.get("name") or "").strip()],
        mode=mode, instructions=instructions, include_evidence=include_evidence, context_columns=list(ctx_cols),
    )
    ls_error = labelset.validate()

    p1, p2 = st.columns([3, 1], vertical_alignment="center", gap="medium")
    with p1.expander("Preview the exact prompt sent to the model", icon=":material/visibility:"):
        html_block(f"<div class='prompt-box'>{esc(build_system_prompt(labelset))}</div>")
    p2.download_button("Save label set", to_json_bytes(labelset.to_dict()), "label_set.json", "application/json",
                       icon=":material/save:", disabled=bool(ls_error), width="stretch")


# ═════════════════════════════ readiness, stepper & session panel ═══════════

sig = None
if df is not None and not ls_error:
    sig_src = json.dumps({"f": ss.file_name, "n": len(df), "t": text_col, "ls": labelset.to_dict()}, sort_keys=True)
    sig = hashlib.sha1(sig_src.encode()).hexdigest()
run = ss.run
same_run = bool(run and sig and run["sig"] == sig)
done_n = len([r for r in run["results"].values() if not r.error]) if same_run else 0
all_done = bool(df is not None and same_run and done_n >= len(df))

checks = [
    ("Reviews uploaded", df is not None),
    (f"{len(labelset.labels)} label{'s' if len(labelset.labels) != 1 else ''} defined" if not ls_error else "Labels defined", not ls_error),
    ("API key entered", bool(api_key) or not needs_key),
    ("Model selected", bool(model)),
]
ready = all(ok for _, ok in checks)

flags = [df is not None, not ls_error, all_done, bool(run and run["results"])]
active = next((i for i, f in enumerate(flags) if not f), None)
steps = [("upload_file", "Step 1", "Upload"), ("sell", "Step 2", "Labels"), ("auto_awesome", "Step 3", "Label"), ("insights", "Step 4", "Results")]
parts = []
for i, (icon, k, t) in enumerate(steps):
    state = "done" if flags[i] else ("active" if i == active else "")
    parts.append(f"<div class='sstep {state}'><div class='bubble'>{ic('check' if flags[i] else icon)}</div>"
                 f"<div class='txt'><div class='k'>{k}</div><div class='t'>{t}</div></div></div>")
    if i < len(steps) - 1:
        parts.append(f"<div class='sline {'done' if flags[i] else ''}'></div>")
stepper_slot.markdown(f"<div class='stepper'>{''.join(parts)}</div>", unsafe_allow_html=True)

key_pill = ("<span class='pill ok'>" + ic("check_circle", "fill") + "Ready</span>") if (api_key or not needs_key) \
    else ("<span class='pill warn'>" + ic("key") + "Add in sidebar</span>")
session_slot.markdown(f"""<div class="session"><div class="title">{ic('dashboard')}Session</div>
  <div class="srow"><span class="k">{ic('hub')}Provider</span><span class="v">{esc(prov.label.split(' (')[0])}</span></div>
  <div class="srow"><span class="k">{ic('neurology')}Model</span><span class="v">{esc(model or '—')}</span></div>
  <div class="srow"><span class="k">{ic('key')}API key</span>{key_pill}</div>
  <div class="srow"><span class="k">{ic('reviews')}Reviews</span><span class="v">{f'{len(df):,}' if df is not None else '—'}</span></div>
  <div class="srow"><span class="k">{ic('sell')}Labels</span><span class="v">{len(labelset.labels) or '—'}</span></div>
  <div class="srow"><span class="k">{ic('task_alt')}Labeled</span><span class="v">{f'{done_n:,} / {len(df):,}' if same_run else '—'}</span></div>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════ 03 RUN ═══════════════════════════════════════

with st.container(key="card_run"):
    card_head("auto_awesome", "STEP 03", "Label the dataset",
              "Test on 10 reviews first to check your labels, then label everything. Reviews are sent in batches, "
              "progress is saved after every batch, and you can continue after a free-tier pause without losing work.")

    html_block("<div class='checks'>" + "".join(
        f"<div class='check {'ok' if ok else 'no'}'>{ic('check_circle' if ok else 'radio_button_unchecked', 'fill' if ok else '')}{esc(t)}</div>"
        for t, ok in checks) + "</div>")

    if run and sig and not same_run and run["results"]:
        notice("", "info", "You changed the data or the labels since the last run. Starting again labels with the new settings — "
               "download the previous results first if you still need them.")

    n_total = len(df) if df is not None else 0
    r1, r2 = st.columns([1.15, 2], vertical_alignment="bottom", gap="medium")
    with r1:
        scope = st.segmented_control(
            "Scope", ["test", "all"], key="scope", required=True,
            format_func=lambda s: ":material/science: Test on 10" if s == "test" else f":material/all_inclusive: All {n_total:,} reviews" if n_total else ":material/all_inclusive: All reviews",
        ) or "test"
    remaining = n_total - done_n
    n_failed = sum(1 for r in run["results"].values() if r.error) if same_run else 0
    finished = False
    if same_run and done_n >= n_total:
        go_label, go_icon, finished = "All reviews labeled", ":material/task_alt:", True
    elif same_run and n_failed and done_n + n_failed >= n_total and scope == "all":
        go_label, go_icon = f"Retry {n_failed} failed review{'s' if n_failed != 1 else ''}", ":material/refresh:"
    elif same_run and 0 < done_n < n_total and scope == "all":
        go_label, go_icon = f"Continue labeling · {remaining:,} left", ":material/resume:"
    elif same_run and scope == "test" and done_n >= min(10, n_total):
        go_label, go_icon, finished = "Test done — choose All reviews", ":material/task_alt:", True
    else:
        go_label, go_icon = "Start labeling", ":material/play_arrow:"
    go = r2.button(go_label, icon=go_icon, type="primary", width="stretch", disabled=not ready or finished, key="go")

    status_slot = st.empty()
    progress_slot = st.empty()
    feed_slot = st.empty()

    if go and ready:
        if not same_run:
            ss.run = run = {"sig": sig, "results": {}, "labelset": labelset, "text_col": text_col,
                            "models": [], "started": datetime.now().strftime("%Y-%m-%d %H:%M"), "tokens": 0, "requests": 0}
        tag = f"{pkey}:{model}"
        if tag not in run["models"]:
            run["models"].append(tag)
        indices = list(df.index[:10]) if scope == "test" else list(df.index)
        client = LLMClient(base_url, api_key, model, temperature=temperature, requests_per_minute=int(rpm))

        status_slot.markdown(f"""<div class="runbox">{ic('progress_activity', 'spin')}
            <div><div class="t">The AI is reading your reviews…</div>
            <div class="s">{esc(model)} · {int(batch_size)} reviews per request · keep this tab open</div></div></div>""",
                             unsafe_allow_html=True)
        bar = progress_slot.progress(0.0, text="Starting…")
        feed: list[tuple[str, list[str]]] = []
        t0 = time.time()
        start_done = sum(1 for i in indices if i in run["results"] and not run["results"][i].error)

        def on_progress(done: int, total: int) -> None:
            el = time.time() - t0
            eta = ""
            if start_done < done < total:
                left = (el / (done - start_done)) * (total - done)
                eta = f" · about {left:.0f}s left" if left < 90 else f" · about {left / 60:.0f} min left"
            bar.progress(done / total if total else 1.0, text=f"{done:,} of {total:,} reviews{eta}")

        def on_batch(new_rows) -> None:
            for idx, r in list(new_rows.items())[-3:]:
                feed.insert(0, (str(df.at[idx, text_col]), r.labels))
            del feed[3:]
            items = "".join(
                f"""<div class="feed-item">{ic('sell', 'fill')}<div class="txt">{esc(t[:220])}{'…' if len(t) > 220 else ''}
                    <div class="tags">{''.join(f"<span class='tag'>{esc(l)}</span>" for l in labs if l != NONE_LABEL)
                                       or "<span class='tag none'>no label</span>"}</div></div></div>"""
                for t, labs in feed)
            feed_slot.markdown(f"<div class='feed'>{items}</div>", unsafe_allow_html=True)

        results, report = label_dataframe(df, text_col, labelset, client, results=run["results"], batch_size=int(batch_size),
                                          row_indices=indices, progress=on_progress, on_batch=on_batch)
        run["tokens"] += client.usage.total_tokens
        run["requests"] += client.usage.requests
        status_slot.empty()
        feed_slot.empty()
        secs = time.time() - t0
        if report.status == "done":
            bar.progress(1.0, text=f"Finished in {secs:.0f}s")
            if scope == "test" and n_total > 10:
                ss.last_notice = ("ok", "task_alt", f"Test finished — {report.labeled} reviews labeled in {secs:.0f}s. "
                                  "Check the results below, then choose <b>All reviews</b> and continue.")
            else:
                ss.last_notice = ("ok", "task_alt", f"All done — {report.labeled:,} reviews labeled"
                                  f"{f', {report.errors} errors' if report.errors else ''}.")
            st.toast("Labeling finished", icon=":material/celebration:")
        elif report.status == "paused_quota":
            ss.last_notice = ("", "pause_circle", f"<b>Free-tier limit reached — your progress is saved.</b> {esc(report.message)} "
                              "Wait a minute and press <b>Continue</b>, or pick another free model in the sidebar.")
        elif report.status == "auth_error":
            ss.last_notice = ("bad", "key_off", esc(report.message))
        st.rerun()

    if ss.last_notice:
        notice(*ss.last_notice)


# ═════════════════════════════ 04 RESULTS ═══════════════════════════════════

run = ss.run
if run and run["results"] and ss.df is not None:
    df = ss.df
    ls: LabelSet = run["labelset"]
    results = run["results"]
    names = ls.names()
    out = results_to_frame(df, results, ls)
    ok_rows = [r for r in results.values() if not r.error]
    n_ok, n_err = len(ok_rows), len(results) - len(ok_rows)
    n_none = metrics.none_count(results)
    n_low = sum(1 for r in ok_rows if r.confidence == "LOW")

    with st.container(key="card_results"):
        card_head("insights", "STEP 04", "Results",
                  f"Labeled with <b>{esc(', '.join(run['models']))}</b> · {run['requests']} request{'s' if run['requests'] != 1 else ''} · "
                  f"{run['tokens']:,} tokens. Every number below updates as you continue labeling.")
        html_block(f"""<div class="kpis">
          <div class="kpi"><div class="top"><span class="label">Labeled</span><span class="ico">{ic('task_alt')}</span></div>
            <div class="value">{n_ok:,}</div><div class="note">of {len(df):,} reviews</div></div>
          <div class="kpi"><div class="top"><span class="label">{'With a label' if ls.mode == MULTI else 'Matched'}</span><span class="ico">{ic('sell')}</span></div>
            <div class="value">{n_ok - n_none:,}</div><div class="note">{(n_ok - n_none) / max(n_ok, 1):.0%} of labeled</div></div>
          <div class="kpi"><div class="top"><span class="label">No label</span><span class="ico">{ic('label_off')}</span></div>
            <div class="value">{n_none:,}</div><div class="note">{'nothing present' if ls.mode == MULTI else 'marked NONE'}</div></div>
          <div class="kpi"><div class="top"><span class="label">Low confidence</span><span class="ico warn">{ic('help')}</span></div>
            <div class="value">{n_low:,}</div><div class="note">worth a manual check</div></div>
          <div class="kpi"><div class="top"><span class="label">Errors</span><span class="ico {'bad' if n_err else ''}">{ic('error')}</span></div>
            <div class="value">{n_err:,}</div><div class="note">{'retry with Continue' if n_err else 'all good'}</div></div>
        </div>""")

        tab_presence, tab_browse, tab_compare, tab_export = st.tabs([
            ":material/label: Label presence", ":material/manage_search: Browse reviews",
            ":material/compare_arrows: Compare", ":material/download: Download"])

        with tab_presence:
            pres = metrics.presence_table(results, names, include_none=(ls.mode == SINGLE))
            cards = "".join(
                f"""<div class="pcard" style="animation-delay:{i * .06:.2f}s">
                    <div class="name">{ic('sell')}<span>{esc(r.label)}</span></div>
                    <div class="row"><span class="num">{r.reviews:,}</span>
                      {"<span class='pill ok'>" + ic('check_circle', 'fill') + "Present</span>" if r.present
                       else "<span class='pill bad'>" + ic('cancel', 'fill') + "Not found</span>"}</div>
                    <div class="meter"><span style="width:{r.share * 100:.1f}%;animation-delay:{.2 + i * .06:.2f}s"></span></div>
                    <div class="pct">{r.share:.1%} of labeled reviews</div></div>"""
                for i, r in enumerate(pres.itertuples()))
            html_block(f"<div class='presence'>{cards}</div>")

            chart_df = pres.assign(text=pres.apply(lambda r: f"{r.reviews:,} · {r.share:.0%}", axis=1))
            base = alt.Chart(chart_df).encode(
                y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelColor=TEXT, labelFontSize=12, ticks=False, domain=False, labelLimit=220, labelPadding=10)),
                x=alt.X("reviews:Q", title=None, axis=None, scale=alt.Scale(domainMax=max(chart_df.reviews.max(), 1) * 1.3)),
                tooltip=[alt.Tooltip("label:N", title="Label"), alt.Tooltip("reviews:Q", title="Reviews", format=","),
                         alt.Tooltip("share:Q", title="Share", format=".1%")],
            )
            chart = (base.mark_bar(color=ACCENT, cornerRadiusEnd=5, height=20)
                     + base.mark_text(align="left", dx=8, color=TEXT, fontSize=12).encode(text="text:N"))
            subhead("bar_chart", "Labels ranked by frequency")
            st.altair_chart(chart.properties(height=max(140, 38 * len(chart_df)), background="transparent")
                            .configure_view(stroke=None).configure(font="Plus Jakarta Sans, sans-serif"), width="stretch", theme=None)

            if ls.mode == MULTI and n_ok:
                subhead("join", "Most common label combinations")
                combos = pd.Series(["; ".join(r.labels) if r.labels else "(no label)" for r in ok_rows]).value_counts().head(8)
                st.dataframe(combos.rename_axis("labels").reset_index(name="reviews"), hide_index=True, width="stretch",
                             column_config={"reviews": st.column_config.ProgressColumn("Reviews", format="%d", min_value=0,
                                                                                       max_value=int(combos.max()))})

        with tab_browse:
            f1, f2, f3 = st.columns([2, 1, 1], gap="medium")
            pick = f1.multiselect("Label", names + [NONE_LABEL if ls.mode == SINGLE else "(no label)"], placeholder="All labels")
            conf = f2.multiselect("Confidence", ["HIGH", "MEDIUM", "LOW"], placeholder="Any")
            status_pick = f3.selectbox("Status", ["labeled", "all", "error", "not labeled"])
            search = st.text_input("Search text", placeholder="Search inside reviews…", icon=":material/search:")
            view = out if status_pick == "all" else out[out["status"] == status_pick]
            if pick:
                lab_col = "labels" if ls.mode == MULTI else "label"

                def has(cell):
                    got = [x.strip() for x in str(cell).split(";") if x.strip()]
                    return any((p == "(no label)" and not got) or p in got for p in pick)
                view = view[view[lab_col].map(has)]
            if conf:
                view = view[view["confidence"].isin(conf)]
            if search.strip():
                view = view[view[run["text_col"]].astype(str).str.contains(search.strip(), case=False, regex=False)]
            st.caption(f"Showing {len(view):,} of {len(out):,} reviews")
            show = [run["text_col"]] + (["labels"] if ls.mode == MULTI else ["label"]) + ["confidence"] + \
                   (["evidence"] if ls.include_evidence else []) + ["warning", "error"]
            st.dataframe(view[show], width="stretch", hide_index=True, height=480,
                         column_config={run["text_col"]: st.column_config.TextColumn("Review", width="large")})

        with tab_compare:
            html_block("<p class='just' style='color:var(--muted)'>Already have human labels in the file? Compare them with "
                       "the model to measure agreement — precision, recall, F1 and Cohen's κ per label. Skip this step when "
                       "your CSV has no label column.</p>")
            cands = candidate_label_columns(df, run["text_col"])
            if not cands:
                notice("", "info", "No column in this file looks like an existing label column — nothing to compare.")
            else:
                cmp_col = st.selectbox("Existing label column", ["—"] + cands)
                if cmp_col != "—":
                    binary_label = None
                    if metrics.is_binary_column(df[cmp_col]):
                        binary_label = st.selectbox(f"'{cmp_col}' is a yes/no column. Which of your labels does it mark?", names)
                    existing = metrics.parse_existing(df[cmp_col], names, binary_label)
                    table, overall = metrics.agreement(existing, results, names if not binary_label else [binary_label])
                    if overall["compared"] == 0 or table.empty:
                        notice("", "warning", "None of the values in that column match your label names.")
                    else:
                        a, b = st.columns(2)
                        a.metric("Reviews compared", f"{overall['compared']:,}")
                        b.metric("Exact match", f"{overall['exact_match']:.1%}")
                        st.dataframe(table, hide_index=True, width="stretch", column_config={
                            c: st.column_config.NumberColumn(format="%.2f") for c in ["precision", "recall", "f1", "kappa", "agreement"]})
                        dis = metrics.disagreements(existing, results)
                        if dis:
                            subhead("rule", f"{len(dis)} disagreements", "worth a manual look")
                            d = out.loc[dis, [run["text_col"], cmp_col] + (["labels"] if ls.mode == MULTI else ["label"]) + ["confidence"]]
                            st.dataframe(d, hide_index=True, width="stretch")

        with tab_export:
            stem = f"labeled_{Path(ss.file_name).stem}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            pres = metrics.presence_table(results, names, include_none=(ls.mode == SINGLE))
            info = pd.DataFrame([
                ("file", ss.file_name), ("text_column", run["text_col"]), ("mode", ls.mode),
                ("models", ", ".join(run["models"])), ("started", run["started"]),
                ("reviews_total", len(df)), ("reviews_labeled", n_ok), ("errors", n_err),
                ("requests", run["requests"]), ("tokens", run["tokens"]), ("instructions", ls.instructions),
            ], columns=["setting", "value"])
            labels_sheet = pd.DataFrame([{"label": l.name, "definition": l.definition, "example": l.examples,
                                          "column": label_column(l.name) if ls.mode == MULTI else "label"} for l in ls.labels])
            html_block("<p class='just' style='color:var(--muted)'>Your original columns plus the new label columns "
                       + ("(one 0/1 column per label and a combined <code>labels</code> column)" if ls.mode == MULTI else "(<code>label</code>)")
                       + ", <code>confidence</code>" + (", <code>evidence</code>" if ls.include_evidence else "")
                       + ". Unlabeled rows stay blank. The Excel file also includes label presence, your definitions and run "
                         "details for your methodology section.</p>")
            e1, e2, e3 = st.columns(3, gap="medium")
            e1.download_button("CSV file", to_csv_bytes(out), f"{stem}.csv", "text/csv", icon=":material/csv:", width="stretch")
            e2.download_button("Excel workbook", to_excel_bytes(
                {"Labeled reviews": out, "Label presence": pres, "Label definitions": labels_sheet, "Run info": info}),
                f"{stem}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon=":material/grid_on:", width="stretch")
            e3.download_button("Label set (.json)", to_json_bytes(ls.to_dict()), "label_set.json", "application/json",
                               icon=":material/data_object:", width="stretch")
            st.write("")
            st.button("Clear results", icon=":material/delete_sweep:", type="tertiary", on_click=clear_results)

else:
    html_block(f"""<div class="subhead" style="margin-top:.4rem">{ic('school')}How it works</div>
    <div class="how">
      <div class="item d1"><div class="ico">{ic('upload_file')}</div><div class="n">01</div><h4>Upload</h4>
        <p>Load any CSV or Excel file of reviews. Plain text is enough — no existing labels are required.</p></div>
      <div class="item d2"><div class="ico">{ic('edit_note')}</div><div class="n">02</div><h4>Describe</h4>
        <p>Write your labels with short definitions and, optionally, your own instructions for the model.</p></div>
      <div class="item d3"><div class="ico">{ic('auto_awesome')}</div><div class="n">03</div><h4>Label</h4>
        <p>A free model reads the reviews in batches and decides which of your labels are present in each one.</p></div>
      <div class="item d4"><div class="ico">{ic('download')}</div><div class="n">04</div><h4>Analyse</h4>
        <p>See how often each label appears, browse the results and download a labeled CSV or Excel file.</p></div>
    </div>""")

html_block("<div class='foot-note'>LLM labels are a strong first pass, not ground truth. Check a random sample by hand "
           "and report the agreement (κ) in your methodology.</div>")
