import json
from types import SimpleNamespace

import openai
import pandas as pd
import pytest

from labeler import metrics
from labeler.data import candidate_label_columns, data_check, guess_text_column, read_table
from labeler.engine import RowResult, label_column, label_dataframe, results_to_frame
from labeler.export import to_csv_bytes, to_excel_bytes
from labeler.llm import LLMClient, ModelError, parse_json_object
from labeler.prompts import (MULTI, NONE_LABEL, SINGLE, LabelSet, LabelSpec, build_system_prompt,
                             build_user_prompt, parse_label_text, parse_results)
from labeler.providers import PROVIDERS
from tests.fakes import FakeLLM


# ── data loading ─────────────────────────────────────────────────────────────

def test_read_csv_without_label_column(sample_df):
    assert list(sample_df.columns) == ["review_id", "app_name", "score", "content"]
    assert guess_text_column(sample_df) == "content"
    chk = data_check(sample_df, "content")
    assert chk["rows"] == 30 and chk["empty"] == 0 and chk["very_short"] == 1


def test_read_semicolon_latin1_and_guess_by_length():
    raw = "id;opinion;stars\n1;Café app is très bien and easy to navigate;5\n2;Crashes;1\n".encode("cp1252")
    df = read_table(raw, "x.csv")
    assert list(df.columns) == ["id", "opinion", "stars"]
    assert guess_text_column(df) == "opinion"
    assert "Café" in df.at[0, "opinion"]


def test_read_excel(tmp_path):
    p = tmp_path / "r.xlsx"
    pd.DataFrame({"text": ["good", "bad"]}).to_excel(p, index=False)
    assert len(read_table(p.read_bytes(), "r.xlsx")) == 2


def test_empty_file_rejected():
    with pytest.raises(ValueError):
        read_table(b"text\n", "e.csv")


# ── label set & prompts ─────────────────────────────────────────────────────

def test_parse_pasted_labels():
    specs = parse_label_text("""
        1. Usability: ease of use
        - Performance — crashes and lag
        Pricing & Ads - cost, ads
        Support
    """)
    assert [s.name for s in specs] == ["Usability", "Performance", "Pricing & Ads", "Support"]
    assert specs[1].definition == "crashes and lag" and specs[3].definition == ""


def test_labelset_validation():
    assert LabelSet([]).validate()
    assert LabelSet([LabelSpec("A"), LabelSpec("a")]).validate() == "Label names must be unique."
    assert "NONE" in LabelSet([LabelSpec("None")]).validate()
    assert LabelSet([LabelSpec("A")], mode=SINGLE).validate()
    assert LabelSet([LabelSpec("A")], mode=MULTI).validate() == ""


def test_system_prompt_contains_user_labels_and_instructions(general_labels):
    p = build_system_prompt(general_labels)
    assert "1. Usability — Ease of use" in p
    assert general_labels.instructions in p
    assert "zero, one, or several" in p and '"labels"' in p
    single = build_system_prompt(LabelSet(general_labels.labels, mode=SINGLE))
    assert "exactly ONE label" in single and '"label"' in single


def test_user_prompt_truncates_and_adds_context():
    u = build_user_prompt([(1, "x" * 3000, {"app_name": "Strava", "score": 5})])
    assert "[id=1] (app_name: Strava | score: 5)" in u and "[truncated]" in u


def test_parse_results_normalises_and_flags_unknown():
    ls = LabelSet([LabelSpec("Pricing & Ads"), LabelSpec("Usability")])
    data = {"results": [
        {"id": 1, "labels": ["pricing and ads", "USABILITY", "Magic"], "confidence": "high"},
        {"id": "2", "labels": "Usability; none", "confidence": "??"},
        {"id": 9, "labels": ["Usability"]},
    ]}
    out = parse_results(data, [1, 2], ls)
    assert out[1].labels == ["Usability"] or out[1].labels == ["Pricing & Ads", "Usability"]
    assert "Magic" in out[1].warning and out[1].confidence == "HIGH"
    assert out[2].labels == ["Usability"] and out[2].confidence == ""
    assert 9 not in out


def test_parse_results_single_mode():
    ls = LabelSet([LabelSpec("A"), LabelSpec("B")], mode=SINGLE)
    out = parse_results({"results": [{"id": 1, "label": "b"}, {"id": 2, "label": "NONE"}, {"id": 3, "labels": ["A", "B"]}]}, [1, 2, 3], ls)
    assert out[1].labels == ["B"] and out[2].labels == [NONE_LABEL]
    assert out[3].labels == ["A"] and "kept the first" in out[3].warning


def test_parse_json_object_variants():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('<think>hmm</think>Sure! {"a": 2} hope this helps') == {"a": 2}
    assert parse_json_object('[{"id": 1}]') == {"results": [{"id": 1}]}
    with pytest.raises(ModelError):
        parse_json_object("no json here")


# ── engine ──────────────────────────────────────────────────────────────────

def test_multi_label_end_to_end(sample_df, general_labels):
    fake = FakeLLM()
    results, report = label_dataframe(sample_df, "content", general_labels, fake, batch_size=8)
    assert report.status == "done" and report.labeled == 30 and report.errors == 0
    assert fake.usage.requests == 4                     # 30 reviews / 8 per batch
    assert "Performance" in results[1].labels           # "crashes every time"
    assert set(results[24].labels) >= {"Pricing & Ads", "Customer Support"}   # R0025: charged + support
    out = results_to_frame(sample_df, results, general_labels)
    assert out.loc[1, "has_performance"] == 1 and out.loc[0, "has_pricing_ads"] == 0
    assert list(out.columns[:4]) == ["review_id", "app_name", "score", "content"]
    assert (out["status"] == "labeled").all()


def test_single_label_mode(sample_df, general_labels):
    ls = LabelSet(general_labels.labels, mode=SINGLE)
    results, _ = label_dataframe(sample_df, "content", ls, FakeLLM(), batch_size=10)
    assert all(len(r.labels) == 1 for r in results.values())
    out = results_to_frame(sample_df, results, ls)
    assert "label" in out.columns and out.loc[22, "label"] == NONE_LABEL   # "Good"


def test_quota_pause_and_resume_keeps_progress(sample_df, general_labels):
    results, report = label_dataframe(sample_df, "content", general_labels, FakeLLM(fail_after=2), batch_size=5)
    assert report.status == "paused_quota" and len(results) == 10
    fake2 = FakeLLM()
    results, report = label_dataframe(sample_df, "content", general_labels, fake2, results=results, batch_size=5)
    assert report.status == "done" and report.labeled == 30
    assert fake2.usage.requests == 4                    # only the 20 remaining rows


def test_missing_ids_are_retried_individually(sample_df, general_labels):
    fake = FakeLLM(drop_ids={2, 3})
    results, report = label_dataframe(sample_df.head(5), "content", general_labels, fake, batch_size=5)
    assert report.labeled == 5 and fake.usage.requests == 3


def test_bad_json_batch_falls_back_to_single_rows(sample_df, general_labels):
    fake = FakeLLM(bad_json_first=True)
    results, report = label_dataframe(sample_df.head(4), "content", general_labels, fake, batch_size=4)
    assert report.labeled == 4 and len(fake.calls) == 5


def test_empty_text_rows_are_skipped(general_labels):
    df = pd.DataFrame({"t": ["The app crashes", "", None, "  "]})
    fake = FakeLLM()
    results, report = label_dataframe(df, "t", general_labels, fake)
    assert fake.usage.requests == 1
    assert results[1].warning == "empty review text" and results[1].labels == []


def test_subset_of_rows_only(sample_df, general_labels):
    results, _ = label_dataframe(sample_df, "content", general_labels, FakeLLM(), row_indices=list(range(10)))
    assert sorted(results) == list(range(10))
    out = results_to_frame(sample_df, results, general_labels)
    assert (out["status"] == "not labeled").sum() == 20 and pd.isna(out.loc[20, "has_usability"])


def test_context_columns_are_sent(sample_df, general_labels):
    general_labels.context_columns = ["app_name", "score"]
    fake = FakeLLM()
    label_dataframe(sample_df.head(2), "content", general_labels, fake)
    assert "(app_name: Strava | score: 5)" in fake.calls[0][1]


# ── metrics ─────────────────────────────────────────────────────────────────

def test_presence_table_shows_absent_labels(sample_df, nielsen_labels):
    results, _ = label_dataframe(sample_df, "content", nielsen_labels, FakeLLM())
    pres = metrics.presence_table(results, nielsen_labels.names() + ["Accessibility"])
    row = pres.set_index("label")
    assert row.loc["Memorability", "present"] and row.loc["Memorability", "reviews"] >= 3
    assert not row.loc["Accessibility", "present"]


def test_agreement_with_binary_column(sample_df, general_labels):
    results, _ = label_dataframe(sample_df, "content", general_labels, FakeLLM())
    human = sample_df["content"].str.contains("crash|freez|battery|load", case=False).map({True: "Yes", False: "No"})
    assert metrics.is_binary_column(human)
    existing = metrics.parse_existing(human, general_labels.names(), binary_label="Performance")
    table, overall = metrics.agreement(existing, results, ["Performance"])
    assert overall["compared"] == 30
    assert table.loc[0, "recall"] == 1.0 and 0 < table.loc[0, "kappa"] <= 1


def test_agreement_with_label_name_column():
    ls_names = ["Usability", "Performance"]
    existing = metrics.parse_existing(pd.Series(["Usability", "Usability; Performance", "Performance", None]), ls_names)
    results = {0: RowResult(["Usability"]), 1: RowResult(["Usability"]), 2: RowResult(["Performance"]), 3: RowResult([])}
    table, overall = metrics.agreement(existing, results, ls_names)
    assert overall["compared"] == 3 and overall["exact_match"] == pytest.approx(2 / 3)
    assert metrics.disagreements(existing, results) == [1]


def test_cohen_kappa_known_value():
    assert metrics.cohen_kappa(20, 5, 10, 15) == pytest.approx(0.4)
    assert metrics.cohen_kappa(10, 0, 0, 10) == 1.0


def test_candidate_label_columns(sample_df):
    df = sample_df.assign(usability=["Yes", "No"] * 15)
    cands = candidate_label_columns(df, "content")
    assert "usability" in cands and "content" not in cands and "review_id" not in cands


# ── export ──────────────────────────────────────────────────────────────────

def test_exports(sample_df, general_labels):
    results, _ = label_dataframe(sample_df, "content", general_labels, FakeLLM())
    out = results_to_frame(sample_df, results, general_labels)
    out.loc[0, "evidence"] = "bad\x07char"
    csv_bytes = to_csv_bytes(out)
    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    xls = pd.read_excel(__import__("io").BytesIO(to_excel_bytes({"Labeled": out, "Info": pd.DataFrame({"a": [1]})})), sheet_name=None)
    assert set(xls) == {"Labeled", "Info"} and len(xls["Labeled"]) == 30


def test_label_column_names():
    assert label_column("Pricing & Ads") == "has_pricing_ads"
    assert label_column("Feature Request") == "has_feature_request"


# ── LLM client (mocked OpenAI SDK) ──────────────────────────────────────────

class _Completions:
    def __init__(self, behaviour):
        self.behaviour, self.kwargs = behaviour, []

    def create(self, **kw):
        self.kwargs.append(kw)
        return self.behaviour(kw)


def _sdk(behaviour):
    comp = _Completions(behaviour)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp),
                           models=SimpleNamespace(list=lambda: [SimpleNamespace(id="b"), SimpleNamespace(id="a")])), comp


def _resp(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def test_client_json_mode_and_usage():
    sdk, comp = _sdk(lambda kw: _resp('{"results": []}'))
    c = LLMClient("http://x", "k", "m", sdk_client=sdk)
    assert c.chat_json("s", "u") == {"results": []}
    assert comp.kwargs[0]["response_format"] == {"type": "json_object"} and comp.kwargs[0]["temperature"] == 0.0
    assert c.usage.total_tokens == 15 and c.list_models() == ["a", "b"]


def test_client_falls_back_when_json_mode_unsupported():
    def behaviour(kw):
        if "response_format" in kw:
            raise openai.BadRequestError("response_format not supported", response=SimpleNamespace(
                request=None, status_code=400, headers={}), body=None)
        return _resp('{"ok": 1}')
    sdk, comp = _sdk(behaviour)
    c = LLMClient("http://x", "k", "m", sdk_client=sdk)
    assert c.chat_json("s", "u") == {"ok": 1} and c.chat_json("s", "u") == {"ok": 1}
    assert len(comp.kwargs) == 3          # 1 failed JSON-mode call, then plain calls only


def test_client_throttles():
    slept = []
    sdk, _ = _sdk(lambda kw: _resp("{}"))
    c = LLMClient("http://x", "k", "m", requests_per_minute=30, sdk_client=sdk, sleep_fn=slept.append)
    c.chat_json("s", "u"); c.chat_json("s", "u")
    assert slept and 1.5 < slept[0] <= 2.0


def test_providers_are_complete():
    for p in PROVIDERS.values():
        assert p.label and p.secret_name and p.batch_size > 0
        if p.key != "custom":
            assert p.base_url.startswith("http")


def test_cli_end_to_end(tmp_path, monkeypatch, sample_df):
    import label_cli
    from tests.fakes import FakeLLM

    class Patched:
        def __init__(self, *a, **kw):
            self._f = FakeLLM(fail_after=2)
            self.usage, self.model = self._f.usage, "fake"
        def chat_json(self, s, u):
            return self._f.chat_json(s, u)

    monkeypatch.setattr(label_cli, "LLMClient", Patched)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    csv_in = tmp_path / "r.csv"
    csv_in.write_bytes((root / "sample_data" / "reviews_sample.csv").read_bytes())
    args = [str(csv_in), str(root / "sample_data" / "labels_general.json"), "--batch-size", "5"]
    assert label_cli.main(args) == 1                       # paused by fake quota after 10 rows
    assert (tmp_path / "r_labeled.checkpoint.json").exists()
    assert label_cli.main(args) == 1                       # resumes: 10 more
    monkeypatch.setattr(label_cli, "LLMClient", lambda *a, **kw: type("C", (), {
        "usage": FakeLLM().usage, "model": "fake", "chat_json": lambda self, s, u: FakeLLM().chat_json(s, u)})())
    assert label_cli.main(args) == 0
    out = pd.read_csv(tmp_path / "r_labeled.csv", encoding="utf-8-sig")
    assert len(out) == 30 and (out["status"] == "labeled").all()
