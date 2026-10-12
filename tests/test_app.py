"""Click through the Streamlit app headlessly with the fake model."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import labeler.llm
from tests.fakes import FakeLLM

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def fake_model(monkeypatch):
    class Patched(labeler.llm.LLMClient):
        def __init__(self, *a, **kw):
            self._fake = FakeLLM()
            self.usage = self._fake.usage

        def chat_json(self, system, user):
            return self._fake.chat_json(system, user)

        def list_models(self):
            return ["free-model-a", "free-model-b"]

    monkeypatch.setattr(labeler.llm, "LLMClient", Patched)


def btn(at, label):
    return next(b for b in at.button if b.label.startswith(label))


def go(at):
    return at.button(key="go")


def html_text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.sidebar.selectbox[0].value == "groq"
    assert "How it works" in html_text(at)


def test_start_disabled_until_ready(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert go(at).disabled
    page = html_text(at)
    assert "check no" in page and "Reviews uploaded" in page and "API key entered" in page


def test_full_flow_sample_data(fake_model):
    at = AppTest.from_file(APP, default_timeout=60).run()
    btn(at, "Use sample data").click().run()
    btn(at, "General example").click().run()
    at.sidebar.text_input(key="key_groq").input("gsk_test").run()
    assert not at.exception
    assert at.selectbox(key="textcol_reviews_sample.csv").value == "content"
    assert not go(at).disabled

    go(at).click().run()                                          # test on first 10
    assert len(at.session_state["run"]["results"]) == 10 and not at.exception
    assert "Test finished" in html_text(at)

    at.segmented_control(key="scope").set_value("all").run()      # scope → all reviews
    assert go(at).label.startswith("Continue labeling · 20 left")
    go(at).click().run()
    assert len(at.session_state["run"]["results"]) == 30
    assert any("Label presence" in t.label for t in at.tabs)
    assert "Present" in html_text(at)
    assert not at.exception


def test_single_mode_and_paste_labels(fake_model):
    at = AppTest.from_file(APP, default_timeout=60).run()
    btn(at, "Use sample data").click().run()
    at.text_area(key="paste_text").input("Performance: crashes and lag\nPricing & Ads: ads, price\nUsability: easy, confusing").run()
    btn(at, "Use these labels").click().run()
    at.segmented_control(key="ls_mode").set_value("single").run()
    at.sidebar.text_input(key="key_groq").input("gsk_test").run()
    go(at).click().run()
    run = at.session_state["run"]
    assert run["labelset"].mode == "single" and run["labelset"].names() == ["Performance", "Pricing & Ads", "Usability"]
    assert all(len(r.labels) == 1 for r in run["results"].values())
    assert not at.exception


def test_fetch_models_button(fake_model):
    at = AppTest.from_file(APP, default_timeout=30).run()
    btn(at, "Models").click().run()
    assert "free-model-a" in at.sidebar.selectbox[1].options
