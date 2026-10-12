# 🏷️ Review Labeler: label reviews with your own labels using a free LLM

Upload a CSV of plain-text reviews (no label column needed), type your labels and their definitions, and a **free** AI model checks **which labels are present in every review** and labels the whole dataset. You get per-label counts and a labeled CSV/Excel file for your research.

No Anthropic or other paid key is needed. Pick one free provider:

| Provider | Cost | Get a key | Notes |
|---|---|---|---|
| **Groq** (recommended) | Free plan, no card | https://console.groq.com/keys | Fast. Roughly 30 requests/min and 1,000 requests/day per model |
| **Google Gemini** | Free tier | https://aistudio.google.com/apikey | Small daily limits; Google may use free-tier data to improve its products |
| **OpenRouter** | Free `:free` models | https://openrouter.ai/keys | Small daily request cap on free accounts |
| **Ollama** | 100% free, no key | https://ollama.com/download | Runs on your own computer, so local use only |
| Any OpenAI-compatible API | varies | | e.g. Microsoft Foundry, Together, Mistral |

Free-tier limits and model names change often. Use **Fetch models** in the sidebar to get the live list.

## The interface
- **Session panel** and **progress stepper** show at a glance what's ready (file, labels, key, model) and which step you're on.
- One card per step with icons, smooth entrance animations and justified text; the Start button stays disabled until the readiness checklist is complete.
- While labeling, a live feed shows the latest reviews and the labels they received.
- Works on desktop and mobile; animations are switched off automatically for users who prefer reduced motion.

## How it works
1. **Upload** a CSV/Excel file and pick the column holding the review text. Empty rows and duplicates are reported.
2. **Define labels**: one row per label with a clear definition (and an optional example phrase). You can also paste them, load a saved `.json` label set, or start from an example.
3. **Choose a mode:**
   - *Check which labels are present*: 0, 1 or several labels per review (one `has_<label>` 0/1 column per label).
   - *Pick the single best label*: exactly one label or `NONE`.
4. Optionally add **your own prompt / instructions**. You can view the exact prompt sent to the model.
5. **Test on 10 reviews**, check the results, then label **all reviews**. Reviews are sent in batches (15–25 per request) to stay within free daily limits.
6. **Results**: each label shows *present / not found* with counts and percentages, plus confidence, a short evidence quote, and a browsable table.
7. **Optional**: if your file already has human labels (e.g. a Yes/No `usability` column), compare them with the model and get precision, recall, F1 and Cohen's κ, plus a list of disagreements.
8. **Download** a CSV, an Excel file (labeled reviews + presence summary + definitions + run info for your methodology), or the label set as JSON.

If you hit a free-tier limit, **nothing is lost**. Wait and press **Continue**, or switch to another free model in the sidebar and continue.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your free key into it
streamlit run app.py
```
Click **Use sample data** and **Example: general** to try it immediately.

## Deploy on Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo. `.streamlit/secrets.toml` is git-ignored, so **don't commit keys**.
2. Go to https://share.streamlit.io, choose **Create app**, select the repo, and set the main file to `app.py`.
3. Open **Advanced settings → Secrets** (or later **App settings → Secrets**) and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_free_key"
   ```
4. Deploy. The sidebar will show "✓ Key loaded from secrets".

> Ollama does not work on Streamlit Cloud because it runs on your own computer. Use Groq, Gemini or OpenRouter there.

## Command line (long runs)
```bash
export GROQ_API_KEY="gsk_..."
python label_cli.py data/reviews.csv sample_data/labels_general.json --provider groq --limit 10   # test
python label_cli.py data/reviews.csv sample_data/labels_general.json --provider groq             # full run
```
If it stops at a limit, run the same command again later and it resumes from the checkpoint.

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```
The tests use a fake model, so they run offline. They cover CSV/Excel loading, prompt building, batching, retries, resuming after a quota error, single/multi-label output, agreement metrics, exports, the CLI, and a click-through of the web app.

## Project layout
| Path | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `label_cli.py` | Command-line version with checkpoint/resume |
| `labeler/providers.py` | Free provider presets (URL, key name, limits) |
| `labeler/llm.py` | OpenAI-compatible client: JSON mode, retries, throttling |
| `labeler/prompts.py` | Builds prompts from your labels; validates model output |
| `labeler/engine.py` | Batch labeling, resume, result table |
| `labeler/metrics.py` | Label presence, precision/recall/F1/κ |
| `sample_data/` | 30 sample reviews + two example label sets |
| `tests/` | Pytest suite with a fake model |

## Tips for good labels
- Write definitions as **inclusion rules** ("mark when the user talks about…"), and add exclusions in the prompt ("ignore pure bug reports").
- Keep temperature at 0, and always test on 10 reviews first.
- LLM labels are a strong first pass, not ground truth. Hand-label a random sample (e.g. 100 reviews) and report agreement (κ) in your paper.
