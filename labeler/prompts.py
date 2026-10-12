"""Turn the user's labels + definitions into prompts, and validate what comes back."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MULTI = "multi"     # which labels are PRESENT (0, 1 or many per review)
SINGLE = "single"   # exactly one best label per review
NONE_LABEL = "NONE"
CONFIDENCE = ("HIGH", "MEDIUM", "LOW")
MAX_REVIEW_CHARS = 2500


@dataclass
class LabelSpec:
    name: str
    definition: str = ""
    examples: str = ""


@dataclass
class LabelSet:
    labels: list[LabelSpec]
    mode: str = MULTI
    instructions: str = ""
    include_evidence: bool = True
    context_columns: list[str] = field(default_factory=list)   # e.g. ["app_name", "score"]

    def names(self) -> list[str]:
        return [l.name for l in self.labels]

    def validate(self) -> str:
        names = [l.name.strip() for l in self.labels if l.name and l.name.strip()]
        if not names:
            return "Add at least one label."
        if self.mode == MULTI and len(names) < 1:
            return "Add at least one label."
        if self.mode == SINGLE and len(names) < 2:
            return "Single-label mode needs at least two labels."
        lowered = [_norm(n) for n in names]
        if len(set(lowered)) != len(lowered):
            return "Label names must be unique."
        if _norm(NONE_LABEL) in lowered:
            return "'NONE' is reserved — the model uses it when no label fits."
        return ""

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "instructions": self.instructions,
            "include_evidence": self.include_evidence,
            "context_columns": self.context_columns,
            "labels": [{"name": l.name, "definition": l.definition, "examples": l.examples} for l in self.labels],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelSet":
        return cls(
            labels=[LabelSpec(x.get("name", ""), x.get("definition", ""), x.get("examples", "")) for x in d.get("labels", [])],
            mode=d.get("mode", MULTI),
            instructions=d.get("instructions", ""),
            include_evidence=d.get("include_evidence", True),
            context_columns=d.get("context_columns", []),
        )


# ── prompt building ──────────────────────────────────────────────────────────

def build_system_prompt(ls: LabelSet) -> str:
    lines = [
        "You are a careful, consistent annotator labelling user reviews for an academic research dataset.",
        "Base every decision only on what the review text actually says. Do not invent information.",
        "",
    ]
    if ls.instructions.strip():
        lines += ["## RESEARCHER INSTRUCTIONS", ls.instructions.strip(), ""]

    lines.append("## LABELS AND DEFINITIONS")
    for i, l in enumerate(ls.labels, 1):
        entry = f"{i}. {l.name.strip()}"
        if l.definition.strip():
            entry += f" — {l.definition.strip()}"
        lines.append(entry)
        if l.examples.strip():
            lines.append(f"   Example: {l.examples.strip()}")
    lines.append("")

    lines.append("## TASK")
    if ls.mode == MULTI:
        lines += [
            "For EACH review, decide which of the labels above are PRESENT in it.",
            "- A label is present only when the review clearly matches its definition.",
            "- A review may have zero, one, or several labels.",
            "- If no label applies, return an empty list [].",
        ]
    else:
        lines += [
            "For EACH review, choose exactly ONE label that best matches its main point.",
            f'- If none of the labels fits, use "{NONE_LABEL}".',
        ]
    lines += [
        "- Use the label names EXACTLY as written above. Never create new labels.",
        "- Judge each review independently.",
        "- confidence: HIGH = clear-cut, MEDIUM = plausible but could be read differently, LOW = vague, very short or ambiguous.",
        "",
        "## OUTPUT",
        "Return ONLY a JSON object, no other text, in this shape:",
    ]
    ev = ', "evidence": "<short quote or reason, max 15 words>"' if ls.include_evidence else ""
    if ls.mode == MULTI:
        lines.append('{"results": [{"id": <review id>, "labels": ["<label>", ...], "confidence": "HIGH|MEDIUM|LOW"' + ev + "}]}")
    else:
        lines.append('{"results": [{"id": <review id>, "label": "<label or NONE>", "confidence": "HIGH|MEDIUM|LOW"' + ev + "}]}")
    lines.append("Include exactly one result for every review id you receive.")
    return "\n".join(lines)


def build_user_prompt(batch: list[tuple[int, str, dict]]) -> str:
    """batch = [(local_id, text, context_dict), ...]"""
    parts = [f"Label these {len(batch)} review(s):", ""]
    for local_id, text, ctx in batch:
        text = (text or "").strip().replace('"""', "'''")
        if len(text) > MAX_REVIEW_CHARS:
            text = text[:MAX_REVIEW_CHARS] + " …[truncated]"
        meta = " | ".join(f"{k}: {v}" for k, v in ctx.items() if str(v).strip())
        parts.append(f"[id={local_id}]" + (f" ({meta})" if meta else ""))
        parts.append(f'"""{text}"""')
        parts.append("")
    return "\n".join(parts)


# ── response parsing ─────────────────────────────────────────────────────────

@dataclass
class ParsedRow:
    labels: list[str]
    confidence: str = ""
    evidence: str = ""
    warning: str = ""


def parse_results(data: dict, expected_ids: list[int], ls: LabelSet) -> dict[int, ParsedRow]:
    """Map local id → ParsedRow. Ids missing from the response are simply absent."""
    canonical = {_norm(n): n for n in ls.names()}
    items = data.get("results")
    if items is None:
        # Some models return {"1": {...}, "2": {...}}
        items = [dict(v, id=k) for k, v in data.items() if isinstance(v, dict)]
    out: dict[int, ParsedRow] = {}
    expected = set(expected_ids)

    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            rid = int(str(item.get("id", "")).strip().strip("[]").replace("id=", ""))
        except ValueError:
            continue
        if rid not in expected:
            continue

        raw = item.get("labels") if ls.mode == MULTI else item.get("label")
        if raw is None:
            raw = item.get("label") if ls.mode == MULTI else item.get("labels")
        values = raw if isinstance(raw, list) else ([] if raw in (None, "") else re.split(r"[;,|]", str(raw)))

        labels, unknown = [], []
        for v in values:
            key = _norm(str(v))
            if not key or key in (_norm(NONE_LABEL), "none", "null", "na"):
                continue
            if key in canonical:
                if canonical[key] not in labels:
                    labels.append(canonical[key])
            else:
                unknown.append(str(v))

        warning = f"ignored unknown label(s): {', '.join(unknown)}" if unknown else ""
        if ls.mode == SINGLE:
            if len(labels) > 1:
                warning = (warning + "; " if warning else "") + f"model gave {len(labels)} labels, kept the first"
                labels = labels[:1]
            if not labels:
                labels = [NONE_LABEL]

        conf = str(item.get("confidence", "")).strip().upper()
        if conf not in CONFIDENCE:
            conf = ""
        out[rid] = ParsedRow(labels=labels, confidence=conf, evidence=str(item.get("evidence", "") or "").strip(), warning=warning)
    return out


def parse_label_text(text: str) -> list[LabelSpec]:
    """Parse pasted lines like 'Usability: the review talks about ease of use'."""
    specs = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-•*").strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(?:\d+[.)]\s*)?(.+?)\s*(?::|—|–|\s-\s|\t|=)\s*(.*)$", line)
        name, definition = (m.group(1), m.group(2)) if m else (line, "")
        specs.append(LabelSpec(name.strip().strip('"'), definition.strip()))
    return specs


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", str(s).casefold())
