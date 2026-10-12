"""A fake LLM that reads the real prompts and answers with keyword rules (offline tests + demo)."""

from __future__ import annotations

import json
import re

from labeler.llm import QuotaError, Usage

KEYWORDS = {
    "Usability": ["easy", "intuitive", "find", "menu", "navigation", "confusing", "button", "tiny", "contrast", "design", "lost", "clean", "taps", "shortcut"],
    "Performance": ["crash", "freez", "load", "battery", "gps", "slow", "fast", "reliable", "logged me out"],
    "Feature Request": ["wish", "would love", "offline", "should be", "add "],
    "Privacy & Security": ["privacy", "data", "password", "logged me out"],
    "Pricing & Ads": ["ads", "price", "premium", "paywall", "subscription", "charged", "refund", "upgrade", "pay"],
    "Customer Support": ["support", "customer service"],
    "Learnability": ["figured", "intuitive", "without any help", "learn", "first"],
    "Efficiency": ["taps", "quick", "shortcut", "one step", "fast"],
    "Memorability": ["redesign", "moved", "forget", "relearn", "can't find", "used to it"],
    "Errors": ["wrong", "error", "undo", "mistake"],
    "Satisfaction": ["beautiful", "clean", "calm", "fun", "simple"],
}


class FakeLLM:
    def __init__(self, fail_after: int | None = None, drop_ids: set[int] | None = None, bad_json_first: bool = False):
        self.usage = Usage()
        self.calls: list[tuple[str, str]] = []
        self.fail_after = fail_after
        self.drop_ids = drop_ids or set()
        self.bad_json_first = bad_json_first

    def chat_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        if self.fail_after is not None and self.usage.requests >= self.fail_after:
            raise QuotaError("daily limit reached (fake)")
        self.usage.requests += 1
        self.usage.prompt_tokens += len((system + user).split())

        names = re.findall(r"^\d+\. (.+?)(?: — |$)", system, flags=re.M)
        single = "exactly ONE label" in system
        items = re.findall(r'\[id=(\d+)\][^\n]*\n"""(.*?)"""', user, flags=re.S)

        if self.bad_json_first and len(items) > 1:
            self.bad_json_first = False
            from labeler.llm import ModelError
            raise ModelError("The model did not return JSON")

        results = []
        for sid, text in items:
            rid = int(sid)
            if rid in self.drop_ids and len(items) > 1:
                continue
            low = text.lower()
            found = [n for n in names if any(k in low for k in KEYWORDS.get(n, [n.lower()]))]
            conf = "HIGH" if len(found) == 1 else ("MEDIUM" if found else "LOW")
            if single:
                results.append({"id": rid, "label": found[0] if found else "NONE", "confidence": conf, "evidence": text[:40]})
            else:
                results.append({"id": rid, "labels": found, "confidence": conf, "evidence": text[:40]})
        return json.loads(json.dumps({"results": results}))
