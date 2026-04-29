"""Pass 7: punctuation variation.

Introduces controlled punctuation variety so the output does not look like
a uniform stream of commas and periods (a recognizable LLM signature, see
research/02 H, punctuation-mark distribution is a stylometric feature).

We do NOT reintroduce em dashes (pass 1 already removed them); we use
en dashes, parenthetical asides, occasional ellipses, and we add or remove
the Oxford comma based on the configured style.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from .base import PipelinePass

EN_DASH = "\u2013"

OXFORD_RE = re.compile(
    r"(\w+(?:\s+\w+){0,3}),\s+(\w+(?:\s+\w+){0,3}),\s+and\s+(\w+(?:\s+\w+){0,3})\b",
    re.IGNORECASE,
)
NO_OXFORD_RE = re.compile(
    r"(\w+(?:\s+\w+){0,3}),\s+(\w+(?:\s+\w+){0,3})\s+and\s+(\w+(?:\s+\w+){0,3})\b",
    re.IGNORECASE,
)


class PunctuationPass(PipelinePass):
    pass_id = 7
    pass_name = "punctuation"

    def apply(self, text: str, config: Dict[str, Any]) -> str:
        # v0.1.1 paragraph preservation: process each paragraph separately
        # so we do not collapse double-newline separators (catastrophic
        # for blogs/books). Wraps the original logic from _apply_one_paragraph.
        if "\n" in text:
            parts = re.split(r"(\n\s*\n)", text)
            out_parts = []
            for part in parts:
                if part.strip() == "" or "\n" in part:
                    out_parts.append(part)
                else:
                    out_parts.append(self._apply_one_paragraph(part, config))
            return "".join(out_parts)
        return self._apply_one_paragraph(text, config)

    def _apply_one_paragraph(self, text: str, config: Dict[str, Any]) -> str:
        self.reset_changes()
        style = config.get("style", {})
        intensity = config.get("intensity", "balanced")
        seed = config.get("seed", 42)

        out = text
        oxford_pref: Optional[bool] = style.get("preserve_oxford_comma")
        if oxford_pref is None:
            out = self._inconsistent_oxford(out, seed)
        elif oxford_pref is False:
            out = self._strip_oxford(out)
        # if True, leave Oxford commas alone

        if intensity in {"balanced", "aggressive"}:
            out = self._sprinkle_ellipses(out, seed, max_inserts=1 if intensity == "balanced" else 2)

        if intensity == "aggressive":
            out = self._sprinkle_en_dashes(out, seed)

        return out

    def _strip_oxford(self, text: str) -> str:
        def _sub(m: re.Match[str]) -> str:
            a, b, c = m.group(1), m.group(2), m.group(3)
            replacement = f"{a}, {b} and {c}"
            self.log_change("oxford_strip", m.group(0), replacement, m.start())
            return replacement

        return OXFORD_RE.sub(_sub, text)

    def _inconsistent_oxford(self, text: str, seed: int) -> str:
        # remove some, leave some, a true human-coded signal
        toggle = {"i": 0}

        def _sub(m: re.Match[str]) -> str:
            toggle["i"] += 1
            if toggle["i"] % 2 == 0:
                return m.group(0)
            a, b, c = m.group(1), m.group(2), m.group(3)
            replacement = f"{a}, {b} and {c}"
            self.log_change("oxford_inconsistent", m.group(0), replacement, m.start())
            return replacement

        return OXFORD_RE.sub(_sub, text)

    def _sprinkle_ellipses(self, text: str, seed: int, max_inserts: int) -> str:
        # convert one or two trailing periods that follow short clauses into
        # an ellipsis to suggest a pause
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        if len(sentences) < 4:
            return text
        digest = hashlib.sha1(f"{seed}:ellipses".encode("utf-8")).digest()
        candidates = [
            i for i, s in enumerate(sentences) if 4 <= len(s.split()) <= 12 and s.endswith(".")
        ]
        if not candidates:
            return text
        chosen = []
        for k in range(min(max_inserts, len(candidates))):
            idx = candidates[(digest[k] if k < len(digest) else 0) % len(candidates)]
            if idx in chosen:
                continue
            chosen.append(idx)
            old = sentences[idx]
            sentences[idx] = old[:-1] + "..."
            self.log_change("ellipsis", old[:60], sentences[idx][:60])

        return " ".join(sentences)

    def _sprinkle_en_dashes(self, text: str, seed: int) -> str:
        # convert " - " (with spaces) to en dashes for a small number of cases
        if " - " not in text:
            return text
        out = text.replace(" - ", f" {EN_DASH} ", 1)
        if out != text:
            self.log_change("en_dash", " - ", f" {EN_DASH} ")
        return out
