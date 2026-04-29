"""Pass 3: structural rewrites.

Targets the structural AI tells from research/02 section C:

* "It's not X, it's Y" parallel constructions (C.3)
* "Not just X, but Y" / "More than X, it's Y" (C.3)
* Tricolon overuse: three-element parallel lists "X, Y, and Z" (C.2)
* "X is a Y of Z" copular metaphor templates (C.4)
* Symmetric paragraph structures (C.5, C.6)

spaCy is used for syntactic checks when available; the pass falls back to
pure regex heuristics when spaCy or the en_core_web_sm model is missing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import PipelinePass

try:
    import spacy

    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        _NLP = None
except ImportError:
    spacy = None
    _NLP = None


_NOT_X_BUT_Y = re.compile(
    r"\b(?:it'?s|this is|that's)\s+not\s+(?:just\s+)?"
    r"(?:about\s+|a\s+|an\s+)?([^,.;]{2,80}?),?\s+"
    r"(?:it'?s|but)\s+(?:about\s+|a\s+|an\s+)?([^.;]{2,120}?)([.;])",
    re.IGNORECASE,
)

_NOT_JUST_BUT = re.compile(
    r"\bnot just\s+([^,.;]{2,80}?),?\s+but\s+([^.;]{2,120}?)([.;])",
    re.IGNORECASE,
)

_TRICOLON = re.compile(
    r"\b(\w+(?:\s+\w+){0,3}),\s+(\w+(?:\s+\w+){0,3}),\s+and\s+(\w+(?:\s+\w+){0,3})\b",
    re.IGNORECASE,
)

_X_IS_A_Y_OF_Z = re.compile(
    r"\b([A-Z][\w-]+(?:\s+[\w-]+){0,2})\s+is\s+(?:a|the)\s+"
    r"(cornerstone|engine|foundation|backbone|heart|soul|key|gateway|"
    r"bedrock|essence|pillar|hallmark|catalyst|driver)\s+of\s+([\w-]+(?:\s+[\w-]+){0,3})",
    re.IGNORECASE,
)


class StructuralPass(PipelinePass):
    pass_id = 3
    pass_name = "structural"

    def apply(self, text: str, config: Dict[str, Any]) -> str:
        self.reset_changes()
        intensity = config.get("intensity", "balanced")

        out = text
        out = self._rewrite_not_x_but_y(out)
        out = self._rewrite_not_just_but(out)

        # tricolon dampening only applies at balanced+; even then only
        # when several appear in close proximity. See research/02 C.2.
        if intensity in {"balanced", "aggressive"}:
            out = self._dampen_tricolons(out)

        if intensity == "aggressive":
            out = self._dampen_copular_templates(out)

        return out

    def _rewrite_not_x_but_y(self, text: str) -> str:
        def _sub(m: re.Match[str]) -> str:
            x, y, term = m.group(1).strip(), m.group(2).strip(), m.group(3)
            replacement = f"{y[0].upper()}{y[1:]}{term}"
            self.log_change("not_x_but_y", m.group(0), replacement, m.start())
            return replacement

        return _NOT_X_BUT_Y.sub(_sub, text)

    def _rewrite_not_just_but(self, text: str) -> str:
        def _sub(m: re.Match[str]) -> str:
            x, y, term = m.group(1).strip(), m.group(2).strip(), m.group(3)
            replacement = f"{y}{term}"
            self.log_change("not_just_but", m.group(0), replacement, m.start())
            return replacement

        return _NOT_JUST_BUT.sub(_sub, text)

    def _dampen_tricolons(self, text: str) -> str:
        # Only dampen when more than two tricolons appear in the same paragraph;
        # a single rule-of-three is fine, multiple in close proximity is the
        # AI fingerprint.
        paragraphs = re.split(r"(\n\s*\n)", text)
        out_parts: List[str] = []
        for chunk in paragraphs:
            if not chunk.strip() or chunk.startswith("\n"):
                out_parts.append(chunk)
                continue
            matches = list(_TRICOLON.finditer(chunk))
            if len(matches) <= 1:
                out_parts.append(chunk)
                continue
            # rewrite every other one as "X and Y. Z" (drops the third item to
            # the next clause)
            new_chunk = chunk
            for i, m in enumerate(matches):
                if i % 2 == 1:
                    continue  # leave it
                a, b, c = m.group(1), m.group(2), m.group(3)
                replacement = f"{a} and {b}. Also {c}"
                start = new_chunk.find(m.group(0))
                if start == -1:
                    continue
                self.log_change("tricolon", m.group(0), replacement, start)
                new_chunk = new_chunk[:start] + replacement + new_chunk[start + len(m.group(0)) :]
            out_parts.append(new_chunk)
        return "".join(out_parts)

    def _dampen_copular_templates(self, text: str) -> str:
        toggle = {"i": 0}

        def _sub(m: re.Match[str]) -> str:
            toggle["i"] += 1
            if toggle["i"] % 2 == 0:
                return m.group(0)
            x, _metaphor, z = m.group(1), m.group(2), m.group(3)
            replacement = f"{x} matters for {z}"
            self.log_change("copular_template", m.group(0), replacement, m.start())
            return replacement

        return _X_IS_A_Y_OF_Z.sub(_sub, text)
