"""Pass 5: contraction insertion.

LLMs (especially in academic mode) over-produce uncontracted forms
("it is", "do not"). Strategic contraction insertion is a low-cost
human-coded signal. The density is configurable; over-contraction is
itself a tell so we cap based on register.

Reference: research/02 D.1 (uniform formality is a signal).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .base import PipelinePass


CONTRACTIONS: List[Tuple[str, str]] = [
    (r"\bit is\b", "it's"),
    (r"\bIt is\b", "It's"),
    (r"\bthat is\b", "that's"),
    (r"\bThat is\b", "That's"),
    (r"\bthere is\b", "there's"),
    (r"\bThere is\b", "There's"),
    (r"\bwhat is\b", "what's"),
    (r"\bWhat is\b", "What's"),
    (r"\bwho is\b", "who's"),
    (r"\bWho is\b", "Who's"),
    (r"\bdo not\b", "don't"),
    (r"\bDo not\b", "Don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bDoes not\b", "Doesn't"),
    (r"\bdid not\b", "didn't"),
    (r"\bDid not\b", "Didn't"),
    (r"\bcannot\b", "can't"),
    (r"\bCannot\b", "Can't"),
    (r"\bwill not\b", "won't"),
    (r"\bWill not\b", "Won't"),
    (r"\bwould not\b", "wouldn't"),
    (r"\bWould not\b", "Wouldn't"),
    (r"\bcould not\b", "couldn't"),
    (r"\bCould not\b", "Couldn't"),
    (r"\bshould not\b", "shouldn't"),
    (r"\bShould not\b", "Shouldn't"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"),
    (r"\bhas not\b", "hasn't"),
    (r"\bhave not\b", "haven't"),
    (r"\bhad not\b", "hadn't"),
    (r"\byou are\b", "you're"),
    (r"\bYou are\b", "You're"),
    (r"\bthey are\b", "they're"),
    (r"\bThey are\b", "They're"),
    (r"\bwe are\b", "we're"),
    (r"\bWe are\b", "We're"),
    (r"\bI am\b", "I'm"),
    (r"\bI have\b", "I've"),
    (r"\bI will\b", "I'll"),
    (r"\bI would\b", "I'd"),
    (r"\byou will\b", "you'll"),
    (r"\bthey will\b", "they'll"),
    (r"\bwe will\b", "we'll"),
]


class ContractionsPass(PipelinePass):
    pass_id = 5
    pass_name = "contractions"

    def apply(self, text: str, config: Dict[str, Any]) -> str:
        self.reset_changes()
        intensity = config.get("intensity", "balanced")
        target_register = config.get("target_register", "neutral")

        # density: fraction of candidates we actually contract
        density = config.get(
            "density",
            {
                "minimal": 0.3,
                "balanced": 0.7,
                "aggressive": 0.95,
            }.get(intensity, 0.7),
        )

        # academic / formal registers contract more sparingly
        if target_register in {"academic", "formal"}:
            density *= 0.2

        out = text
        seed = config.get("seed", 42)

        for i, (pattern, replacement) in enumerate(CONTRACTIONS):
            out = self._apply_pattern(out, pattern, replacement, density, seed + i)
        return out

    def _apply_pattern(
        self,
        text: str,
        pattern: str,
        replacement: str,
        density: float,
        seed: int,
    ) -> str:
        compiled = re.compile(pattern)
        matches = list(compiled.finditer(text))
        if not matches:
            return text

        if density >= 1.0:
            keep_indices = set(range(len(matches)))
        elif density <= 0:
            return text
        else:
            # v0.1.1: replaced the broken stride math (round(1/0.7)=1 -> stride=1
            # -> 100% replacement at "balanced") with a per-match probabilistic
            # keep using a deterministic hash seed. This produces the right
            # average density at any value and avoids over-uniform contraction
            # which is itself a stylometric tell.
            import hashlib

            keep_indices = set()
            for i, m in enumerate(matches):
                digest = hashlib.sha1(
                    f"{seed}:{pattern}:{m.start()}:{i}".encode("utf-8")
                ).digest()
                roll = digest[0] / 256.0
                if roll < density:
                    keep_indices.add(i)

        out_parts: List[str] = []
        cursor = 0
        for i, m in enumerate(matches):
            out_parts.append(text[cursor : m.start()])
            if i in keep_indices:
                out_parts.append(replacement)
                self.log_change("contraction", m.group(0), replacement, m.start())
            else:
                out_parts.append(m.group(0))
            cursor = m.end()
        out_parts.append(text[cursor:])
        return "".join(out_parts)
