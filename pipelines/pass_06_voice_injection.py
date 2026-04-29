"""Pass 6: voice / filler injection.

Inject the conversational filler that LLMs strip out: hedges, asides,
discourse particles, parenthetical observations. Style-aware: the
``allowed_filler`` list comes from the active style preset; ``avoid_filler``
acts as a blocklist.

References:
* research/02 D.1 (uniform formality is a signal)
* research/02 D.2 (absence of slang/idiom is a signal)
* research/03 section 3 (don't add casual fillers to academic / autistic styles)
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Tuple

from .base import PipelinePass

DEFAULT_FILLERS_BY_STYLE = {
    "casual": [
        "honestly", "tbh", "lowkey", "kinda", "fwiw",
        "I mean", "like", "you know", "look",
    ],
    "neutral": ["honestly", "to be fair", "for what it's worth", "I think"],
    "academic": ["arguably", "to a first approximation", "broadly speaking"],
    "formal": [],
}

OPENERS = {
    "casual": ["honestly,", "ok so", "look,", "I mean,", "alright,"],
    "neutral": ["honestly,", "to be fair,", "I think"],
    "academic": ["arguably,", "broadly speaking,"],
    "formal": [],
}


class VoiceInjectionPass(PipelinePass):
    pass_id = 6
    pass_name = "voice_injection"

    def apply(self, text: str, config: Dict[str, Any]) -> str:
        self.reset_changes()
        style = config.get("style", {})
        target_register = config.get("target_register", "neutral")
        intensity = config.get("intensity", "balanced")

        allowed_raw = style.get("allowed_filler")
        if allowed_raw is None:
            allowed_raw = DEFAULT_FILLERS_BY_STYLE.get(target_register, [])
        allowed: List[str] = list(allowed_raw)
        avoid: set[str] = set(style.get("avoid_filler") or [])
        allowed = [f for f in allowed if f.lower() not in {a.lower() for a in avoid}]

        opener_raw = style.get("allowed_openers")
        if opener_raw is None:
            opener_raw = OPENERS.get(target_register, [])
        opener_pool = [o for o in opener_raw if o.lower() not in {a.lower() for a in avoid}]

        if not allowed and not opener_pool:
            return text

        max_inserts = {
            "minimal": 1,
            "balanced": 3,
            "aggressive": 6,
        }.get(intensity, 3)

        seed = config.get("seed", 42)

        return self._inject(text, allowed, opener_pool, max_inserts, seed)

    def _inject(
        self,
        text: str,
        fillers: List[str],
        openers: List[str],
        max_inserts: int,
        seed: int,
    ) -> str:
        # split into sentences via simple regex (avoid heavy NLTK dependency
        # here; the rhythm pass already handled segmentation when needed)
        sentence_re = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
        sentences = sentence_re.split(text)
        if len(sentences) < 2:
            return text

        # decide which sentences receive an insertion. Choose deterministic
        # indexes spread across the text.
        n = len(sentences)
        target_count = min(max_inserts, max(1, n // 5))
        indices: List[int] = []
        if target_count > 0:
            stride = max(1, n // target_count)
            for k in range(target_count):
                idx = (k * stride + 1) % n  # avoid first sentence to keep opening clean
                indices.append(idx)
        indices_set = set(indices)

        out: List[str] = []
        for i, sent in enumerate(sentences):
            if i not in indices_set or not sent.strip():
                out.append(sent)
                continue
            digest = hashlib.sha1(f"{seed}:{i}:{sent[:30]}".encode("utf-8")).digest()
            use_opener = bool(digest[0] & 1) and openers
            if use_opener:
                opener = openers[digest[1] % len(openers)]
                first_char = sent[:1].lower()
                new_sent = f"{opener.capitalize()} {first_char}{sent[1:]}" if sent else sent
                self.log_change("voice_opener", sent[:60], new_sent[:60])
                out.append(new_sent)
            else:
                filler = fillers[digest[2] % len(fillers)]
                # insert as a parenthetical aside after the first comma or after
                # the first 4-7 words
                inserted = self._insert_aside(sent, filler)
                self.log_change("voice_aside", sent[:60], inserted[:60])
                out.append(inserted)

        return " ".join(out)

    @staticmethod
    def _insert_aside(sentence: str, filler: str) -> str:
        words = sentence.split(" ")
        if len(words) < 5:
            return f"{filler}, {sentence[:1].lower()}{sentence[1:]}"
        insert_pos = 4
        head = " ".join(words[:insert_pos])
        tail = " ".join(words[insert_pos:])
        return f"{head}, {filler}, {tail}"
