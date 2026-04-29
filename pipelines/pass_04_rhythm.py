"""Pass 4: rhythm / burstiness injection.

Detects and corrects sentence-length uniformity, the single most-used
statistical feature in commercial detectors (research/02 section C.1).
We compute current burstiness (coefficient of variation of sentence
lengths) and, if it falls below a target, split overly long sentences
on conjunctions and merge short adjacent fragments.

Uses NLTK's punkt tokenizer when available; falls back to a simple
regex split if not.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Tuple

from .base import PipelinePass

try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize

    try:
        _nltk_sent_tokenize("Sanity check.")
        _SENT_TOKENIZE = _nltk_sent_tokenize
    except LookupError:
        _SENT_TOKENIZE = None
except ImportError:
    _SENT_TOKENIZE = None


_FALLBACK_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'(])")


def sentence_split(text: str) -> List[str]:
    if _SENT_TOKENIZE is not None:
        return _SENT_TOKENIZE(text)
    return [s.strip() for s in _FALLBACK_SENT_RE.split(text) if s.strip()]


SPLIT_CONJUNCTIONS = (
    ", and ",
    ", but ",
    ", so ",
    ", because ",
    ", which ",
    "; ",
)


class RhythmPass(PipelinePass):
    pass_id = 4
    pass_name = "rhythm"

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
        intensity = config.get("intensity", "balanced")
        target_cv = {
            "minimal": 0.45,
            "balanced": 0.6,
            "aggressive": 0.8,
        }.get(intensity, 0.6)

        sentences = sentence_split(text)
        if len(sentences) < 4:
            return text

        cv = self._coefficient_of_variation(sentences)
        if cv >= target_cv:
            return text

        # We have a uniformity problem. Split a few long sentences and merge
        # a few adjacent short ones to add variance.
        sentences = self._split_long_sentences(sentences, target_cv)
        sentences = self._merge_short_adjacent(sentences, target_cv)

        return self._reassemble(text, sentences)

    @staticmethod
    def _coefficient_of_variation(sents: List[str]) -> float:
        lengths = [len(s.split()) for s in sents]
        if len(lengths) < 2:
            return 0.0
        mean = statistics.mean(lengths)
        if mean == 0:
            return 0.0
        return statistics.pstdev(lengths) / mean

    def _split_long_sentences(
        self, sents: List[str], target_cv: float
    ) -> List[str]:
        out: List[str] = []
        for s in sents:
            if len(s.split()) < 22:
                out.append(s)
                continue
            split_at = self._find_split_point(s)
            if split_at is None:
                out.append(s)
                continue
            left, right = s[:split_at].rstrip(", ;"), s[split_at:].lstrip()
            # ensure both halves end like sentences
            if not left.endswith((".", "!", "?")):
                left += "."
            right = right[:1].upper() + right[1:] if right else right
            self.log_change("rhythm_split", s, f"{left} {right}")
            out.append(left)
            out.append(right)
        return out

    @staticmethod
    def _find_split_point(s: str) -> int | None:
        # split at the conjunction nearest the middle
        mid = len(s) // 2
        best_idx = None
        best_dist = 10**9
        for conj in SPLIT_CONJUNCTIONS:
            idx = 0
            while True:
                idx = s.find(conj, idx)
                if idx == -1:
                    break
                dist = abs(idx - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx + len(conj)
                idx += 1
        return best_idx

    def _merge_short_adjacent(
        self, sents: List[str], target_cv: float
    ) -> List[str]:
        out: List[str] = []
        i = 0
        while i < len(sents):
            current = sents[i]
            if (
                i + 1 < len(sents)
                and len(current.split()) < 6
                and len(sents[i + 1].split()) < 9
            ):
                merged = current.rstrip(".!?") + ", " + sents[i + 1][:1].lower() + sents[i + 1][1:]
                self.log_change("rhythm_merge", f"{current} {sents[i + 1]}", merged)
                out.append(merged)
                i += 2
                continue
            out.append(current)
            i += 1
        return out

    @staticmethod
    def _reassemble(original: str, sentences: List[str]) -> str:
        # naive: rejoin with single spaces, preserve trailing newline if present
        out = " ".join(s.strip() for s in sentences if s.strip())
        if original.endswith("\n"):
            out += "\n"
        return out
