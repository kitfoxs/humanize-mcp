"""Pass 1: em-dash substitution.

The em dash is the single most-recognizable AI tell at the punctuation level
(see research/02 section B.1, severity star-star-star-star-star). This pass
replaces every em dash (U+2014) with a context-appropriate alternative
chosen by simple heuristics. No LLM call.

Strategy choices, in order of preference:

* Parenthetical pair (text "X -- Y -- Z"): convert the matched pair to commas
  or to parentheses, alternating to avoid uniformity.
* Trailing explanation ("X -- Y."): convert to colon or period.
* Inline conjunction ("X -- and Y"): convert to comma.
* Default: comma.

If the configured style sets ``preserve_em_dashes`` to true (creative
fiction, for instance) the pass is a no-op.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .base import PipelinePass

EM_DASH = "\u2014"
EM_DASH_VARIANTS = ("\u2014", " - ", " -- ")  # only U+2014 is the AI tell
EM_DASH_CHARS = {"\u2014"}


class EmDashPass(PipelinePass):
    pass_id = 1
    pass_name = "em_dash"

    def apply(self, text: str, config: Dict[str, Any]) -> str:
        self.reset_changes()
        style = config.get("style", {})
        if style.get("preserve_em_dashes", False):
            return text

        intensity = config.get("intensity", "balanced")
        replacement_strategy: str = config.get("replacement_strategy", "auto")
        # By default we strip every em dash. We optionally keep some only when
        # the text is dense with them (more than 4) and intensity is minimal,
        # to avoid over-uniform stripped output.
        keep_ratio = {
            "minimal": 0.25,
            "balanced": 0.0,
            "aggressive": 0.0,
        }.get(intensity, 0.0)

        if EM_DASH not in text:
            return text

        new_text, alt_count = self._replace_pairs(text, replacement_strategy)
        # Only allow "keep some" behaviour when many em dashes remain.
        remaining_dashes = new_text.count(EM_DASH)
        effective_keep_ratio = keep_ratio if remaining_dashes >= 4 else 0.0
        new_text = self._replace_singletons(
            new_text, replacement_strategy, effective_keep_ratio
        )
        return new_text

    def _replace_pairs(self, text: str, strategy: str) -> tuple[str, int]:
        """Replace paired em dashes "X --- Y --- Z" with commas or parens."""

        # match: dash + content (no dash, no newline) + dash
        pattern = re.compile(rf"{EM_DASH}\s*([^{EM_DASH}\n]+?)\s*{EM_DASH}")
        toggle = {"i": 0}

        def _sub(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            if strategy in ("commas", "auto"):
                # alternate commas/parens for variety
                use_parens = (toggle["i"] % 3 == 0) and strategy == "auto"
                toggle["i"] += 1
                if use_parens:
                    out = f" ({inner})"
                else:
                    out = f", {inner},"
            elif strategy == "parens":
                out = f" ({inner})"
            else:
                out = f", {inner},"
            self.log_change("em_dash_pair", match.group(0), out, match.start())
            return out

        new_text, count = pattern.subn(_sub, text)
        return new_text, count

    def _replace_singletons(
        self, text: str, strategy: str, keep_ratio: float
    ) -> str:
        """Replace lone em dashes left over after pair handling."""

        positions: List[int] = [i for i, ch in enumerate(text) if ch == EM_DASH]
        if not positions:
            return text

        # pseudo-random but deterministic: keep every Nth dash if keep_ratio > 0
        keep_indices = set()
        if keep_ratio > 0 and positions:
            stride = max(1, int(round(1.0 / keep_ratio)))
            keep_indices = set(positions[::stride])

        out_chars: List[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch != EM_DASH:
                out_chars.append(ch)
                i += 1
                continue

            if i in keep_indices:
                out_chars.append(ch)
                self.log_change("em_dash_kept", ch, ch, i)
                i += 1
                continue

            replacement = self._pick_singleton_replacement(text, i, strategy)
            self.log_change("em_dash_singleton", ch, replacement.strip() or ",", i)

            # consume any whitespace immediately surrounding the dash so we
            # don't emit double spaces.
            while out_chars and out_chars[-1] == " ":
                out_chars.pop()
            out_chars.append(replacement)
            i += 1
            while i < len(text) and text[i] == " ":
                i += 1

        return "".join(out_chars)

    @staticmethod
    def _pick_singleton_replacement(text: str, idx: int, strategy: str) -> str:
        if strategy == "comma":
            return ", "
        if strategy == "period":
            return ". "
        if strategy == "semicolon":
            return "; "
        if strategy == "and":
            return " and "

        # auto: look at what follows the dash and decide.
        after = text[idx + 1 : idx + 60].lstrip()
        before = text[max(0, idx - 60) : idx].rstrip()

        if after.startswith(("and ", "but ", "so ", "or ")):
            return ", "
        if after[:1].isupper() and before.endswith((".", "!", "?")) is False:
            # looks like a new sentence is starting -> period
            return ". "
        # if the surrounding clause is short, prefer a comma
        if len(before) < 30 or len(after) < 30:
            return ", "
        # otherwise default to a semicolon (joins independent clauses)
        return "; "
