"""Pass 2: lexical substitution of the AI "delve cluster".

Replaces overused AI vocabulary documented in research/02 section A.1
(Kobak et al. 2024, Juzek and Ward 2024) with simpler, register-appropriate
alternatives. The substitution table lives at ``styles/lexical_substitutions.json``
so it can be tuned without code changes; per-style overrides extend or
suppress entries.

The pass is deterministic given a seed: when a target word has multiple
alternatives the choice is taken from a hash of the surrounding text, so
the same input always produces the same output without random.choice.

At ``minimal`` intensity the pass replaces only the highest-severity
markers; at ``balanced`` it handles the canonical list; at ``aggressive``
it also touches discourse markers and hedging vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List

from .base import PipelinePass

DEFAULT_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "styles",
    "lexical_substitutions.json",
)


def _load_table(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


_TABLE_CACHE: Dict[str, Dict[str, Any]] = {}


def get_table(path: str = DEFAULT_TABLE_PATH) -> Dict[str, Any]:
    if path not in _TABLE_CACHE:
        _TABLE_CACHE[path] = _load_table(path)
    return _TABLE_CACHE[path]


class LexicalPass(PipelinePass):
    pass_id = 2
    pass_name = "lexical"

    def apply(self, text: str, config: Dict[str, Any]) -> str:
        self.reset_changes()
        style = config.get("style", {})
        intensity = config.get("intensity", "balanced")
        seed = config.get("seed", 42)

        table = get_table(config.get("table_path", DEFAULT_TABLE_PATH))
        per_style_overrides: Dict[str, Any] = style.get("lexical_substitutions", {})
        merged = self._merge_table(table, per_style_overrides)

        # severity threshold (1-5 stars). Higher threshold = fewer substitutions.
        threshold = {"minimal": 4, "balanced": 3, "aggressive": 2}.get(intensity, 3)

        out = text
        for entry in merged["entries"]:
            if entry.get("severity", 3) < threshold:
                continue
            if entry.get("disabled"):
                continue
            out = self._substitute_entry(out, entry, seed)
        return out

    def _merge_table(
        self, base: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        merged_entries = list(base.get("entries", []))
        index = {e["pattern"]: i for i, e in enumerate(merged_entries)}

        for ov_pattern, ov in (overrides or {}).items():
            if ov_pattern in index:
                merged_entries[index[ov_pattern]] = {
                    **merged_entries[index[ov_pattern]],
                    **ov,
                    "pattern": ov_pattern,
                }
            else:
                merged_entries.append({"pattern": ov_pattern, **ov})

        return {"entries": merged_entries}

    def _substitute_entry(
        self, text: str, entry: Dict[str, Any], seed: int
    ) -> str:
        pattern = entry["pattern"]
        alternatives = entry.get("alternatives", [])
        if not alternatives:
            return text

        # word-boundary regex, case-insensitive
        flags = re.IGNORECASE if entry.get("case_insensitive", True) else 0
        if entry.get("regex", False):
            compiled = re.compile(pattern, flags)
        else:
            compiled = re.compile(rf"\b{re.escape(pattern)}\b", flags)

        max_per_para = entry.get("max_per_paragraph")

        # paragraph-aware substitution so we don't replace every instance
        paragraphs = re.split(r"(\n\s*\n)", text)
        out_parts: List[str] = []
        for chunk in paragraphs:
            if chunk.strip() == "" or chunk.startswith("\n"):
                out_parts.append(chunk)
                continue
            replaced = self._sub_in_paragraph(
                chunk, compiled, alternatives, seed, max_per_para
            )
            out_parts.append(replaced)
        return "".join(out_parts)

    def _sub_in_paragraph(
        self,
        para: str,
        compiled: re.Pattern[str],
        alternatives: List[str],
        seed: int,
        max_per_para: Any,
    ) -> str:
        replacements_done = {"n": 0}
        max_n = max_per_para if isinstance(max_per_para, int) else None

        def _sub(match: re.Match[str]) -> str:
            if max_n is not None and replacements_done["n"] >= max_n:
                return match.group(0)
            original = match.group(0)
            choice = self._choose_alternative(
                alternatives, original, match.start(), seed
            )
            replaced = self._match_case(original, choice)
            self.log_change("lexical", original, replaced, match.start())
            replacements_done["n"] += 1
            return replaced

        return compiled.sub(_sub, para)

    @staticmethod
    def _choose_alternative(
        alternatives: List[str], original: str, offset: int, seed: int
    ) -> str:
        digest = hashlib.sha1(
            f"{seed}:{offset}:{original.lower()}".encode("utf-8")
        ).digest()
        idx = digest[0] % len(alternatives)
        return alternatives[idx]

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        if original.isupper():
            return replacement.upper()
        if original[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement
