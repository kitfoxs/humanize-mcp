"""AI tells linter.

Scans text for the AI fingerprints catalogued in research/02 and returns
a structured list of :class:`Tell` records. Used by the MCP
``detect_tells`` tool. The detector is read-only; it makes no edits.

Categories covered:

* Lexical (delve cluster, discourse markers, hedging, evaluative adjectives)
* Punctuation (em dashes, smart quotes)
* Structural (parallel constructions, copular metaphors, tricolons)
* Conversational scaffolding (great question, let me know if, as an AI)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Pattern, Tuple


@dataclass
class Tell:
    line: int
    char_offset: int
    end_offset: int
    tell_type: str
    severity: int
    matched_text: str
    suggestion: str
    category: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "line": self.line,
            "char_offset": self.char_offset,
            "end_offset": self.end_offset,
            "tell_type": self.tell_type,
            "severity": self.severity,
            "matched_text": self.matched_text,
            "suggestion": self.suggestion,
            "category": self.category,
        }


@dataclass
class TellPattern:
    name: str
    pattern: Pattern[str]
    severity: int
    suggestion: str
    category: str


def _wb(words: Iterable[str]) -> Pattern[str]:
    """Word-boundary alternation regex (case-insensitive)."""
    joined = "|".join(re.escape(w) for w in words)
    return re.compile(rf"\b({joined})\b", re.IGNORECASE)


_PATTERNS: List[TellPattern] = [
    # Lexical -- delve cluster (research/02 A.1)
    TellPattern(
        "delve_cluster_high",
        _wb([
            "delve", "delving", "delved",
            "intricate", "intricacies",
            "underscore", "underscores", "underscoring",
            "tapestry",
            "meticulous", "meticulously",
            "leverage", "leveraging",
            "navigate", "navigating",
            "realm", "realms",
            "pivotal", "paramount",
        ]),
        severity=5,
        suggestion="Replace with a simpler synonym; this is the strongest lexical AI marker.",
        category="lexical",
    ),
    TellPattern(
        "delve_cluster_mid",
        _wb([
            "nuanced", "nuances",
            "multifaceted",
            "holistic",
            "robust",
            "comprehensive",
            "crucial", "crucially",
            "seamless", "seamlessly",
            "garner", "garnered",
            "fosters", "fostering",
            "testament",
            "myriad",
            "plethora",
        ]),
        severity=3,
        suggestion="Replace with a simpler synonym.",
        category="lexical",
    ),
    TellPattern(
        "delve_cluster_low",
        _wb([
            "paradigm", "facet", "facets", "harness",
            "cutting-edge", "groundbreaking", "transformative",
        ]),
        severity=2,
        suggestion="Use a more concrete word.",
        category="lexical",
    ),
    # Discourse markers (research/02 A.2)
    TellPattern(
        "discourse_marker_initial",
        re.compile(
            r"(?m)^(?:Moreover|Furthermore|Additionally|However|Nevertheless|"
            r"Nonetheless|Consequently|Therefore|Thus|Ultimately)\b",
        ),
        severity=4,
        suggestion="Drop the connector or replace with 'And'/'But'.",
        category="discourse",
    ),
    TellPattern(
        "it_should_be_noted",
        re.compile(
            r"\b(?:it (?:is|should be) (?:important|worth) (?:to note|noting)"
            r"|it should be noted)\b",
            re.IGNORECASE,
        ),
        severity=4,
        suggestion="Delete the meta-comment; state the point directly.",
        category="discourse",
    ),
    TellPattern(
        "in_conclusion",
        re.compile(
            r"\b(?:in conclusion|to conclude|to summarize|in summary)\b",
            re.IGNORECASE,
        ),
        severity=4,
        suggestion="Don't announce the conclusion. Just write it.",
        category="discourse",
    ),
    # Hedging vocabulary (research/02 A.3)
    TellPattern(
        "hedge_cluster",
        _wb([
            "arguably", "potentially", "possibly", "perhaps",
            "various", "numerous",
        ]),
        severity=2,
        suggestion="Drop the hedge or use a specific qualifier.",
        category="hedge",
    ),
    # Evaluative adjectives (research/02 A.4)
    TellPattern(
        "evaluative_cluster",
        _wb([
            "fascinating", "intriguing", "compelling", "captivating", "remarkable",
            "groundbreaking", "revolutionary", "unprecedented", "transformative",
            "profound", "striking",
            "vital", "essential",
        ]),
        severity=3,
        suggestion="Show, don't tell. Replace with concrete description.",
        category="evaluative",
    ),
    # Punctuation (research/02 B.1, B.4)
    TellPattern(
        "em_dash",
        re.compile(r"\u2014"),
        severity=5,
        suggestion="Replace em dash with comma, parentheses, period, or semicolon.",
        category="punctuation",
    ),
    TellPattern(
        "smart_quote",
        re.compile(r"[\u201c\u201d\u2018\u2019]"),
        severity=1,
        suggestion="Smart quotes can suggest LLM origin; consider straight quotes.",
        category="punctuation",
    ),
    # Structural (research/02 C.3)
    TellPattern(
        "not_x_but_y",
        re.compile(
            r"\b(?:it'?s|this is|that's)\s+not\s+(?:just\s+)?[^.;]{2,80}?,?\s+"
            r"(?:it'?s|but)\s+[^.;]{2,120}?[.;]",
            re.IGNORECASE,
        ),
        severity=4,
        suggestion="Rewrite as a single positive statement.",
        category="structural",
    ),
    TellPattern(
        "not_just_but",
        re.compile(r"\bnot just\s+[^.;]{2,80}?,?\s+but\s+[^.;]{2,120}?[.;]", re.IGNORECASE),
        severity=4,
        suggestion="Rewrite as a single positive statement.",
        category="structural",
    ),
    TellPattern(
        "x_is_a_y_of_z",
        re.compile(
            r"\b[A-Z][\w-]+(?:\s+[\w-]+){0,2}\s+is\s+(?:a|the)\s+"
            r"(?:cornerstone|engine|foundation|backbone|heart|soul|key|gateway|"
            r"bedrock|essence|pillar|hallmark|catalyst|driver)\s+of\s+",
            re.IGNORECASE,
        ),
        severity=3,
        suggestion="Replace the metaphor template with a concrete claim.",
        category="structural",
    ),
    # Conversational scaffolding (research/02 F)
    TellPattern(
        "opening_compliment",
        re.compile(
            r"^\s*(?:Great question|What an interesting topic|"
            r"I'?d be happy to help|Certainly|Of course|Absolutely)[!,.:]",
            re.IGNORECASE | re.MULTILINE,
        ),
        severity=5,
        suggestion="Strip the conversational opener.",
        category="scaffolding",
    ),
    TellPattern(
        "self_reference",
        re.compile(
            r"\bAs an AI(?: language model)?\b|"
            r"\bI'm just an AI\b|"
            r"\bWhile I don'?t have personal experiences\b",
            re.IGNORECASE,
        ),
        severity=5,
        suggestion="Strip AI self-reference entirely.",
        category="scaffolding",
    ),
    TellPattern(
        "closing_offer",
        re.compile(
            r"\b(?:Let me know if (?:you'?d like|you have|you need)|"
            r"Feel free to (?:ask|reach out)|"
            r"I hope this helps|Hope (?:that|this) helps)\b",
            re.IGNORECASE,
        ),
        severity=5,
        suggestion="Strip the closing offer.",
        category="scaffolding",
    ),
    # Optimism / cliché closer (research/02 E.7)
    TellPattern(
        "balanced_approach_closer",
        re.compile(
            r"\ba balanced approach (?:that|is needed|will be required)\b",
            re.IGNORECASE,
        ),
        severity=4,
        suggestion="Replace with a specific recommendation or unresolved tension.",
        category="closing",
    ),
]


class TellsDetector:
    """Scans text for AI tells. Returns a list of :class:`Tell` records."""

    def __init__(
        self,
        patterns: Optional[List[TellPattern]] = None,
        min_severity: int = 1,
    ) -> None:
        self.patterns = patterns or _PATTERNS
        self.min_severity = min_severity

    def detect(self, text: str) -> List[Tell]:
        if not text:
            return []
        # precompute line offsets
        line_starts: List[int] = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                line_starts.append(i + 1)

        out: List[Tell] = []
        for pattern in self.patterns:
            if pattern.severity < self.min_severity:
                continue
            for m in pattern.pattern.finditer(text):
                line_no = self._line_for_offset(line_starts, m.start())
                out.append(
                    Tell(
                        line=line_no,
                        char_offset=m.start(),
                        end_offset=m.end(),
                        tell_type=pattern.name,
                        severity=pattern.severity,
                        matched_text=m.group(0),
                        suggestion=pattern.suggestion,
                        category=pattern.category,
                    )
                )

        out.sort(key=lambda t: (t.char_offset, -t.severity))
        return out

    @staticmethod
    def _line_for_offset(line_starts: List[int], offset: int) -> int:
        # binary search for the largest line_start <= offset
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1  # 1-indexed lines


def detect_tells(text: str, min_severity: int = 1) -> List[Dict[str, object]]:
    """Module-level convenience wrapper used by the MCP server."""
    detector = TellsDetector(min_severity=min_severity)
    return [t.to_dict() for t in detector.detect(text)]


def summarize_tells(tells: List[Tell]) -> Dict[str, object]:
    """Roll up detected tells by category and severity."""
    by_category: Dict[str, int] = {}
    by_severity: Dict[int, int] = {}
    for t in tells:
        by_category[t.category] = by_category.get(t.category, 0) + 1
        by_severity[t.severity] = by_severity.get(t.severity, 0) + 1
    return {
        "total": len(tells),
        "by_category": by_category,
        "by_severity": by_severity,
    }
