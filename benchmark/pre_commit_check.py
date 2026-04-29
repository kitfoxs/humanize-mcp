"""Lightweight "lint your output before posting" helper.

Provides :func:`pre_commit_check` which runs the cheap detectors over a
text and returns a structured :class:`PreCommitResult` with:

* a pass/fail verdict against a configurable AI-probability threshold
* a list of concrete suggestions for what to fix (drawn from the
  heuristic detector's tell-hits and stylometric stats)
* the underlying :class:`BenchmarkReport` for transparency

The MCP server's ``detect_tells()`` tool uses this indirectly.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .benchmark_suite import BenchmarkReport, BenchmarkSuite
from .detectors import HeuristicDetector, RobertaOpenAIDetector


class PreCommitResult(BaseModel):
    """Result of a pre-commit check on a text.

    Attributes:
        passed: True if the AI score is below ``threshold``.
        score: The AI probability used for the pass/fail decision.
        threshold: Threshold above which the check fails.
        suggestions: Human-readable suggestions for what to fix.
        report: Underlying :class:`BenchmarkReport` for inspection.
    """

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    suggestions: list[str] = Field(default_factory=list)
    report: BenchmarkReport


def _suggestions_from_report(report: BenchmarkReport) -> list[str]:
    """Convert a report's heuristic details into actionable suggestions."""
    suggestions: list[str] = []
    stats = report.statistics
    if stats.tells_count > 0:
        suggestions.append(
            f"Found {stats.tells_count} surface AI tells; consider rewriting: "
            f"{', '.join(stats.tells_hits[:6])}"
        )
    if stats.sentence_count >= 3 and stats.burstiness < 0.3:
        suggestions.append(
            f"Sentence-length variance is low (burstiness={stats.burstiness:.2f}). "
            "Mix short and long sentences to sound more human."
        )
    if stats.lexical_diversity < 0.4 and stats.word_count >= 60:
        suggestions.append(
            f"Lexical diversity is low ({stats.lexical_diversity:.2f}). "
            "Vary word choice; you are repeating the same vocabulary."
        )
    if stats.avg_sentence_length_words > 28:
        suggestions.append(
            "Average sentence is long. Break some sentences into shorter ones."
        )
    if not suggestions:
        suggestions.append(
            "No structural issues found. If the score is still high, "
            "consider voice/register changes that the heuristic does not see."
        )
    return suggestions


def pre_commit_check(
    text: str,
    *,
    threshold: float = 0.6,
    use_classifier: bool = True,
    suite: Optional[BenchmarkSuite] = None,
) -> PreCommitResult:
    """Run a fast pre-commit check on ``text``.

    Args:
        text: The text to check.
        threshold: AI-probability threshold. Default 0.6 — anything at or
            above is treated as "needs more work".
        use_classifier: If True (default), include the RoBERTa-base OpenAI
            detector along with the heuristic; the resulting score is the
            mean of the two. If False, only the heuristic is used (faster,
            no model download).
        suite: Optional pre-built :class:`BenchmarkSuite` to use instead of
            constructing one. Useful for tests or for callers that want to
            cache loaded models across many calls.

    Returns:
        A :class:`PreCommitResult` describing the verdict and suggestions.
    """
    if suite is None:
        detectors = [HeuristicDetector()]
        if use_classifier:
            detectors.append(RobertaOpenAIDetector())
        suite = BenchmarkSuite(detectors=detectors)
    report = suite.score(text)
    score = (
        report.trusted_mean_ai_probability
        if report.trusted_mean_ai_probability is not None
        else report.raw_mean_ai_probability
    )
    suggestions = _suggestions_from_report(report)
    return PreCommitResult(
        passed=score < threshold,
        score=round(score, 4),
        threshold=threshold,
        suggestions=suggestions,
        report=report,
    )


__all__ = ["pre_commit_check", "PreCommitResult"]
