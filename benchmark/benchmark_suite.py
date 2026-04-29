"""Benchmark suite that orchestrates multiple detectors over text samples.

Provides :class:`BenchmarkSuite` for scoring single or paired (before/after)
text samples and producing structured :class:`BenchmarkReport` objects that
downstream code (CLI, MCP tools, README generators) can serialize.

The suite is designed to be honest about detector calibration. Each detector
contributes its own score along with documented bias notes; the aggregate is
computed two ways:

* ``raw_mean`` — unweighted mean across all detectors (for transparency).
* ``trusted_mean`` — mean restricted to detectors that *do not* carry
  documented bias caveats. This is the number we recommend for headlines.
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import Iterable, Optional

from pydantic import BaseModel, Field

from .detectors import (
    Detector,
    DetectorScore,
    HeuristicDetector,
    burstiness,
    count_tells,
    default_detectors,
    lexical_diversity,
)

LOGGER = logging.getLogger(__name__)


class TextStatistics(BaseModel):
    """Cheap stylometric statistics for a single piece of text."""

    char_count: int
    word_count: int
    sentence_count: int
    avg_sentence_length_words: float
    burstiness: float
    lexical_diversity: float
    tells_count: int
    tells_hits: list[str]


def text_statistics(text: str) -> TextStatistics:
    """Compute :class:`TextStatistics` for ``text``."""
    from .detectors import _split_sentences, _word_tokens

    sents = _split_sentences(text)
    words = _word_tokens(text)
    n_words = len(words)
    n_sents = len(sents)
    avg_len = (n_words / n_sents) if n_sents else 0.0
    n_tells, hits = count_tells(text)
    return TextStatistics(
        char_count=len(text),
        word_count=n_words,
        sentence_count=n_sents,
        avg_sentence_length_words=round(avg_len, 3),
        burstiness=round(burstiness(text), 4),
        lexical_diversity=round(lexical_diversity(text), 4),
        tells_count=n_tells,
        tells_hits=hits,
    )


class BenchmarkReport(BaseModel):
    """Aggregated result of running the suite over a single text.

    Attributes:
        text_preview: First 200 characters, for human inspection.
        statistics: Cheap stylometric statistics for the input.
        detector_scores: One :class:`DetectorScore` per detector that ran.
        raw_mean_ai_probability: Unweighted mean over all non-error scores.
        trusted_mean_ai_probability: Mean restricted to detectors with no
            documented bias caveats. ``None`` if no such detector ran.
        agreement: Stdev of AI probabilities across detectors. Lower means
            detectors agree with each other.
        verdict: Short human-readable summary verdict.
        bias_warnings: Aggregate of all detectors' ``bias_notes`` so they
            are surfaced once in any UI rather than once per detector.
        total_latency_seconds: Sum of per-detector latencies.
    """

    text_preview: str
    statistics: TextStatistics
    detector_scores: list[DetectorScore]
    raw_mean_ai_probability: float
    trusted_mean_ai_probability: Optional[float] = None
    agreement: float = 0.0
    verdict: str = "uncertain"
    bias_warnings: list[str] = Field(default_factory=list)
    total_latency_seconds: float = 0.0


class ComparisonReport(BaseModel):
    """Pair of :class:`BenchmarkReport` values plus deltas.

    Attributes:
        before: Report on the original text.
        after: Report on the humanized text.
        per_detector_delta: ai_prob(before) - ai_prob(after) for each
            detector (positive => humanization reduced AI score).
        mean_delta_raw: Mean per-detector delta over all detectors.
        mean_delta_trusted: Mean per-detector delta restricted to detectors
            without documented bias caveats.
        verdict: Short human-readable summary of the comparison.
    """

    before: BenchmarkReport
    after: BenchmarkReport
    per_detector_delta: dict[str, float]
    mean_delta_raw: float
    mean_delta_trusted: Optional[float] = None
    verdict: str = ""


def _aggregate(scores: list[DetectorScore]) -> tuple[float, Optional[float], float, list[str]]:
    """Compute (raw_mean, trusted_mean, agreement_stdev, bias_warnings)."""
    valid = [s for s in scores if not s.error]
    if not valid:
        return 0.5, None, 0.0, []
    raw = statistics.mean(s.ai_probability for s in valid)
    trusted = [s for s in valid if not s.bias_notes]
    trusted_mean = statistics.mean(s.ai_probability for s in trusted) if trusted else None
    agreement = statistics.pstdev([s.ai_probability for s in valid]) if len(valid) > 1 else 0.0
    bias_warnings: list[str] = []
    seen: set[str] = set()
    for s in valid:
        for note in s.bias_notes:
            tag = f"{s.detector}: {note}"
            if tag not in seen:
                seen.add(tag)
                bias_warnings.append(tag)
    return raw, trusted_mean, agreement, bias_warnings


def _verdict_for(raw_mean: float, trusted_mean: Optional[float], agreement: float) -> str:
    """Produce a short human-readable verdict string."""
    score = trusted_mean if trusted_mean is not None else raw_mean
    if agreement > 0.25:
        confidence_tag = "low confidence (detectors disagree)"
    elif agreement > 0.12:
        confidence_tag = "moderate confidence"
    else:
        confidence_tag = "high confidence"
    if score >= 0.7:
        label = "likely AI"
    elif score >= 0.55:
        label = "leans AI"
    elif score <= 0.3:
        label = "likely human"
    elif score <= 0.45:
        label = "leans human"
    else:
        label = "uncertain"
    return f"{label} ({confidence_tag}; mean ai-prob {score:.2f})"


class BenchmarkSuite:
    """Run multiple detectors over text and produce structured reports.

    Args:
        detectors: Optional list of detector instances to use. If ``None``,
            calls :func:`default_detectors` with ``include_optional=False``.
        include_optional: If ``detectors`` is ``None``, controls whether the
            optional zero-shot detectors (Fast-DetectGPT, Binoculars) and
            Desklib are included in the default set.
    """

    def __init__(
        self,
        detectors: Optional[Iterable[Detector]] = None,
        *,
        include_optional: bool = False,
    ) -> None:
        if detectors is None:
            self.detectors: list[Detector] = default_detectors(
                include_optional=include_optional
            )
        else:
            self.detectors = list(detectors)
        if not self.detectors:
            raise ValueError("BenchmarkSuite needs at least one detector")

    @property
    def detector_names(self) -> list[str]:
        return [d.name for d in self.detectors]

    def score(self, text: str) -> BenchmarkReport:
        """Score a single text across all wrapped detectors."""
        if not text or not text.strip():
            raise ValueError("Cannot score empty text")
        t0 = time.time()
        results: list[DetectorScore] = []
        for det in self.detectors:
            results.append(det.score(text))
        raw_mean, trusted_mean, agreement, bias_warnings = _aggregate(results)
        verdict = _verdict_for(raw_mean, trusted_mean, agreement)
        return BenchmarkReport(
            text_preview=text.strip()[:200],
            statistics=text_statistics(text),
            detector_scores=results,
            raw_mean_ai_probability=round(raw_mean, 4),
            trusted_mean_ai_probability=(
                round(trusted_mean, 4) if trusted_mean is not None else None
            ),
            agreement=round(agreement, 4),
            verdict=verdict,
            bias_warnings=bias_warnings,
            total_latency_seconds=round(time.time() - t0, 3),
        )

    def compare(self, before_text: str, after_text: str) -> ComparisonReport:
        """Score two texts and report the delta per detector."""
        before = self.score(before_text)
        after = self.score(after_text)

        deltas: dict[str, float] = {}
        for b in before.detector_scores:
            a = next(
                (x for x in after.detector_scores if x.detector == b.detector),
                None,
            )
            if a is None or b.error or a.error:
                continue
            deltas[b.detector] = round(b.ai_probability - a.ai_probability, 4)

        if deltas:
            mean_delta_raw = round(statistics.mean(deltas.values()), 4)
            trusted_names = {
                d.detector for d in before.detector_scores if not d.bias_notes and not d.error
            }
            trusted_deltas = [v for k, v in deltas.items() if k in trusted_names]
            mean_delta_trusted = (
                round(statistics.mean(trusted_deltas), 4) if trusted_deltas else None
            )
        else:
            mean_delta_raw = 0.0
            mean_delta_trusted = None

        if mean_delta_raw > 0.15:
            verdict = (
                f"humanization reduced AI score by {mean_delta_raw:+.2f} on average"
            )
        elif mean_delta_raw > 0.0:
            verdict = (
                f"small reduction in AI score ({mean_delta_raw:+.2f}); marginal effect"
            )
        elif mean_delta_raw == 0.0:
            verdict = "no change in AI scores"
        else:
            verdict = (
                f"warning: humanization increased AI score by {-mean_delta_raw:+.2f} on average"
            )

        return ComparisonReport(
            before=before,
            after=after,
            per_detector_delta=deltas,
            mean_delta_raw=mean_delta_raw,
            mean_delta_trusted=mean_delta_trusted,
            verdict=verdict,
        )


def score_text(
    text: str,
    *,
    include_optional: bool = False,
) -> BenchmarkReport:
    """Convenience: score one text using the default detector suite."""
    suite = BenchmarkSuite(include_optional=include_optional)
    return suite.score(text)


__all__ = [
    "BenchmarkSuite",
    "BenchmarkReport",
    "ComparisonReport",
    "TextStatistics",
    "text_statistics",
    "score_text",
]
