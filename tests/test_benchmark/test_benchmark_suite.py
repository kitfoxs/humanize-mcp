"""Tests for benchmark.benchmark_suite orchestration."""

from __future__ import annotations

from typing import ClassVar

import pytest

from benchmark.benchmark_suite import (
    BenchmarkSuite,
    score_text,
    text_statistics,
)
from benchmark.detectors import (
    TIER_CLASSIFIER,
    Detector,
    DetectorScore,
    HeuristicDetector,
)


class _FakeAIDetector(Detector):
    """Always returns ai_probability=0.9, no bias notes."""

    name = "fake_ai"
    tier = TIER_CLASSIFIER
    bias_notes: ClassVar[list[str]] = []

    def _score_impl(self, text: str) -> DetectorScore:
        return DetectorScore.from_ai_prob(self.name, 0.9, tier=self.tier)


class _FakeHumanDetector(Detector):
    """Always returns ai_probability=0.1, no bias notes."""

    name = "fake_human"
    tier = TIER_CLASSIFIER
    bias_notes: ClassVar[list[str]] = []

    def _score_impl(self, text: str) -> DetectorScore:
        return DetectorScore.from_ai_prob(self.name, 0.1, tier=self.tier)


class _BiasedDetector(Detector):
    """Always 1.0; carries a bias note so it should be excluded from trusted_mean."""

    name = "biased"
    tier = TIER_CLASSIFIER
    bias_notes: ClassVar[list[str]] = ["known to over-flag formal text"]

    def _score_impl(self, text: str) -> DetectorScore:
        return DetectorScore.from_ai_prob(
            self.name, 1.0, tier=self.tier, bias_notes=self.bias_notes
        )


class _BrokenDetector(Detector):
    """Always raises; should produce an error score, not crash the suite."""

    name = "broken"
    tier = TIER_CLASSIFIER

    def _score_impl(self, text: str) -> DetectorScore:
        raise RuntimeError("intentional failure")


def test_score_returns_per_detector_results() -> None:
    suite = BenchmarkSuite(detectors=[_FakeAIDetector(), _FakeHumanDetector()])
    report = suite.score("hello world. This is a test.")
    assert {s.detector for s in report.detector_scores} == {"fake_ai", "fake_human"}
    assert pytest.approx(report.raw_mean_ai_probability) == 0.5
    assert report.trusted_mean_ai_probability is not None
    assert pytest.approx(report.trusted_mean_ai_probability) == 0.5


def test_trusted_mean_excludes_biased_detector() -> None:
    suite = BenchmarkSuite(
        detectors=[_FakeHumanDetector(), _BiasedDetector()],
    )
    report = suite.score("hello world. testing bias filter.")
    # Raw mean averages both: (0.1 + 1.0) / 2 = 0.55
    assert pytest.approx(report.raw_mean_ai_probability) == 0.55
    # Trusted mean excludes the biased one: 0.1
    assert report.trusted_mean_ai_probability is not None
    assert pytest.approx(report.trusted_mean_ai_probability) == 0.1
    assert any("biased" in w for w in report.bias_warnings)


def test_broken_detector_does_not_crash_suite() -> None:
    suite = BenchmarkSuite(detectors=[_FakeAIDetector(), _BrokenDetector()])
    report = suite.score("hello world.")
    broken = next(s for s in report.detector_scores if s.detector == "broken")
    assert broken.error != ""
    # The valid detector still contributed.
    valid = next(s for s in report.detector_scores if s.detector == "fake_ai")
    assert valid.error == ""
    # Aggregation excludes the errored score.
    assert pytest.approx(report.raw_mean_ai_probability) == 0.9


def test_compare_returns_positive_delta_when_after_scores_lower() -> None:
    suite = BenchmarkSuite(detectors=[_FakeAIDetector()])
    # Both texts go through the fake detector; deltas will be 0 because the
    # fake detector returns 0.9 regardless. Use a real heuristic detector
    # to actually exercise the delta logic.
    suite = BenchmarkSuite(detectors=[HeuristicDetector()])
    obvious_ai = (
        "It is important to note that in today's world we must delve into "
        "the realm of robust solutions. Furthermore, we should leverage "
        "these tools. In conclusion, this is multifaceted."
    )
    casual = (
        "yeah honestly i just kept poking at it. eventually it worked. "
        "wasn't sure why."
    )
    cmp = suite.compare(before_text=obvious_ai, after_text=casual)
    assert cmp.mean_delta_raw > 0
    assert "heuristic" in cmp.per_detector_delta


def test_score_rejects_empty_input() -> None:
    suite = BenchmarkSuite(detectors=[_FakeAIDetector()])
    with pytest.raises(ValueError):
        suite.score("")


def test_score_text_convenience_uses_default_suite() -> None:
    # Use a tiny custom suite indirectly by calling score_text; we check it
    # works and emits valid output. include_optional=False keeps it cheap.
    report = score_text(
        "Hello world. This is just a smoke test.", include_optional=False
    )
    assert report.statistics.word_count > 0
    assert 0.0 <= report.raw_mean_ai_probability <= 1.0


def test_text_statistics_basic_fields() -> None:
    stats = text_statistics(
        "First sentence. Second sentence is a bit longer. Third one is even longer than that one."
    )
    assert stats.sentence_count == 3
    assert stats.word_count > 0
    assert stats.avg_sentence_length_words > 0


def test_verdict_string_includes_human_label_for_low_score() -> None:
    suite = BenchmarkSuite(detectors=[_FakeHumanDetector()])
    report = suite.score("Hello world. testing.")
    assert "human" in report.verdict.lower()


def test_verdict_string_includes_ai_label_for_high_score() -> None:
    suite = BenchmarkSuite(detectors=[_FakeAIDetector()])
    report = suite.score("Hello world. testing.")
    assert "ai" in report.verdict.lower()


def test_pre_commit_check_runs_without_classifier() -> None:
    # Avoid downloading the RoBERTa model in CI by disabling the classifier.
    from benchmark import pre_commit_check

    result = pre_commit_check(
        "Hello world. just a quick smoke check.",
        threshold=0.6,
        use_classifier=False,
    )
    assert isinstance(result.passed, bool)
    assert result.suggestions  # never empty
