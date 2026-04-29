"""Tests for the detector-guided iterative humanizer (Bet 3 of v0.2.0).

These tests use heavy mocking so the suite runs in well under 5 seconds:
the BenchmarkSuite is replaced with a deterministic ``FakeSuite`` that
returns pre-programmed scores, and the paraphrase pass is replaced with a
``FakeParaphrasePass`` that returns controlled candidate lists. Real
integration (loading detector models, running heavy paraphrase) is the
domain of the slow integration suite, not these unit tests.

Reference: docs/REVIEW_v0.1.0.md sections 2.9, 3.3, 3.4, 5 (Bet 3) and
research/04_humanization_techniques.md section 1.3 (Cheng et al. 2025).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import pytest

from pipelines import HumanizationPipeline, HumanizeConfig, HumanizeResult
from pipelines.iterative import IterativeHumanizer, IterativeResult


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeReport:
    """Stand-in for :class:`benchmark.benchmark_suite.BenchmarkReport`.

    Only carries the fields the IterativeHumanizer reads.
    """

    def __init__(self, trusted: Optional[float], raw: float = 0.0) -> None:
        self.trusted_mean_ai_probability = trusted
        self.raw_mean_ai_probability = raw


class FakeSuite:
    """Mock BenchmarkSuite that returns whatever the scorer function says.

    The ``scorer`` callable receives the input text and returns the float
    that ``trusted_mean_ai_probability`` should report. This lets tests
    program the search landscape (e.g. paragraph-2 always scores 0.9, the
    second candidate always scores 0.1, etc.) without any real models.
    """

    def __init__(self, scorer: Callable[[str], float]) -> None:
        self._scorer = scorer
        self.calls: list[str] = []

    def score(self, text: str) -> FakeReport:
        self.calls.append(text)
        value = self._scorer(text)
        return FakeReport(trusted=value, raw=value)


class FakePipeline:
    """Stand-in for :class:`HumanizationPipeline` whose ``run`` is identity.

    Carries a single fake paraphrase pass so the iterative humanizer can
    locate it and call ``paraphrase_candidates``.
    """

    def __init__(self, para_pass: Any, baseline_text: Optional[str] = None) -> None:
        self.passes = [para_pass]
        self._baseline_text = baseline_text

    def run(self, text: str, config: HumanizeConfig) -> HumanizeResult:
        out = self._baseline_text if self._baseline_text is not None else text
        return HumanizeResult(
            text=out,
            passes_applied=["fake"],
            tells_removed_count=0,
            processing_time_ms=0.0,
            pass_log=[],
            warnings=[],
        )


class FakeParaphrasePass:
    """Programmable paraphrase pass.

    Each call to ``paraphrase_candidates`` returns the next item from
    ``responses``, an ordered list of ``list[str]``. If a ``seed`` kwarg
    is forwarded we record it so seed-determinism tests can assert it.
    """

    pass_id = 9
    pass_name = "paraphrase"

    def __init__(self, responses: list[list[str]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def paraphrase_candidates(  # noqa: D401 - test helper
        self, text: str, n: int = 3, **kwargs: Any
    ) -> list[str]:
        self.calls.append({"text": text, "n": n, **kwargs})
        if not self._responses:
            return []
        return self._responses.pop(0)

    def apply(self, text: str, config: dict[str, Any]) -> str:
        # Used by the deterministic fallback path. We append a token so the
        # result is non-empty and not equal to the input, which lets the
        # iterative loop accept it as a single-candidate fallback.
        return text + " [fallback paraphrased]"

    def reset_changes(self) -> None:
        pass

    def changes(self) -> list[Any]:
        return []


class LegacyParaphrasePass:
    """Mimics a ParaphrasePass shipped before Bet 1: no paraphrase_candidates."""

    pass_id = 9
    pass_name = "paraphrase"

    def apply(self, text: str, config: dict[str, Any]) -> str:
        return text + " [legacy fallback paraphrased]"

    def reset_changes(self) -> None:
        pass

    def changes(self) -> list[Any]:
        return []


def _fixed_score_humanizer(
    score: float, *, baseline_text: Optional[str] = None
) -> tuple[IterativeHumanizer, FakeSuite, FakeParaphrasePass]:
    """Build an IterativeHumanizer whose suite always returns ``score``."""
    para = FakeParaphrasePass(responses=[])
    pipeline = FakePipeline(para, baseline_text=baseline_text)
    suite = FakeSuite(lambda _t: score)
    h = IterativeHumanizer(pipeline, suite=suite)  # type: ignore[arg-type]
    return h, suite, para


# ---------------------------------------------------------------------------
# Returns immediately if input already scores below target
# ---------------------------------------------------------------------------


def test_returns_immediately_when_baseline_below_target() -> None:
    """If the baseline humanization already meets the target, no iteration runs."""
    h, _suite, para = _fixed_score_humanizer(score=0.05)
    result = h.humanize_and_verify(
        "First paragraph.\n\nSecond paragraph.",
        target_ai_score=0.15,
    )
    assert isinstance(result, IterativeResult)
    assert result.iterations == 0
    assert result.target_reached is True
    assert result.final_score == pytest.approx(0.05)
    assert result.per_iteration_scores == [pytest.approx(0.05)]
    # Paraphrase pass should never have been asked for candidates.
    assert para.calls == []


# ---------------------------------------------------------------------------
# Loop terminates at max_iterations when target is unreachable
# ---------------------------------------------------------------------------


def test_runs_at_most_max_iterations_when_target_unreachable() -> None:
    """If the detector keeps reporting high scores, the loop must respect the cap."""
    para = FakeParaphrasePass(
        responses=[
            ["candidate A1", "candidate A2", "candidate A3"],
            ["candidate B1", "candidate B2", "candidate B3"],
            ["candidate C1", "candidate C2", "candidate C3"],
            ["candidate D1", "candidate D2", "candidate D3"],
            ["candidate E1", "candidate E2", "candidate E3"],
        ]
    )
    pipeline = FakePipeline(para)
    # Always report the score is above target; but each new candidate is
    # slightly better than the worst paragraph (so the swap is accepted)
    # by giving fresh candidates a marginally lower score than originals.
    call_count = {"n": 0}

    def scorer(text: str) -> float:
        call_count["n"] += 1
        if "candidate" in text:
            # Candidates are always a hair better than the original so the
            # loop accepts them; but never below 0.5.
            return 0.6
        return 0.9

    suite = FakeSuite(scorer)
    h = IterativeHumanizer(pipeline, suite=suite)  # type: ignore[arg-type]

    result = h.humanize_and_verify(
        "Para one.\n\nPara two.",
        target_ai_score=0.1,
        max_iterations=2,
        candidates_per_iteration=3,
    )
    assert result.target_reached is False
    assert result.iterations <= 2
    # Worst-paragraph identification + 3 candidates + whole-text rescore
    # per iteration means well above 0 calls were made.
    assert call_count["n"] > 5


# ---------------------------------------------------------------------------
# Loop selects the lowest-scoring candidate
# ---------------------------------------------------------------------------


def test_selects_lowest_scoring_candidate() -> None:
    """The replacement at each iteration must be the candidate with the lowest score."""
    para = FakeParaphrasePass(
        responses=[
            # Candidate B is the cleanest; the loop must pick it.
            ["CAND_A_high", "CAND_B_low", "CAND_C_mid"],
        ]
    )
    pipeline = FakePipeline(para)

    def scorer(text: str) -> float:
        if "CAND_A_high" in text:
            return 0.8
        if "CAND_B_low" in text:
            # Both the candidate alone and the whole-text containing it
            # score below target; the loop should pick this and exit.
            return 0.05
        if "CAND_C_mid" in text:
            return 0.4
        # Initial whole-text score, plus per-paragraph scoring on baseline.
        return 0.7

    suite = FakeSuite(scorer)
    h = IterativeHumanizer(pipeline, suite=suite)  # type: ignore[arg-type]
    result = h.humanize_and_verify(
        "Only paragraph here.",
        target_ai_score=0.15,
        max_iterations=3,
        candidates_per_iteration=3,
    )
    assert result.target_reached is True
    assert "CAND_B_low" in result.text
    assert "CAND_A_high" not in result.text
    assert "CAND_C_mid" not in result.text
    assert result.iterations == 1


# ---------------------------------------------------------------------------
# target_reached field is correctly set
# ---------------------------------------------------------------------------


def test_target_reached_true_when_score_meets_target() -> None:
    para = FakeParaphrasePass(responses=[["clean candidate"]])
    pipeline = FakePipeline(para)
    scores = [0.9, 0.9, 0.05, 0.05]  # baseline whole, baseline para, candidate, final whole
    score_iter = iter(scores)

    def scorer(_t: str) -> float:
        try:
            return next(score_iter)
        except StopIteration:
            return 0.05

    suite = FakeSuite(scorer)
    h = IterativeHumanizer(pipeline, suite=suite)  # type: ignore[arg-type]
    result = h.humanize_and_verify(
        "Text.",
        target_ai_score=0.15,
        candidates_per_iteration=1,
    )
    assert result.target_reached is True


def test_target_reached_false_when_score_misses_target() -> None:
    h, _suite, _para = _fixed_score_humanizer(score=0.5)
    result = h.humanize_and_verify(
        "Text.",
        target_ai_score=0.15,
        max_iterations=1,
    )
    assert result.target_reached is False


# ---------------------------------------------------------------------------
# Worst-paragraph identification
# ---------------------------------------------------------------------------


def test_worst_paragraph_is_correctly_identified() -> None:
    """When several paragraphs are present, the highest-scoring one is replaced."""
    para = FakeParaphrasePass(
        responses=[["[REPLACED PARAGRAPH]"]],
    )
    pipeline = FakePipeline(para)

    def scorer(text: str) -> float:
        # Whole-text scoring (the call before the per-paragraph scan).
        if "ALPHA" in text and "BETA" in text and "GAMMA" in text:
            return 0.8
        if "[REPLACED PARAGRAPH]" in text:
            # After the swap. Below target so the loop exits.
            return 0.1
        # Per-paragraph scan: only one of the three is "worst".
        if "BETA" in text and "ALPHA" not in text:
            return 0.95
        if "ALPHA" in text:
            return 0.5
        if "GAMMA" in text:
            return 0.4
        return 0.5

    suite = FakeSuite(scorer)
    h = IterativeHumanizer(pipeline, suite=suite)  # type: ignore[arg-type]
    text = "ALPHA paragraph.\n\nBETA paragraph.\n\nGAMMA paragraph."
    result = h.humanize_and_verify(
        text,
        target_ai_score=0.15,
        max_iterations=1,
        candidates_per_iteration=1,
    )
    # The BETA paragraph must have been replaced; ALPHA and GAMMA must remain.
    assert "[REPLACED PARAGRAPH]" in result.text
    assert "ALPHA paragraph." in result.text
    assert "GAMMA paragraph." in result.text
    assert "BETA paragraph." not in result.text


# ---------------------------------------------------------------------------
# Graceful fallback when paraphrase_candidates is missing
# ---------------------------------------------------------------------------


def test_falls_back_when_paraphrase_candidates_missing() -> None:
    """If the paraphrase pass lacks paraphrase_candidates, fall back to apply()."""
    para = LegacyParaphrasePass()
    pipeline = FakePipeline(para)

    # Always report above target so we attempt at least one iteration.
    suite = FakeSuite(lambda t: 0.5 if "legacy fallback paraphrased" not in t else 0.4)
    h = IterativeHumanizer(pipeline, suite=suite)  # type: ignore[arg-type]
    result = h.humanize_and_verify(
        "Single para.",
        target_ai_score=0.1,
        max_iterations=1,
        candidates_per_iteration=3,
    )
    # The loop must not crash. It should at minimum produce a valid result.
    assert isinstance(result, IterativeResult)
    # Either the fallback paraphrase appended its token, or the loop exited
    # cleanly without doing harm.
    assert result.target_reached is False or "legacy fallback paraphrased" in result.text


def test_handles_empty_candidate_list_without_crashing() -> None:
    """If paraphrase_candidates returns []  the loop logs and stops, not crashes."""
    para = FakeParaphrasePass(responses=[[]])
    pipeline = FakePipeline(para)
    suite = FakeSuite(lambda _t: 0.5)
    h = IterativeHumanizer(pipeline, suite=suite)  # type: ignore[arg-type]
    result = h.humanize_and_verify(
        "Text here.",
        target_ai_score=0.1,
        max_iterations=2,
    )
    assert isinstance(result, IterativeResult)
    assert result.target_reached is False


# ---------------------------------------------------------------------------
# Deterministic seed propagation
# ---------------------------------------------------------------------------


def test_seed_is_forwarded_to_paraphrase_candidates() -> None:
    """The iteration seed (base seed + iteration index) must reach paraphrase_candidates."""
    para = FakeParaphrasePass(responses=[["c1", "c2", "c3"]])
    pipeline = FakePipeline(para)
    # Stay above target so an iteration runs; "c2" is the best.
    suite = FakeSuite(lambda t: 0.05 if t == "c2" else 0.5)
    h = IterativeHumanizer(pipeline, suite=suite)  # type: ignore[arg-type]
    h.humanize_and_verify(
        "input",
        target_ai_score=0.1,
        max_iterations=1,
        candidates_per_iteration=3,
        seed=42,
    )
    assert len(para.calls) == 1
    assert para.calls[0]["n"] == 3
    # iter_seed = base_seed + 1 (first iteration).
    assert para.calls[0]["seed"] == 43


# ---------------------------------------------------------------------------
# Score trace and notes
# ---------------------------------------------------------------------------


def test_per_iteration_scores_includes_baseline() -> None:
    """The first entry of per_iteration_scores is always the baseline whole-text score."""
    h, _suite, _para = _fixed_score_humanizer(score=0.05)
    result = h.humanize_and_verify(
        "anything",
        target_ai_score=0.15,
    )
    assert result.per_iteration_scores
    assert result.per_iteration_scores[0] == pytest.approx(0.05)


def test_notes_record_iteration_decisions() -> None:
    para = FakeParaphrasePass(responses=[["good candidate"]])
    pipeline = FakePipeline(para)
    suite = FakeSuite(lambda t: 0.05 if "good candidate" in t else 0.5)
    h = IterativeHumanizer(pipeline, suite=suite)  # type: ignore[arg-type]
    result = h.humanize_and_verify(
        "Text.",
        target_ai_score=0.15,
        max_iterations=1,
        candidates_per_iteration=1,
    )
    notes_blob = "\n".join(result.notes)
    assert "baseline humanize" in notes_blob
    assert "iteration 0 score" in notes_blob


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_invalid_target_ai_score() -> None:
    h, _, _ = _fixed_score_humanizer(score=0.5)
    with pytest.raises(ValueError):
        h.humanize_and_verify("text", target_ai_score=1.5)
    with pytest.raises(ValueError):
        h.humanize_and_verify("text", target_ai_score=-0.1)


def test_rejects_zero_max_iterations() -> None:
    h, _, _ = _fixed_score_humanizer(score=0.5)
    with pytest.raises(ValueError):
        h.humanize_and_verify("text", max_iterations=0)


def test_rejects_zero_candidates_per_iteration() -> None:
    h, _, _ = _fixed_score_humanizer(score=0.5)
    with pytest.raises(ValueError):
        h.humanize_and_verify("text", candidates_per_iteration=0)


def test_empty_input_returns_empty_result() -> None:
    h, _, _ = _fixed_score_humanizer(score=0.5)
    result = h.humanize_and_verify("")
    assert result.text == ""
    assert result.iterations == 0
    assert result.target_reached is False


# ---------------------------------------------------------------------------
# Scoring unavailable (suite returns -1.0 / suite is None)
# ---------------------------------------------------------------------------


def test_returns_baseline_when_scoring_unavailable() -> None:
    """If BenchmarkSuite cannot be built, return the baseline humanization."""
    para = FakeParaphrasePass(responses=[])
    pipeline = FakePipeline(para, baseline_text="baseline output")
    # A suite that always reports -1.0 (the "unknown" sentinel).
    suite = FakeSuite(lambda _t: -1.0)
    # We have to bypass the real -1 detection: the iterative humanizer
    # checks for negative trusted scores via _score_whole, which calls
    # _extract_score, which returns -1.0 when trusted is None. Force that.
    class NullSuite:
        def score(self, _text: str) -> FakeReport:
            return FakeReport(trusted=None, raw=-1.0)

    h = IterativeHumanizer(pipeline, suite=NullSuite())  # type: ignore[arg-type]
    result = h.humanize_and_verify(
        "AI text",
        target_ai_score=0.15,
    )
    assert result.text == "baseline output"
    assert result.iterations == 0
    assert result.target_reached is False


# ---------------------------------------------------------------------------
# Loop refuses to accept a non-improving candidate
# ---------------------------------------------------------------------------


def test_does_not_swap_when_candidates_are_worse_than_original() -> None:
    """If every candidate scores higher than the worst paragraph, do not swap."""
    para = FakeParaphrasePass(
        responses=[["worse1", "worse2", "worse3"]],
    )
    pipeline = FakePipeline(para)

    def scorer(text: str) -> float:
        if text.startswith("worse"):
            return 0.99  # all candidates worse than original
        if text == "ORIGINAL":
            return 0.5  # the original paragraph score
        return 0.5

    suite = FakeSuite(scorer)
    h = IterativeHumanizer(pipeline, suite=suite)  # type: ignore[arg-type]
    result = h.humanize_and_verify(
        "ORIGINAL",
        target_ai_score=0.1,
        max_iterations=2,
        candidates_per_iteration=3,
    )
    assert result.text == "ORIGINAL"
    assert "no candidate improved" in "\n".join(result.notes)


# ---------------------------------------------------------------------------
# Performance: full mock-based suite runs fast
# ---------------------------------------------------------------------------


def test_mock_suite_runs_in_under_5_seconds() -> None:
    """The whole mock-based test suite must stay snappy (<5s aggregate)."""
    t0 = time.perf_counter()
    h, _, _ = _fixed_score_humanizer(score=0.05)
    for _ in range(50):
        h.humanize_and_verify("hello world", target_ai_score=0.15)
    elapsed = time.perf_counter() - t0
    # 50 calls in well under 1 second on any reasonable machine.
    assert elapsed < 5.0, f"50 mock-runs took {elapsed:.2f}s; too slow"


# ---------------------------------------------------------------------------
# Real integration: marked slow, opt-in
# ---------------------------------------------------------------------------


@pytest.mark.heavy
def test_real_pipeline_integration() -> None:
    """End-to-end integration with the real default pipeline.

    Marked ``heavy`` because it builds the real HumanizationPipeline. The
    BenchmarkSuite is still mocked here so we do not pay model-load cost
    in CI; the slow part is just the 9-pass run. Use ``pytest -m heavy``
    to opt in.
    """
    pipeline = HumanizationPipeline()

    # Stub out the suite with a constant low score so the loop exits at
    # iteration 0. This validates that the IterativeHumanizer plays nicely
    # with the real pipeline's pass list and config.
    suite = FakeSuite(lambda _t: 0.1)
    h = IterativeHumanizer(pipeline, suite=suite)  # type: ignore[arg-type]
    result = h.humanize_and_verify(
        "Furthermore, it is important to delve into the multifaceted "
        "implications of this approach. Subsequently, we shall navigate "
        "the complex landscape of considerations.",
        style="blog",
        target_ai_score=0.15,
    )
    assert isinstance(result, IterativeResult)
    assert result.iterations == 0
    assert result.target_reached is True
