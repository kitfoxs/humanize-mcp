"""Tests for benchmark.detectors.HeuristicDetector and helper functions."""

from __future__ import annotations

import pytest

from benchmark.detectors import (
    HeuristicDetector,
    burstiness,
    count_tells,
    lexical_diversity,
)


def test_burstiness_zero_for_short_text() -> None:
    assert burstiness("Hello.") == 0.0
    assert burstiness("") == 0.0


def test_burstiness_higher_for_varied_lengths() -> None:
    uniform = (
        "I went to the store. I bought some food. I came back home. "
        "I ate the food. I went to bed."
    )
    varied = (
        "Yes. After a long pause that surprised even me, I finally "
        "agreed to come along, though I had no real interest in seeing "
        "the place again. Hm. We left."
    )
    assert burstiness(varied) > burstiness(uniform)


def test_lexical_diversity_in_unit_interval() -> None:
    text = "The cat sat on the mat. The dog sat on the rug."
    div = lexical_diversity(text)
    assert 0.0 <= div <= 1.0


def test_count_tells_picks_up_known_phrases() -> None:
    text = (
        "It is important to note that we should delve into the realm of "
        "modern engineering. Furthermore, we must leverage these tools."
    )
    n, hits = count_tells(text)
    assert n >= 3
    assert any("delve" in h for h in hits)
    assert any("important to note" in h for h in hits)


def test_count_tells_em_dash_threshold() -> None:
    no_dash = "Hello world. Goodbye world."
    one_dash = "Hello world \u2014 goodbye."
    many_dashes = "A \u2014 B \u2014 C \u2014 D."
    assert count_tells(no_dash)[0] == 0
    # Single em dash should not be flagged (legit punctuation).
    assert all(not h.startswith("em_dash") for h in count_tells(one_dash)[1])
    assert any(h.startswith("em_dash") for h in count_tells(many_dashes)[1])


def test_heuristic_detector_returns_valid_score() -> None:
    det = HeuristicDetector()
    score = det.score("Hello world. This is a short test sentence.")
    assert 0.0 <= score.ai_probability <= 1.0
    assert 0.0 <= score.human_probability <= 1.0
    assert pytest.approx(score.ai_probability + score.human_probability, abs=1e-6) == 1.0
    assert score.detector == "heuristic"
    assert score.tier == "heuristic"
    assert score.error == ""


def test_heuristic_detector_pushes_obvious_ai_text_higher() -> None:
    det = HeuristicDetector()
    obvious_ai = (
        "It is important to note that in today's fast-paced world, we must "
        "delve into the realm of modern engineering. Furthermore, we should "
        "leverage these robust tools to navigate the complexities of the "
        "ever-evolving landscape. In conclusion, this approach is essential."
    )
    casual_human = (
        "lol same here, my back hurts so bad after sitting all day. "
        "tried that posture corrector thing, kinda helps. annoying tho."
    )
    ai_score = det.score(obvious_ai)
    human_score = det.score(casual_human)
    assert ai_score.ai_probability > human_score.ai_probability


def test_heuristic_detector_handles_empty_text() -> None:
    det = HeuristicDetector()
    score = det.score("")
    assert score.error != ""
    assert score.verdict == "error"


def test_heuristic_detector_records_details() -> None:
    det = HeuristicDetector()
    score = det.score(
        "It is important to note that this is a robust solution. "
        "Furthermore, we should delve into the details. In conclusion, "
        "this approach is multifaceted."
    )
    assert "burstiness" in score.details
    assert "tells_count" in score.details
    assert score.details["tells_count"] >= 1
