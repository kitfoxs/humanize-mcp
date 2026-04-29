"""Unit tests for pass_04_rhythm (v0.2.0 rewrite, Bet 2)."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List

import pytest

from pipelines.pass_04_rhythm import (
    FRAGMENT_POOLS,
    RhythmPass,
    sentence_split,
    _fragments_for_style,
)


def _apply(text: str, **cfg: Any) -> str:
    p = RhythmPass()
    config: Dict[str, Any] = {"intensity": "balanced", "style": {}, "seed": 42}
    config.update(cfg)
    return p.apply(text, config)


def _cv(text: str) -> float:
    lens = [len(s.split()) for s in sentence_split(text)]
    if len(lens) < 2:
        return 0.0
    return statistics.pstdev(lens) / statistics.mean(lens)


def _changes(p: RhythmPass) -> List[str]:
    return [c["kind"] for c in p.changes()]


# ----- existing v0.1 invariants (kept for regression coverage) -----


def test_uniform_text_gets_more_bursty() -> None:
    text = (
        "The model produces text. The model accepts inputs. The model "
        "outputs predictions. The model uses parameters. The model trains "
        "on data. The model needs compute. The model deploys to servers."
    )
    out = _apply(text)
    assert _cv(out) >= _cv(text) - 0.01


def test_short_text_passes_through() -> None:
    # Two sentences, below the 3-sentence floor, no work attempted.
    text = "Short. Text here."
    assert _apply(text) == text


def test_already_bursty_input_not_destroyed() -> None:
    # Mixed lengths from the v0.1 test, expanded to 4 sentences so the
    # new pass actually inspects it. It should not collapse the variance.
    text = (
        "Yes. The story is more complicated than you think, and that "
        "complication is exactly the point of why this whole field is so "
        "exciting right now. Maybe. So this is fine."
    )
    out = _apply(text)
    in_cv = _cv(text)
    out_cv = _cv(out)
    # Polish round may run, but our guard prevents net regression.
    assert out_cv >= in_cv - 0.05


# ----- v0.2 new behaviour (the Bet 2 contract) -----


# A 17-word sentence with a comma + conjunction so split-points exist.
_UNIFORM_SENT = (
    "The model accepts inputs from users, then it processes them "
    "carefully and returns consistent results every time."
)
_UNIFORM_TEXT = " ".join([_UNIFORM_SENT] * 15)


def test_uniform_input_balanced_hits_target() -> None:
    """15 uniform sentences should land at CV >= 0.6 at balanced."""
    out = _apply(_UNIFORM_TEXT, intensity="balanced", style={"name": "blog"})
    assert _cv(out) >= 0.6, f"balanced cv was {_cv(out):.3f}"


def test_uniform_input_aggressive_hits_target() -> None:
    """15 uniform sentences should land at CV >= 0.8 at aggressive."""
    out = _apply(_UNIFORM_TEXT, intensity="aggressive", style={"name": "blog"})
    assert _cv(out) >= 0.8, f"aggressive cv was {_cv(out):.3f}"


def test_aggressive_inserts_fragments_on_uniform_input() -> None:
    """Fragments must actually be inserted at aggressive on uniform input."""
    p = RhythmPass()
    p.apply(
        _UNIFORM_TEXT,
        {"intensity": "aggressive", "style": {"name": "blog"}, "seed": 42},
    )
    kinds = _changes(p)
    assert any(k == "rhythm_fragment_insert" for k in kinds), kinds


def test_paragraph_preservation() -> None:
    """Multi-paragraph input keeps its paragraph count."""
    para = " ".join([_UNIFORM_SENT] * 6)
    text = f"{para}\n\n{para}\n\n{para}"
    out = _apply(text, intensity="aggressive", style={"name": "blog"})
    in_count = len([p for p in text.split("\n\n") if p.strip()])
    out_count = len([p for p in out.split("\n\n") if p.strip()])
    assert in_count == out_count == 3


def test_style_specific_fragments_reddit() -> None:
    """Reddit pool includes 'Lol.' / 'Yeah.' that the academic pool lacks."""
    p = RhythmPass()
    p.apply(
        _UNIFORM_TEXT,
        {"intensity": "aggressive", "style": {"name": "reddit"}, "seed": 42},
    )
    inserted = [c["after"] for c in p.changes() if c["kind"] == "rhythm_fragment_insert"]
    assert inserted, "expected at least one fragment insertion"
    reddit_pool = set(FRAGMENT_POOLS["reddit"])
    assert all(frag in reddit_pool for frag in inserted), inserted
    # And it must not pull from the academic pool.
    academic_only = set(FRAGMENT_POOLS["academic_human"]) - reddit_pool
    assert not any(frag in academic_only for frag in inserted)


def test_style_specific_fragments_academic() -> None:
    p = RhythmPass()
    p.apply(
        _UNIFORM_TEXT,
        {"intensity": "aggressive", "style": {"name": "academic_human"}, "seed": 42},
    )
    inserted = [c["after"] for c in p.changes() if c["kind"] == "rhythm_fragment_insert"]
    assert inserted, "expected at least one fragment insertion"
    pool = set(FRAGMENT_POOLS["academic_human"])
    assert all(frag in pool for frag in inserted), inserted
    # No casual leakage.
    assert not any(frag == "Lol." for frag in inserted)


def test_style_specific_fragments_book_chapter() -> None:
    p = RhythmPass()
    p.apply(
        _UNIFORM_TEXT,
        {"intensity": "aggressive", "style": {"name": "book_chapter"}, "seed": 42},
    )
    inserted = [c["after"] for c in p.changes() if c["kind"] == "rhythm_fragment_insert"]
    pool = set(FRAGMENT_POOLS["book_chapter"])
    assert inserted and all(frag in pool for frag in inserted), inserted


def test_unknown_style_falls_back_to_blog_pool() -> None:
    assert _fragments_for_style("nonexistent_style") == FRAGMENT_POOLS["blog"]


def test_already_bursty_at_aggressive_pushed_higher() -> None:
    """A passably-bursty paragraph (CV ~0.5) at aggressive should still
    be pushed higher by the polish round (or at least not lowered).
    """
    text = (
        "Maybe. The model accepts inputs from users, then it processes "
        "them carefully and returns consistent results every time. Yeah. "
        "Sometimes the system surprises you with a deeply long answer "
        "that wanders through several ideas before finally landing on a "
        "useful conclusion that you can act on. Right. It works."
    )
    in_cv = _cv(text)
    out = _apply(text, intensity="aggressive", style={"name": "blog"})
    out_cv = _cv(out)
    assert out_cv >= in_cv - 0.02


def test_target_cv_override_via_style() -> None:
    """Style overrides via pass_configs.pass_04_rhythm.target_cv take effect."""
    style = {
        "name": "blog",
        "pass_configs": {"pass_04_rhythm": {"target_cv": 0.95}},
    }
    p1 = RhythmPass()
    out_high = p1.apply(
        _UNIFORM_TEXT,
        {"intensity": "balanced", "style": style, "seed": 42},
    )
    p2 = RhythmPass()
    out_default = p2.apply(
        _UNIFORM_TEXT,
        {"intensity": "balanced", "style": {"name": "blog"}, "seed": 42},
    )
    # With a higher target the pass should keep iterating and produce
    # at least as high a CV as the default.
    assert _cv(out_high) >= _cv(out_default) - 0.01


def test_deterministic_with_seed() -> None:
    """Same input + same seed -> identical output."""
    a = _apply(_UNIFORM_TEXT, intensity="aggressive", style={"name": "reddit"}, seed=99)
    b = _apply(_UNIFORM_TEXT, intensity="aggressive", style={"name": "reddit"}, seed=99)
    assert a == b


def test_different_seeds_produce_different_outputs() -> None:
    """Same input, different seeds -> different fragment / connector picks."""
    a = _apply(_UNIFORM_TEXT, intensity="aggressive", style={"name": "blog"}, seed=1)
    b = _apply(_UNIFORM_TEXT, intensity="aggressive", style={"name": "blog"}, seed=2)
    assert a != b


def test_long_sentence_split_lower_threshold() -> None:
    """A 17-word sentence with a comma+conjunction should now be split
    (v0.1 required >= 22 words; v0.2 requires >= 16).
    """
    text = (
        "The model accepts inputs from users, and it processes them "
        "carefully every time."  # 13 words actually; let's craft 17.
    )
    sentence = (
        "The robust model accepts diverse inputs from many users, and "
        "it processes them carefully and returns consistent results "
        "every single time."
    )
    text = " ".join([sentence] * 4)
    p = RhythmPass()
    p.apply(text, {"intensity": "balanced", "style": {"name": "blog"}, "seed": 42})
    kinds = _changes(p)
    assert any(k == "rhythm_split" for k in kinds), kinds


def test_minimal_intensity_does_not_insert_fragments() -> None:
    """At minimal intensity we don't run variance pumps."""
    p = RhythmPass()
    p.apply(
        _UNIFORM_TEXT,
        {"intensity": "minimal", "style": {"name": "blog"}, "seed": 42},
    )
    kinds = _changes(p)
    assert not any(k == "rhythm_fragment_insert" for k in kinds)
    assert not any(k == "rhythm_long_extend" for k in kinds)


def test_paragraph_count_preserved_with_mixed_paragraphs() -> None:
    """A mix of short and long paragraphs maintains its structure."""
    short = "Short paragraph. With three sentences. That stay together."
    long = " ".join([_UNIFORM_SENT] * 8)
    text = f"{short}\n\n{long}\n\n{short}"
    out = _apply(text, intensity="aggressive", style={"name": "blog"})
    assert out.count("\n\n") == text.count("\n\n")
