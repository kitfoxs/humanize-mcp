"""Unit tests for pass_07_punctuation."""

from __future__ import annotations

from pipelines.pass_07_punctuation import PunctuationPass


def _apply(text: str, **cfg) -> str:
    p = PunctuationPass()
    return p.apply(text, {"intensity": "balanced", "style": {}, **cfg})


def test_strip_oxford_comma_when_configured() -> None:
    text = "Apples, bananas, and oranges are fruits."
    out = _apply(text, style={"preserve_oxford_comma": False})
    assert "bananas, and" not in out
    assert "bananas and" in out


def test_keep_oxford_comma_when_configured() -> None:
    text = "Apples, bananas, and oranges are fruits."
    out = _apply(text, style={"preserve_oxford_comma": True})
    assert "bananas, and oranges" in out


def test_inconsistent_oxford_alternates() -> None:
    text = (
        "A, B, and C. D, E, and F. G, H, and I. J, K, and L."
    )
    out = _apply(text, style={"preserve_oxford_comma": None})
    # at least one "X, and Y" gets rewritten to "X and Y"
    assert " and " in out
    # but not all of them
    assert "and" in out


def test_no_change_when_no_lists() -> None:
    text = "A simple sentence with no list of three."
    out = _apply(text, style={"preserve_oxford_comma": False})
    assert out == text
