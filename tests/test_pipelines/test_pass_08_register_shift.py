"""Unit tests for pass_08_register_shift."""

from __future__ import annotations

from pipelines.pass_08_register_shift import RegisterShiftPass


def _apply(text: str, **cfg) -> str:
    p = RegisterShiftPass()
    return p.apply(text, {"intensity": "balanced", "style": {}, **cfg})


def test_academic_softens_certainty() -> None:
    text = "This shows that the model works."
    out = _apply(text, target_register="academic")
    assert "appear to show" in out or "shows" not in out


def test_casual_swaps_in_casual_words() -> None:
    text = "We must utilize this approach."
    out = _apply(text, target_register="casual")
    assert "utilize" not in out


def test_formal_decontracts() -> None:
    text = "I don't know and we can't tell."
    out = _apply(text, target_register="formal")
    assert "do not" in out


def test_neutral_is_noop() -> None:
    text = "Some text in a neutral register."
    assert _apply(text, target_register="neutral") == text
