"""Unit tests for pass_03_structural."""

from __future__ import annotations

from pipelines.pass_03_structural import StructuralPass


def _apply(text: str, **cfg) -> str:
    p = StructuralPass()
    return p.apply(text, {"intensity": "balanced", "style": {}, **cfg})


def test_rewrites_not_x_but_y() -> None:
    text = "It's not just a feature, it's a paradigm shift."
    out = _apply(text)
    assert "not just" not in out.lower()


def test_rewrites_not_just_but() -> None:
    text = "We need not just speed, but accuracy."
    out = _apply(text)
    assert "not just" not in out.lower()


def test_dampens_stacked_tricolons() -> None:
    text = (
        "It empowers users, supports developers, and accelerates innovation. "
        "It is fast, reliable, and secure. "
        "It builds trust, drives growth, and unlocks value."
    )
    out = _apply(text)
    # at least one of the three tricolons should be broken up
    assert " and Also " in out or out != text


def test_aggressive_dampens_copular_template() -> None:
    text = "Privacy is a cornerstone of digital trust."
    out = _apply(text, intensity="aggressive")
    assert "cornerstone" not in out


def test_idempotent_when_no_pattern() -> None:
    text = "A simple sentence with no flagged structures."
    assert _apply(text) == text
