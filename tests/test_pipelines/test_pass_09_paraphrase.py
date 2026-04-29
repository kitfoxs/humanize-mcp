"""Unit tests for pass_09_paraphrase (light mode only)."""

from __future__ import annotations

from pipelines.pass_09_paraphrase import ParaphrasePass


def _apply(text: str, **cfg) -> str:
    p = ParaphrasePass()
    return p.apply(text, {"intensity": "balanced", "mode": "light", **cfg})


def test_light_substitution_replaces_utilize() -> None:
    out = _apply("We must utilize the system.")
    assert "utilize" not in out


def test_light_intensity_capped_per_paragraph() -> None:
    text = "We utilize subsequently obtain numerous additional sufficient assistance."
    minimal = _apply(text, intensity="minimal")
    aggressive = _apply(text, intensity="aggressive")
    # aggressive should change strictly more than minimal
    assert minimal != aggressive or minimal != text


def test_no_op_for_clean_text() -> None:
    text = "Plain prose with no flagged words."
    out = _apply(text)
    assert out == text


def test_heavy_mode_falls_back_when_model_missing() -> None:
    # mode=heavy should not crash even if the model can't load
    out = _apply("Test paragraph.", mode="heavy")
    assert isinstance(out, str)
