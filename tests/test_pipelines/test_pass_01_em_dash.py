"""Unit tests for pass_01_em_dash."""

from __future__ import annotations

from pipelines.pass_01_em_dash import EmDashPass


def _apply(text: str, **cfg) -> str:
    p = EmDashPass()
    return p.apply(text, {"intensity": "balanced", "style": {}, **cfg})


def test_strips_singleton_em_dash() -> None:
    out = _apply("This is fine \u2014 and that is also fine.")
    assert "\u2014" not in out
    assert "fine" in out and "that" in out


def test_replaces_paired_em_dash_with_commas_or_parens() -> None:
    out = _apply(
        "The model \u2014 trained on books and code \u2014 produced fluent text."
    )
    assert "\u2014" not in out
    assert "trained on books" in out


def test_preserve_em_dashes_when_configured() -> None:
    text = "She paused \u2014 then spoke."
    out = _apply(text, style={"preserve_em_dashes": True})
    assert out == text


def test_idempotent() -> None:
    text = "First clause \u2014 second clause \u2014 third clause."
    once = _apply(text)
    twice = _apply(once)
    assert once == twice


def test_no_em_dash_in_text_is_noop() -> None:
    text = "No special punctuation here, just commas."
    assert _apply(text) == text


def test_changes_logged() -> None:
    p = EmDashPass()
    p.apply("A \u2014 B \u2014 C.", {"intensity": "balanced", "style": {}})
    assert any(c["kind"].startswith("em_dash") for c in p.changes())
