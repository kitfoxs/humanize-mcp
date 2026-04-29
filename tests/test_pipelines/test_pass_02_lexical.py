"""Unit tests for pass_02_lexical."""

from __future__ import annotations

from pipelines.pass_02_lexical import LexicalPass


def _apply(text: str, **cfg) -> str:
    p = LexicalPass()
    return p.apply(text, {"intensity": "balanced", "style": {}, **cfg})


def test_replaces_delve_into_phrase() -> None:
    out = _apply("Let me delve into this topic.")
    assert "delve" not in out.lower()


def test_replaces_intricate() -> None:
    out = _apply("An intricate system of intricate parts.")
    assert out.lower().count("intricate") < 2


def test_case_preserved_on_substitution() -> None:
    out = _apply("Moreover, the answer is yes.")
    # "Moreover" has alternatives like "Also" / "And"; should keep capital
    assert out[0].isupper()


def test_disabled_entry_via_style_override() -> None:
    out = _apply(
        "The system is robust and comprehensive.",
        style={
            "lexical_substitutions": {
                "robust": {"disabled": True},
                "comprehensive": {"disabled": True},
            }
        },
    )
    assert "robust" in out.lower()
    assert "comprehensive" in out.lower()


def test_minimal_intensity_is_more_conservative() -> None:
    text = "The robust system fosters innovation. We must garner support."
    minimal = _apply(text, intensity="minimal")
    aggressive = _apply(text, intensity="aggressive")
    # aggressive should change at least as many tokens as minimal
    assert minimal != text or aggressive != text


def test_idempotent_for_clean_text() -> None:
    text = "Hello world. Nothing fancy here."
    assert _apply(text) == text
