"""Unit tests for pass_06_voice_injection."""

from __future__ import annotations

from pipelines.pass_06_voice_injection import VoiceInjectionPass


def _apply(text: str, **cfg) -> str:
    p = VoiceInjectionPass()
    return p.apply(text, {"intensity": "balanced", "style": {}, **cfg})


def test_no_filler_when_style_is_empty() -> None:
    text = "First sentence. Second sentence. Third sentence."
    out = _apply(text, style={"allowed_filler": [], "allowed_openers": []})
    assert out == text


def test_injects_when_filler_available() -> None:
    text = (
        "First sentence here. Second one is a bit longer though. "
        "And here is a third one. A fourth one to round it out. "
        "And one more for good measure."
    )
    out = _apply(
        text,
        target_register="casual",
        style={
            "allowed_filler": ["honestly", "I mean"],
            "allowed_openers": ["honestly,"],
        },
    )
    # at least one filler appears
    assert "honestly" in out.lower() or "I mean" in out


def test_first_sentence_preserved() -> None:
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    out = _apply(
        text,
        style={
            "allowed_filler": ["honestly"],
            "allowed_openers": ["honestly,"],
        },
    )
    # we explicitly skip index 0
    assert out.startswith("First sentence")
