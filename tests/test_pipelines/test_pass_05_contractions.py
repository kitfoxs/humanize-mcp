"""Unit tests for pass_05_contractions."""

from __future__ import annotations

from pipelines.pass_05_contractions import ContractionsPass


def _apply(text: str, **cfg) -> str:
    p = ContractionsPass()
    return p.apply(text, {"intensity": "balanced", "style": {}, **cfg})


def test_aggressive_contracts_everything() -> None:
    text = "It is fine. They are happy. We do not know."
    out = _apply(text, intensity="aggressive", density=1.0)
    assert "It's" in out
    assert "They're" in out
    assert "don't" in out


def test_academic_register_contracts_sparingly() -> None:
    text = "It is fine. We do not know. They are not sure."
    out = _apply(text, target_register="academic")
    # at most one or two contractions
    contractions = sum(1 for w in ["It's", "don't", "aren't", "We're"] if w in out)
    assert contractions <= 2


def test_no_false_positives_on_yu_have() -> None:
    # we removed "you have" -> "you've" because it's wrong in many contexts
    text = "Let me know if you have questions."
    out = _apply(text, intensity="aggressive", density=1.0)
    assert "you've questions" not in out
    assert "you have questions" in out


def test_minimal_intensity_makes_few_changes() -> None:
    text = "It is. It is. It is. It is. It is. It is. It is. It is. It is. It is."
    out = _apply(text, intensity="minimal")
    n = out.count("It's")
    assert n <= 5  # fewer than half
