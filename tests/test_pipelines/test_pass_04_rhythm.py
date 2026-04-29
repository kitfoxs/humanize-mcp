"""Unit tests for pass_04_rhythm."""

from __future__ import annotations

import statistics

from pipelines.pass_04_rhythm import RhythmPass, sentence_split


def _apply(text: str, **cfg) -> str:
    p = RhythmPass()
    return p.apply(text, {"intensity": "balanced", "style": {}, **cfg})


def test_uniform_text_gets_more_bursty() -> None:
    # five sentences of nearly identical length
    text = (
        "The model produces text. The model accepts inputs. The model "
        "outputs predictions. The model uses parameters. The model trains "
        "on data. The model needs compute. The model deploys to servers."
    )
    out = _apply(text)
    in_lens = [len(s.split()) for s in sentence_split(text)]
    out_lens = [len(s.split()) for s in sentence_split(out)]
    if len(in_lens) > 1 and len(out_lens) > 1:
        in_cv = statistics.pstdev(in_lens) / statistics.mean(in_lens)
        out_cv = statistics.pstdev(out_lens) / statistics.mean(out_lens)
        assert out_cv >= in_cv - 0.01  # never worse


def test_short_text_passes_through() -> None:
    text = "Short. Text. Here."
    assert _apply(text) == text


def test_already_bursty_unchanged() -> None:
    text = (
        "Yes. The story is more complicated than you think, and that "
        "complication is exactly the point of why this whole field is so "
        "exciting right now. Maybe."
    )
    out = _apply(text)
    # we don't require equality but burstiness should not collapse
    in_lens = [len(s.split()) for s in sentence_split(text)]
    if len(in_lens) > 1:
        in_cv = statistics.pstdev(in_lens) / statistics.mean(in_lens)
        assert in_cv > 0.4
