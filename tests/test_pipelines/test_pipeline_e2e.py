"""End-to-end pipeline tests."""

from __future__ import annotations

from pipelines import HumanizationPipeline, HumanizeConfig
from pipelines.tells_detector import detect_tells
from styles import list_styles


SAMPLE = (
    "Great question! Let me delve into this fascinating topic. "
    "It's important to note that AI systems leverage intricate neural "
    "networks \u2014 and they navigate a complex tapestry of data. Moreover, "
    "this is a paradigm shift that demonstrates robust performance across "
    "various domains.\n\n"
    "Furthermore, we should explore how these systems foster innovation, "
    "enable creativity, and accelerate discovery. Ultimately, a balanced "
    "approach is needed. Let me know if you have follow-up questions!"
)


def test_pipeline_runs_for_every_style() -> None:
    p = HumanizationPipeline()
    for style in list_styles():
        cfg = HumanizeConfig(style=style, intensity="balanced")
        result = p.run(SAMPLE, cfg)
        assert isinstance(result.text, str) and result.text
        assert result.processing_time_ms >= 0
        assert isinstance(result.passes_applied, list)


def test_pipeline_reduces_tells_count() -> None:
    p = HumanizationPipeline()
    cfg = HumanizeConfig(style="blog", intensity="aggressive")
    before = len(detect_tells(SAMPLE))
    result = p.run(SAMPLE, cfg)
    after = len(detect_tells(result.text))
    assert after < before, f"tells should decrease: {before} -> {after}"


def test_pipeline_strips_em_dashes_unless_preserved() -> None:
    p = HumanizationPipeline()
    cfg = HumanizeConfig(style="blog", intensity="balanced")
    result = p.run(SAMPLE, cfg)
    assert "\u2014" not in result.text


def test_pipeline_preserves_em_dashes_in_creative_fiction() -> None:
    p = HumanizationPipeline()
    cfg = HumanizeConfig(style="creative_fiction", intensity="balanced")
    result = p.run(SAMPLE, cfg)
    assert "\u2014" in result.text


def test_skip_passes_works() -> None:
    p = HumanizationPipeline()
    cfg = HumanizeConfig(style="blog", skip_passes=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    result = p.run(SAMPLE, cfg)
    assert result.text == SAMPLE
    assert result.tells_removed_count == 0


def test_per_pass_log_returned() -> None:
    p = HumanizationPipeline()
    cfg = HumanizeConfig(style="blog", return_per_pass_log=True)
    result = p.run(SAMPLE, cfg)
    assert result.pass_log
    assert all("pass_id" in entry for entry in result.pass_log)


def test_deterministic_with_same_seed() -> None:
    p = HumanizationPipeline()
    cfg = HumanizeConfig(style="blog", intensity="balanced", seed=99)
    a = p.run(SAMPLE, cfg).text
    b = p.run(SAMPLE, cfg).text
    assert a == b


def test_idempotent_on_clean_text() -> None:
    text = "A short clean paragraph with no AI fingerprints."
    p = HumanizationPipeline()
    cfg = HumanizeConfig(style="blog", intensity="balanced")
    once = p.run(text, cfg).text
    twice = p.run(once, cfg).text
    # should converge quickly; second pass changes little
    assert len(twice) >= len(once) - 5
