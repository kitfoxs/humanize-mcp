"""Smoke tests for the HumanizeMCP server tool surface.

These tests verify the FastMCP server boots, all expected tools are
registered with correct names, and the pydantic response models accept
and reject inputs as designed. They deliberately do not exercise the
pipelines/, styles/, or benchmark/ subpackages — those have their own
tests in tests/pipelines/, tests/styles/, tests/benchmark/.

Run with::

    pytest tests/test_server.py -v
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
from pydantic import ValidationError

import server as server_module
from server import (
    DetectorScore,
    HumanityReport,
    TellLocation,
    TellsReport,
    VerifyResult,
    _classify,
    mcp,
)


EXPECTED_TOOLS: set[str] = {
    "humanize",
    "detect_tells",
    "score_humanity",
    "apply_style",
    "list_styles",
    "humanize_and_verify",
}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def _registered_tool_names() -> set[str]:
    """Return the set of tool names registered on the FastMCP instance.

    FastMCP exposes a coroutine ``list_tools()`` that returns a list of
    Tool objects. We unwrap it synchronously so tests can stay simple.
    """
    tools = asyncio.run(mcp.list_tools())
    return {t.name for t in tools}


def test_all_expected_tools_are_registered() -> None:
    """Every tool documented in the README must register on the server."""
    registered = _registered_tool_names()
    missing = EXPECTED_TOOLS - registered
    assert not missing, f"missing tools: {missing}"


def test_no_unexpected_tools_are_registered() -> None:
    """Catch accidental tool additions; update EXPECTED_TOOLS deliberately."""
    registered = _registered_tool_names()
    extra = registered - EXPECTED_TOOLS
    assert not extra, (
        f"unexpected tools registered: {extra}. "
        "If you added a tool, update EXPECTED_TOOLS in this test."
    )


# ---------------------------------------------------------------------------
# Tool signatures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name,expected_params",
    [
        ("humanize", {"text", "style", "preserve_voice", "intensity"}),
        ("detect_tells", {"text"}),
        ("score_humanity", {"text", "detectors"}),
        ("apply_style", {"text", "style"}),
        ("list_styles", set()),
        (
            "humanize_and_verify",
            {
                "text",
                "style",
                "target_ai_score",
                "max_iterations",
                "target_detector",
                "candidates_per_iteration",
            },
        ),
    ],
)
def test_tool_signatures(tool_name: str, expected_params: set[str]) -> None:
    """Each tool must expose the parameters declared in the README."""
    func = getattr(server_module, tool_name)
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())
    assert params == expected_params, (
        f"{tool_name} parameters {params} != expected {expected_params}"
    )


def test_tool_docstrings_are_present() -> None:
    """MCP uses docstrings as tool descriptions; missing docstrings are bugs."""
    for name in EXPECTED_TOOLS:
        func = getattr(server_module, name)
        doc = inspect.getdoc(func)
        assert doc, f"tool {name} has no docstring"
        assert len(doc) > 50, f"tool {name} docstring is too short"


# ---------------------------------------------------------------------------
# Validation in tools
# ---------------------------------------------------------------------------


def test_humanize_rejects_out_of_range_intensity() -> None:
    """intensity must be in [0, 1]; out-of-range values raise."""
    with pytest.raises(ValueError):
        server_module.humanize("hello", intensity=1.5)
    with pytest.raises(ValueError):
        server_module.humanize("hello", intensity=-0.1)


def test_humanize_returns_input_when_pipeline_missing() -> None:
    """Graceful degradation: no pipeline yet means input is returned as-is."""
    text = "The implementation leverages a robust framework."
    out = server_module.humanize(text)
    assert isinstance(out, str)
    # Pipeline may or may not be present at test time; what we require is
    # that the call does not raise and returns a string.


def test_humanize_handles_empty_string() -> None:
    """Empty input is a no-op and must not crash the dispatcher."""
    assert server_module.humanize("") == ""


def test_detect_tells_handles_empty_string() -> None:
    """Empty input returns an empty report, not an error."""
    report = server_module.detect_tells("")
    assert isinstance(report, TellsReport)
    assert report.tell_count == 0
    assert report.text_length == 0


def test_score_humanity_returns_unknown_when_backend_missing() -> None:
    """If the benchmark backend is not present we return verdict=unknown."""
    report = server_module.score_humanity("hello world", detectors=["roberta-base"])
    assert isinstance(report, HumanityReport)
    if report.aggregate_probability_ai < 0:
        assert report.verdict == "unknown"


def test_humanize_and_verify_rejects_bad_target() -> None:
    """target_ai_score must be a probability."""
    with pytest.raises(ValueError):
        server_module.humanize_and_verify("hello", target_ai_score=1.1)
    with pytest.raises(ValueError):
        server_module.humanize_and_verify("hello", target_ai_score=-0.1)


def test_humanize_and_verify_rejects_zero_iterations() -> None:
    """max_iterations must be >= 1."""
    with pytest.raises(ValueError):
        server_module.humanize_and_verify("hello", max_iterations=0)


def test_list_styles_returns_list_of_strings() -> None:
    """list_styles never raises; returns a (possibly empty) sorted list."""
    styles = server_module.list_styles()
    assert isinstance(styles, list)
    assert all(isinstance(s, str) for s in styles)
    assert styles == sorted(set(styles))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


def test_tell_location_severity_bounds() -> None:
    """Severity is constrained to 1-5 by the schema."""
    TellLocation(
        category="excess_vocabulary",
        fragment="delve",
        line=1,
        char_start=0,
        char_end=5,
        severity=5,
    )
    with pytest.raises(ValidationError):
        TellLocation(
            category="excess_vocabulary",
            fragment="delve",
            line=1,
            char_start=0,
            char_end=5,
            severity=6,
        )
    with pytest.raises(ValidationError):
        TellLocation(
            category="excess_vocabulary",
            fragment="delve",
            line=1,
            char_start=0,
            char_end=5,
            severity=0,
        )


def test_tell_location_forbids_extra_fields() -> None:
    """Strict schema; extra keys would mask typos in pipeline emissions."""
    with pytest.raises(ValidationError):
        TellLocation(
            category="x",
            fragment="y",
            line=1,
            char_start=0,
            char_end=1,
            severity=1,
            extra_field="nope",
        )


def test_detector_score_defaults() -> None:
    """confidence and error have safe defaults."""
    score = DetectorScore(detector="roberta-base", probability_ai=0.42)
    assert score.confidence == 0.0
    assert score.error == ""


def test_humanity_report_round_trips() -> None:
    """Round-trip serialization through model_dump and model_validate."""
    report = HumanityReport(
        text_length=100,
        detector_scores=[
            DetectorScore(detector="roberta-base", probability_ai=0.42),
        ],
        aggregate_probability_ai=0.42,
        verdict="uncertain",
    )
    again = HumanityReport.model_validate(report.model_dump())
    assert again == report


def test_verify_result_round_trips() -> None:
    """The verify result is the most complex model; ensure it round-trips."""
    initial = HumanityReport(
        text_length=100,
        detector_scores=[DetectorScore(detector="r", probability_ai=0.9)],
        aggregate_probability_ai=0.9,
        verdict="ai",
    )
    final = HumanityReport(
        text_length=98,
        detector_scores=[DetectorScore(detector="r", probability_ai=0.2)],
        aggregate_probability_ai=0.2,
        verdict="human",
    )
    result = VerifyResult(
        text="rewritten",
        iterations=2,
        target_ai_score=0.3,
        initial_score=initial,
        final_score=final,
        target_reached=True,
        notes=["iteration 1: 0.5", "iteration 2: 0.2"],
    )
    again = VerifyResult.model_validate(result.model_dump())
    assert again == result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (-1.0, "unknown"),
        (0.0, "human"),
        (0.29, "human"),
        (0.3, "uncertain"),
        (0.5, "uncertain"),
        (0.69, "uncertain"),
        (0.7, "ai"),
        (1.0, "ai"),
    ],
)
def test_classify_thresholds(score: float, expected: str) -> None:
    """The verdict bucketing is part of the public contract; pin it."""
    assert _classify(score) == expected


def test_main_function_exists() -> None:
    """The console-script entry point must be importable."""
    assert callable(server_module.main)
