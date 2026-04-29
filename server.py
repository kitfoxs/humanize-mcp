"""HumanizeMCP — FastMCP server exposing the humanization tool surface.

This module is the public MCP surface for the project. It defines the tool
contracts that MCP-aware clients (Claude Code, Copilot CLI, Continue, Zed,
Cursor, etc.) call. The actual humanization passes live in ``pipelines/``
and the detector adapters live in ``benchmark/``. This file deliberately
contains no humanization logic of its own; it dispatches.

The dispatch is intentionally lazy and forgiving: pipelines and detectors
are imported at call time, and a missing module is logged as a warning
rather than raising. This lets the server boot and respond to clients
during early development when not every component is present.

References
----------
The pipeline pass design follows ``research/06_implementation_recommendations.md``,
section 2 (the nine recommended passes). The tool surface is shaped by the
preset and intensity vocabulary defined in section 1 of the same document.
The detector list is informed by ``research/01_detector_landscape.md``.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("humanize_mcp.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# Make sibling packages importable without requiring an editable install.
# The pipelines/, styles/, and benchmark/ directories live next to this file.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


mcp: FastMCP = FastMCP(
    name="humanize-mcp",
    instructions=(
        "HumanizeMCP rewrites AI-generated prose so it reads as human-authored, "
        "with optional preservation of neurodivergent, ESL, and academic voice. "
        "Use humanize() for most cases. Use detect_tells() for diagnostics. Use "
        "score_humanity() to evaluate text against open detectors. Use "
        "humanize_and_verify() when a numeric AI-score target matters."
    ),
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class TellLocation(BaseModel):
    """A single AI tell located in the input text.

    Attributes
    ----------
    category : str
        Coarse class of the tell, e.g. ``"em_dash"``, ``"excess_vocabulary"``,
        ``"discourse_marker"``, ``"copular_template"``, ``"preamble"``,
        ``"closing_offer"``, ``"parallel_structure"``.
    fragment : str
        The exact substring that triggered the rule.
    line : int
        1-indexed line number in the input.
    char_start : int
        0-indexed character offset (inclusive).
    char_end : int
        0-indexed character offset (exclusive).
    severity : int
        Star rating 1 to 5, mirroring ``research/02_ai_tells_catalog.md``.
        5 is a near-certain AI signature, 1 is a weak co-occurring signal.
    suggestion : str
        Optional human-readable replacement hint.
    """

    model_config = ConfigDict(extra="forbid")

    category: str
    fragment: str
    line: int
    char_start: int
    char_end: int
    severity: int = Field(ge=1, le=5)
    suggestion: str = ""


class TellsReport(BaseModel):
    """The full output of :func:`detect_tells`."""

    model_config = ConfigDict(extra="forbid")

    text_length: int
    tell_count: int
    tells: list[TellLocation]
    summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of tells per category.",
    )


class DetectorScore(BaseModel):
    """One detector's verdict on a single text.

    Attributes
    ----------
    detector : str
        Adapter name, e.g. ``"roberta-base"``, ``"fast_detect_gpt"``,
        ``"binoculars"``, ``"stylometric_xgboost"``.
    probability_ai : float
        Probability the text is AI-generated, in [0, 1]. Higher is "more AI".
    confidence : float
        Detector self-reported confidence in [0, 1]. Some detectors do not
        expose this; in that case it equals the larger of probability_ai
        and (1 - probability_ai).
    error : str
        Empty on success. On failure, contains a short reason; the score
        will be NaN-equivalent (-1.0) and should be excluded from aggregates.
    """

    model_config = ConfigDict(extra="forbid")

    detector: str
    probability_ai: float
    confidence: float = 0.0
    error: str = ""


class HumanityReport(BaseModel):
    """The full output of :func:`score_humanity`."""

    model_config = ConfigDict(extra="forbid")

    text_length: int
    detector_scores: list[DetectorScore]
    aggregate_probability_ai: float = Field(
        description=(
            "Mean of successful detector probability_ai scores. "
            "Equals -1.0 if no detector returned a score."
        )
    )
    verdict: str = Field(
        description='One of "human", "uncertain", "ai", or "unknown".'
    )


class VerifyResult(BaseModel):
    """The full output of :func:`humanize_and_verify`.

    v0.2.0: extended with the per-iteration trace fields produced by the
    detector-guided iterative loop (``IterativeHumanizer``). The original
    fields remain to keep existing clients working; the new fields are
    optional with safe defaults so older clients can ignore them.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    iterations: int
    target_ai_score: float
    initial_score: HumanityReport
    final_score: HumanityReport
    target_reached: bool
    notes: list[str] = Field(default_factory=list)

    # v0.2.0 additions: surface of the Cheng et al. 2025 loop.
    target_detector: str = Field(
        default="trusted_mean",
        description=(
            "Which detector field the loop optimized against. One of "
            '"trusted_mean", "raw_mean", or a specific detector name.'
        ),
    )
    candidates_per_iteration: int = Field(
        default=1,
        description=(
            "How many stochastic paraphrase candidates were generated per "
            "iteration. 1 reduces the loop to deterministic paraphrasing."
        ),
    )
    per_iteration_scores: list[float] = Field(
        default_factory=list,
        description=(
            "Whole-text AI probability after iteration 0 (baseline) then "
            "after each accepted improvement. Empty if scoring was unavailable."
        ),
    )
    total_time_ms: int = Field(
        default=0,
        description="Total wall-clock time including baseline humanization.",
    )


# ---------------------------------------------------------------------------
# Lazy dispatch helpers
# ---------------------------------------------------------------------------


def _try_import(module_path: str) -> Any | None:
    """Import a module by dotted path, returning None and logging if missing.

    The server is designed to boot even when sibling packages have not yet
    been written. Each tool function calls this helper and degrades to a
    documented stub response when the underlying module is absent.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        logger.warning("optional module %s unavailable: %s", module_path, exc)
        return None
    except Exception as exc:
        logger.exception("error importing %s: %s", module_path, exc)
        return None


def _intensity_to_label(intensity: float) -> str:
    """Map a continuous intensity in [0, 1] to the categorical label used
    by ``pipelines.HumanizeConfig`` (``minimal`` / ``balanced`` / ``aggressive``).

    Boundaries follow ``research/06_implementation_recommendations.md``
    section 1: minimal under 1/3, balanced through 2/3, aggressive above.
    """
    if intensity < 1 / 3:
        return "minimal"
    if intensity < 2 / 3:
        return "balanced"
    return "aggressive"


def _normalize_tell(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate the pipelines.tells_detector schema into the TellLocation schema.

    The pipeline emits ``{line, char_offset, end_offset, tell_type,
    severity, matched_text, suggestion}``; the MCP surface expects
    ``{line, char_start, char_end, category, fragment, severity,
    suggestion}``. This shim isolates the two schemas so neither side has
    to compromise. If the input already matches the target schema (older
    callers, future stub) we pass it through.
    """
    if "char_start" in raw and "category" in raw:
        return raw
    return {
        "category": raw.get("tell_type", raw.get("category", "unknown")),
        "fragment": raw.get("matched_text", raw.get("fragment", "")),
        "line": int(raw.get("line", 1)),
        "char_start": int(raw.get("char_offset", raw.get("char_start", 0))),
        "char_end": int(raw.get("end_offset", raw.get("char_end", 0))),
        "severity": int(raw.get("severity", 1)),
        "suggestion": str(raw.get("suggestion", "") or ""),
    }


def _normalize_detector_score(raw: Any, fallback_name: str) -> dict[str, Any]:
    """Translate a benchmark.detectors.DetectorScore (or dict) into the MCP schema.

    The benchmark package ships its own pydantic ``DetectorScore`` with
    fields including ``name``, ``probability_ai``, ``confidence``,
    ``latency_seconds``, and an ``error`` tier. The MCP-facing model has
    a narrower public surface: ``detector``, ``probability_ai``,
    ``confidence``, ``error``. This shim accepts either a pydantic model
    or a plain dict and projects to the public surface.
    """
    if hasattr(raw, "model_dump"):
        data = raw.model_dump()
    elif isinstance(raw, dict):
        data = dict(raw)
    else:
        data = {"probability_ai": float(raw)}

    error_value = data.get("error", "") or ""
    if not isinstance(error_value, str):
        # Some detectors carry a richer error object; flatten to message.
        error_value = str(error_value)

    return {
        "detector": str(data.get("name", data.get("detector", fallback_name))),
        "probability_ai": float(data.get("probability_ai", -1.0)),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "error": error_value,
    }


def _classify(score: float) -> str:
    """Map a probability_ai value to a coarse three-way verdict."""
    if score < 0.0:
        return "unknown"
    if score < 0.3:
        return "human"
    if score < 0.7:
        return "uncertain"
    return "ai"


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool
def humanize(
    text: str,
    style: str = "default",
    preserve_voice: bool = True,
    intensity: float = 0.7,
) -> str:
    """Rewrite AI-generated text so it reads as human-authored.

    Runs the configured pipeline of passes (preprocess, surface-tell
    substitution, watermark scrub, stylometric smoothing, optional
    paraphrase passes; see ``docs/ARCHITECTURE.md``) and returns the final
    text. For diagnostic output (per-pass diff, before/after detector
    scores) use :func:`humanize_and_verify` instead.

    Parameters
    ----------
    text : str
        The input prose to humanize. Markdown formatting is tolerated and
        normalized in the preprocess pass.
    style : str, default "default"
        Name of a style preset registered in ``styles/``. Common presets
        include ``"casual"``, ``"blog"``, ``"academic"``, ``"esl"``,
        ``"neurodivergent"``, ``"preserve"``. Use :func:`list_styles` to
        enumerate what is actually available at runtime.
    preserve_voice : bool, default True
        When True, the pipeline retains identifying stylistic features the
        author would want to keep (sentence-length distribution, lexical
        idiom, characteristic punctuation). When False, the pipeline is
        free to flatten voice in pursuit of detector evasion.
    intensity : float, default 0.7
        Continuous control in [0, 1] mapping to the ``minimal`` /
        ``balanced`` / ``aggressive`` levers documented in the research
        dossier. Roughly: 0.0-0.33 minimal, 0.34-0.66 balanced, 0.67-1.0
        aggressive. Higher values run more passes and apply heavier
        per-pass edits.

    Returns
    -------
    str
        The humanized text. If no pipeline module is available the input
        is returned unchanged with a logged warning.
    """
    if not text:
        return text
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("intensity must be in [0.0, 1.0]")

    pipelines_pkg = _try_import("pipelines")
    if pipelines_pkg is None or not hasattr(pipelines_pkg, "HumanizationPipeline"):
        logger.warning(
            "pipelines.HumanizationPipeline not found; returning input unchanged. "
            "Install the pipeline package or wait for the Pipeline Builder."
        )
        return text

    try:
        config = pipelines_pkg.HumanizeConfig(
            style=style,
            intensity=_intensity_to_label(intensity),
            preserve_voice=preserve_voice,
        )
        pipeline = pipelines_pkg.HumanizationPipeline()
        result = pipeline.run(text, config)
        return result.text
    except Exception as exc:
        logger.exception("humanize pipeline failed: %s", exc)
        raise


@mcp.tool
def detect_tells(text: str) -> TellsReport:
    """Locate AI writing tells in the input text.

    Reports surface signatures catalogued in
    ``research/02_ai_tells_catalog.md``: excess vocabulary (the "delve"
    cluster), em-dash overuse, sentence-initial discourse markers,
    copular templates, conversational scaffolding, and parallel-structure
    overuse. Each tell carries a 1-indexed line number, character offsets,
    a 1-to-5 severity, and an optional substitution suggestion.

    The intended use is diagnostic: surface what would be edited by
    :func:`humanize` so a writer can decide which tells to preserve and
    which to remove.

    Parameters
    ----------
    text : str
        The input prose to scan.

    Returns
    -------
    TellsReport
        Structured report containing the tell list and a per-category
        count summary.
    """
    if not text:
        return TellsReport(text_length=0, tell_count=0, tells=[], summary={})

    pipelines_pkg = _try_import("pipelines")
    if pipelines_pkg is None or not hasattr(pipelines_pkg, "detect_tells"):
        logger.warning(
            "pipelines.detect_tells not found; returning empty report."
        )
        return TellsReport(
            text_length=len(text),
            tell_count=0,
            tells=[],
            summary={},
        )

    try:
        raw = pipelines_pkg.detect_tells(text)
    except Exception as exc:
        logger.exception("detect_tells failed: %s", exc)
        raise

    tells = [TellLocation.model_validate(_normalize_tell(item)) for item in raw]
    summary: dict[str, int] = {}
    for tell in tells:
        summary[tell.category] = summary.get(tell.category, 0) + 1

    return TellsReport(
        text_length=len(text),
        tell_count=len(tells),
        tells=tells,
        summary=summary,
    )


@mcp.tool
def score_humanity(
    text: str,
    detectors: list[str] | None = None,
) -> HumanityReport:
    """Score how AI-like the text reads to one or more open detectors.

    Wraps the local detector adapters in ``benchmark/``. The default
    detector list is ``["roberta-base"]`` (the canonical academic
    baseline; see ``research/01_detector_landscape.md``). Other adapters
    such as ``"fast_detect_gpt"`` and ``"binoculars"`` are added as the
    benchmark package matures.

    Aggregate probability is the arithmetic mean of detector scores that
    returned successfully. If every detector failed, ``aggregate`` is
    -1.0 and the verdict is ``"unknown"``.

    Parameters
    ----------
    text : str
        The input prose to score.
    detectors : list of str, optional
        Names of detector adapters to run. Defaults to ``["roberta-base"]``
        if not given or empty.

    Returns
    -------
    HumanityReport
        Per-detector scores plus the aggregate and a coarse verdict.
    """
    detector_names = detectors or ["roberta-base"]

    bench = _try_import("benchmark.detectors")
    scores: list[DetectorScore] = []

    if bench is None or not hasattr(bench, "default_detectors"):
        logger.warning(
            "benchmark.detectors.default_detectors not found; returning unknown verdict."
        )
        for name in detector_names:
            scores.append(
                DetectorScore(
                    detector=name,
                    probability_ai=-1.0,
                    confidence=0.0,
                    error="benchmark.detectors module not available",
                )
            )
        return HumanityReport(
            text_length=len(text),
            detector_scores=scores,
            aggregate_probability_ai=-1.0,
            verdict="unknown",
        )

    # Build a name -> Detector lookup. ``default_detectors`` returns a list
    # of pre-instantiated Detector objects each carrying a ``.name``.
    try:
        available = {d.name: d for d in bench.default_detectors(include_optional=True)}
    except TypeError:
        # Older signature without the kwarg.
        available = {d.name: d for d in bench.default_detectors()}
    except Exception as exc:
        logger.exception("failed to enumerate detectors: %s", exc)
        available = {}

    for name in detector_names:
        detector = available.get(name)
        if detector is None:
            scores.append(
                DetectorScore(
                    detector=name,
                    probability_ai=-1.0,
                    confidence=0.0,
                    error=f"unknown detector '{name}'",
                )
            )
            continue
        try:
            raw = detector.score(text)
            scores.append(
                DetectorScore.model_validate(_normalize_detector_score(raw, name))
            )
        except Exception as exc:
            logger.warning("detector %s failed: %s", name, exc)
            scores.append(
                DetectorScore(
                    detector=name,
                    probability_ai=-1.0,
                    confidence=0.0,
                    error=str(exc),
                )
            )

    successful = [s.probability_ai for s in scores if s.probability_ai >= 0.0 and not s.error]
    aggregate = sum(successful) / len(successful) if successful else -1.0

    return HumanityReport(
        text_length=len(text),
        detector_scores=scores,
        aggregate_probability_ai=aggregate,
        verdict=_classify(aggregate),
    )


@mcp.tool
def apply_style(text: str, style: str) -> str:
    """Apply a style preset to text without running humanization passes.

    Useful when the caller wants pure register translation (formal to
    casual, academic to blog, etc.) without removing AI tells. The set
    of legal style names is whatever :func:`list_styles` returns.

    Parameters
    ----------
    text : str
        The input prose.
    style : str
        Name of a style preset registered in ``styles/``.

    Returns
    -------
    str
        The restyled text. If the style module is missing the input is
        returned unchanged with a logged warning.
    """
    if not text:
        return text

    # Style transfer is a thin wrapper over the pipeline with most passes
    # disabled. The brief specifies "pure style transfer without
    # humanization passes". We approximate that here by running the
    # pipeline with only the register-shift pass enabled (pass 8). If a
    # dedicated styles transfer entry point appears later we will prefer it.
    pipelines_pkg = _try_import("pipelines")
    if pipelines_pkg is None or not hasattr(pipelines_pkg, "HumanizationPipeline"):
        logger.warning(
            "pipelines.HumanizationPipeline not found; returning input unchanged."
        )
        return text

    try:
        # Skip every pass except 8 (register shift / style transfer).
        config = pipelines_pkg.HumanizeConfig(
            style=style,
            intensity="balanced",
            preserve_voice=True,
            skip_passes=[1, 2, 3, 4, 5, 6, 7, 9],
        )
        pipeline = pipelines_pkg.HumanizationPipeline()
        result = pipeline.run(text, config)
        return result.text
    except Exception as exc:
        logger.exception("apply_style failed: %s", exc)
        raise


@mcp.tool
def list_styles() -> list[str]:
    """List the names of all currently registered style presets.

    Style presets are loaded from the ``styles/`` package. If the package
    is unavailable an empty list is returned.

    Returns
    -------
    list of str
        Sorted list of preset names, e.g. ``["academic", "blog", "casual",
        "esl", "neurodivergent", "preserve"]``.
    """
    styles_mod = _try_import("styles")
    if styles_mod is None or not hasattr(styles_mod, "list_styles"):
        logger.warning(
            "styles.list_styles not found; returning empty list."
        )
        return []
    try:
        names = list(styles_mod.list_styles())
        return sorted(set(names))
    except Exception as exc:
        logger.exception("list_styles failed: %s", exc)
        return []


@mcp.tool
def humanize_and_verify(
    text: str,
    style: str = "blog",
    target_ai_score: float = 0.3,
    max_iterations: int = 3,
    target_detector: str = "trusted_mean",
    candidates_per_iteration: int = 3,
) -> VerifyResult:
    """Humanize, then iterate against detectors until a target score is met.

    v0.2.0 (Bet 3): wraps :class:`pipelines.IterativeHumanizer`, which
    implements the Cheng et al. 2025 detector-guided loop (research/04
    section 1.3). Each iteration:

    1. Locates the worst-scoring paragraph in the current text.
    2. Generates ``candidates_per_iteration`` stochastic paraphrase
       candidates of that paragraph (via ``ParaphrasePass.paraphrase_candidates``).
    3. Scores each candidate, keeps the lowest, splices it back in.
    4. Re-scores the whole text. If at or below ``target_ai_score``, returns.

    This replaces the v0.1.0 loop, which re-ran the *deterministic* 9-pass
    pipeline at ramped intensities. As documented in
    ``docs/REVIEW_v0.1.0.md`` section 2.9, every pass except 9-heavy is
    idempotent on its own output, so iterations 2-3 of the old loop did no
    work. The new loop is meaningfully different because it depends on
    *stochastic* candidate generation: only non-deterministic search can
    converge on a lower score after a deterministic fixed point.

    The function always returns a result, even if the target was not
    reached; callers should check ``target_reached`` to know.

    Parameters
    ----------
    text : str
        The input prose to humanize.
    style : str, default "blog"
        Name of a style preset; see :func:`list_styles`.
    target_ai_score : float, default 0.3
        The aggregate probability_ai value below which the loop exits.
        Must be in [0, 1]. Cheng et al. 2025 use 0.15 as their stop value;
        we keep 0.3 ("comfortably human") as the default for backwards
        compatibility with v0.1.x callers.
    max_iterations : int, default 3
        Hard upper bound on improvement iterations. Must be >= 1.
        Iteration 0 is always the baseline humanization and does not
        count against this bound.
    target_detector : str, default "trusted_mean"
        Which detector field the loop optimizes against.
        ``"trusted_mean"`` (default) uses the suite's mean over detectors
        without documented bias caveats. ``"raw_mean"`` uses the unweighted
        mean across all detectors. Any other value is treated as a
        specific detector name (e.g. ``"roberta_openai"``).
    candidates_per_iteration : int, default 3
        How many stochastic paraphrase candidates to generate per
        iteration. The Cheng et al. 2025 paper uses 3-5; we default to 3
        as a quality / latency tradeoff. Must be >= 1.

    Returns
    -------
    VerifyResult
        Final text, iteration count, before/after scores, per-iteration
        score trace, and ``target_reached``. The ``notes`` field carries
        a human-readable trace for debugging.
    """
    if not 0.0 <= target_ai_score <= 1.0:
        raise ValueError("target_ai_score must be in [0.0, 1.0]")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if candidates_per_iteration < 1:
        raise ValueError("candidates_per_iteration must be >= 1")

    # We always populate `initial_score` with the legacy detector report so
    # the response schema is preserved for v0.1.x clients. The iterative
    # loop has its own (richer) scoring backed by BenchmarkSuite, so even
    # when `score_humanity` returns "unknown" (e.g. roberta-base unavailable)
    # the loop can still operate against the heuristic detector suite.
    initial = score_humanity(text)

    # Build (or reuse) the iterative humanizer. We cache it on the module
    # so the BenchmarkSuite's lazy-loaded models stay warm across calls
    # within the same server process.
    pipelines_pkg = _try_import("pipelines")
    if pipelines_pkg is None or not hasattr(pipelines_pkg, "IterativeHumanizer"):
        # Fall back to the v0.1.x deterministic ramp loop. This keeps the
        # tool functional in environments where the pipelines package is a
        # stub or stale install.
        logger.warning(
            "pipelines.IterativeHumanizer not found; falling back to v0.1 "
            "deterministic ramp loop."
        )
        return _v01_fallback_verify(
            text,
            style=style,
            target_ai_score=target_ai_score,
            max_iterations=max_iterations,
            initial=initial,
            target_detector=target_detector,
            candidates_per_iteration=candidates_per_iteration,
        )

    try:
        humanizer = _get_iterative_humanizer(pipelines_pkg)
        result = humanizer.humanize_and_verify(
            text,
            style=style,
            target_ai_score=target_ai_score,
            max_iterations=max_iterations,
            candidates_per_iteration=candidates_per_iteration,
            target_detector=target_detector,
        )
    except Exception as exc:
        logger.exception("iterative humanize_and_verify failed: %s", exc)
        return _v01_fallback_verify(
            text,
            style=style,
            target_ai_score=target_ai_score,
            max_iterations=max_iterations,
            initial=initial,
            target_detector=target_detector,
            candidates_per_iteration=candidates_per_iteration,
            extra_note=f"iterative loop crashed: {exc}; fell back to ramp loop",
        )

    final = score_humanity(result.text)

    # If the legacy initial score was unavailable but the iterative loop
    # produced its own score trace, surface that on the report so the user
    # can see the iteration progress regardless of which detector backend
    # is configured.
    notes = list(result.notes)
    if initial.aggregate_probability_ai < 0.0 and result.per_iteration_scores:
        notes.insert(
            0,
            (
                "score_humanity backend unavailable; per_iteration_scores "
                "below come from the BenchmarkSuite heuristic detectors used "
                "by the iterative loop."
            ),
        )

    return VerifyResult(
        text=result.text,
        iterations=result.iterations,
        target_ai_score=target_ai_score,
        initial_score=initial,
        final_score=final,
        target_reached=result.target_reached,
        notes=notes,
        target_detector=result.target_detector,
        candidates_per_iteration=result.candidates_per_iteration,
        per_iteration_scores=list(result.per_iteration_scores),
        total_time_ms=result.total_time_ms,
    )


# Module-level cache for the iterative humanizer. Reuse keeps the
# BenchmarkSuite's lazy-loaded detector models warm across tool calls
# within the same server process.
_ITERATIVE_HUMANIZER_CACHE: dict[str, Any] = {}


def _get_iterative_humanizer(pipelines_pkg: Any) -> Any:
    """Build (or return cached) IterativeHumanizer with the default pipeline."""
    cache_key = "default"
    cached = _ITERATIVE_HUMANIZER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    pipeline = pipelines_pkg.HumanizationPipeline()
    humanizer = pipelines_pkg.IterativeHumanizer(pipeline)
    _ITERATIVE_HUMANIZER_CACHE[cache_key] = humanizer
    return humanizer


def _v01_fallback_verify(
    text: str,
    *,
    style: str,
    target_ai_score: float,
    max_iterations: int,
    initial: HumanityReport,
    target_detector: str,
    candidates_per_iteration: int,
    extra_note: str | None = None,
) -> VerifyResult:
    """The v0.1.x deterministic ramp loop, kept as a fallback path.

    Used when ``pipelines.IterativeHumanizer`` cannot be imported (stub
    install) or when the iterative loop crashes for an environmental
    reason. Documented in section 2.9 of the v0.1.0 review as essentially
    a no-op for iterations 2-3, but it is better than crashing.
    """
    notes: list[str] = []
    if extra_note:
        notes.append(extra_note)
    current_text = text
    current_score = initial

    for i in range(1, max_iterations + 1):
        intensity = min(0.5 + 0.2 * (i - 1), 1.0)
        try:
            current_text = humanize(
                current_text,
                style=style,
                preserve_voice=True,
                intensity=intensity,
            )
        except Exception as exc:
            notes.append(f"iteration {i} humanize failed: {exc}")
            break

        current_score = score_humanity(current_text)
        notes.append(
            f"iteration {i}: intensity={intensity:.2f}, "
            f"aggregate={current_score.aggregate_probability_ai:.3f}"
        )

        if (
            current_score.aggregate_probability_ai >= 0.0
            and current_score.aggregate_probability_ai <= target_ai_score
        ):
            return VerifyResult(
                text=current_text,
                iterations=i,
                target_ai_score=target_ai_score,
                initial_score=initial,
                final_score=current_score,
                target_reached=True,
                notes=notes,
                target_detector=target_detector,
                candidates_per_iteration=candidates_per_iteration,
                per_iteration_scores=[],
                total_time_ms=0,
            )

    return VerifyResult(
        text=current_text,
        iterations=max_iterations,
        target_ai_score=target_ai_score,
        initial_score=initial,
        final_score=current_score,
        target_reached=False,
        notes=notes,
        target_detector=target_detector,
        candidates_per_iteration=candidates_per_iteration,
        per_iteration_scores=[],
        total_time_ms=0,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server over stdio (the canonical MCP transport).

    Invoked by the ``humanize-mcp`` console script defined in pyproject.toml.
    Clients (Claude Code, Copilot CLI, etc.) launch this as a subprocess
    and speak MCP over stdin/stdout.
    """
    mcp.run()


if __name__ == "__main__":
    main()
