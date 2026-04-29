"""HumanizeMCP pipeline package.

Public API:

* :class:`HumanizationPipeline` -- orchestrator
* :class:`HumanizeConfig` / :class:`HumanizeResult` -- pydantic IO models
* :class:`PipelinePass` -- ABC for new passes
* The nine concrete pass classes (``EmDashPass`` through ``ParaphrasePass``)
* :func:`detect_tells` -- linter used by the MCP ``detect_tells`` tool
"""

from __future__ import annotations

from .base import (
    HumanizationPipeline,
    HumanizeConfig,
    HumanizeResult,
    PipelinePass,
    PassRecord,
)
from .pass_01_em_dash import EmDashPass
from .pass_02_lexical import LexicalPass
from .pass_03_structural import StructuralPass
from .pass_04_rhythm import RhythmPass
from .pass_05_contractions import ContractionsPass
from .pass_06_voice_injection import VoiceInjectionPass
from .pass_07_punctuation import PunctuationPass
from .pass_08_register_shift import RegisterShiftPass
from .pass_09_paraphrase import ParaphrasePass
from .iterative import IterativeHumanizer, IterativeResult
from .tells_detector import (
    Tell,
    TellPattern,
    TellsDetector,
    detect_tells,
    summarize_tells,
)

__all__ = [
    "HumanizationPipeline",
    "HumanizeConfig",
    "HumanizeResult",
    "PipelinePass",
    "PassRecord",
    "EmDashPass",
    "LexicalPass",
    "StructuralPass",
    "RhythmPass",
    "ContractionsPass",
    "VoiceInjectionPass",
    "PunctuationPass",
    "RegisterShiftPass",
    "ParaphrasePass",
    "IterativeHumanizer",
    "IterativeResult",
    "Tell",
    "TellPattern",
    "TellsDetector",
    "detect_tells",
    "summarize_tells",
    "humanize",
    "Pipeline",
]


# v0.1.1: convenience shim so the benchmark runner (which probes for a
# module-level `humanize` callable or a `Pipeline` class) can find the
# pipeline. Without this the benchmark silently scored raw text only.
def humanize(
    text: str,
    style: str = "blog",
    intensity: str = "balanced",
    iterate: bool = True,
    target_detector: str = "trusted_mean",
    target_ai_score: float = 0.20,
    max_iterations: int = 1,
    candidates_per_iteration: int = 5,
    **kwargs,
) -> str:
    """Convenience wrapper: humanize text with a single function call.

    v0.2.1: defaults now run a single round of detector-guided iteration
    (Cheng et al. 2025) so the output is best-in-class against arbitrary
    detectors out of the box, not just the heuristic. To skip iteration
    and get the v0.1.x-style fast deterministic pass, set ``iterate=False``.

    Args:
        text: input text to humanize.
        style: a registered style preset (see ``list_styles()``).
        intensity: ``light``, ``balanced``, or ``aggressive``.
        iterate: when True (default), run one round of detector-guided
            paragraph swap to drive any spike on transformer detectors back
            down. When False, returns the deterministic pipeline output.
        target_detector: which detector to optimize against. Use
            ``"trusted_mean"`` (default) to average over all non-biased
            detectors, or pass a specific detector name like
            ``"roberta_openai"``.
        target_ai_score: threshold below which iteration stops early.
        max_iterations: hard cap on iteration rounds.
        candidates_per_iteration: number of stochastic paraphrase candidates
            generated per iteration. Higher = better odds of finding a
            low-scoring rewrite, slower runtime.
        **kwargs: forwarded to ``HumanizeConfig``.

    Returns:
        the humanized text.
    """
    if not iterate:
        config = HumanizeConfig(style=style, intensity=intensity, **kwargs)
        return HumanizationPipeline().run(text, config).text

    # Iterative path (v0.2.1 default).
    from .iterative import IterativeHumanizer

    iterator = IterativeHumanizer(base_pipeline=HumanizationPipeline())
    result = iterator.humanize_and_verify(
        text=text,
        style=style,
        target_ai_score=target_ai_score,
        max_iterations=max_iterations,
        candidates_per_iteration=candidates_per_iteration,
        target_detector=target_detector,
    )
    return result.text


# Back-compat alias used by the benchmark runner's pipeline probe.
Pipeline = HumanizationPipeline
