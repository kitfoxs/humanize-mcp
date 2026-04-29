"""HumanizeMCP benchmarking module.

Public surface:

* :class:`BenchmarkSuite` — orchestrates running detectors over text.
* :class:`BenchmarkReport`, :class:`ComparisonReport`,
  :class:`TextStatistics` — pydantic models for results.
* :func:`score_text` — convenience function for one-shot scoring.
* Detector classes: :class:`HeuristicDetector`,
  :class:`RobertaOpenAIDetector`, :class:`ChatGPTRobertaDetector`,
  :class:`DesklibAIDetector`, :class:`FastDetectGPT`,
  :class:`BinocularsDetector`.
* :func:`pre_commit_check` — quick "lint your text before posting" helper.

The MCP server module imports :func:`score_text` from here.
"""

from __future__ import annotations

from .benchmark_suite import (
    BenchmarkReport,
    BenchmarkSuite,
    ComparisonReport,
    TextStatistics,
    score_text,
    text_statistics,
)
from .detectors import (
    BinocularsDetector,
    ChatGPTRobertaDetector,
    DesklibAIDetector,
    Detector,
    DetectorScore,
    FastDetectGPT,
    HeuristicDetector,
    RobertaOpenAIDetector,
    burstiness,
    count_tells,
    default_detectors,
    lexical_diversity,
)
from .pre_commit_check import PreCommitResult, pre_commit_check

__all__ = [
    "BenchmarkSuite",
    "BenchmarkReport",
    "ComparisonReport",
    "TextStatistics",
    "score_text",
    "text_statistics",
    "Detector",
    "DetectorScore",
    "HeuristicDetector",
    "RobertaOpenAIDetector",
    "ChatGPTRobertaDetector",
    "DesklibAIDetector",
    "FastDetectGPT",
    "BinocularsDetector",
    "default_detectors",
    "burstiness",
    "lexical_diversity",
    "count_tells",
    "pre_commit_check",
    "PreCommitResult",
]
