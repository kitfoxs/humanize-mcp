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
    "Tell",
    "TellPattern",
    "TellsDetector",
    "detect_tells",
    "summarize_tells",
]
