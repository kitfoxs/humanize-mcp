"""Base classes for the HumanizeMCP pipeline.

Defines the abstract pass interface, the orchestrator, and the pydantic
config / result models. Each concrete pass lives in its own module and
implements :class:`PipelinePass`.

Reference: research/06_implementation_recommendations.md sections 1, 2, 9.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class PassRecord:
    """What a single pass did to the text."""

    pass_id: int
    pass_name: str
    enabled: bool
    skipped: bool = False
    skipped_reason: Optional[str] = None
    changes: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def change_count(self) -> int:
        return len(self.changes)


class HumanizeConfig(BaseModel):
    """Configuration for a humanization run.

    Mirrors the public MCP tool signature documented in research/06 section 1.
    """

    style: str = Field(default="blog", description="Style preset name.")
    intensity: str = Field(
        default="balanced",
        description="One of 'minimal', 'balanced', 'aggressive'.",
    )
    preserve_voice: bool = Field(
        default=True,
        description="If true, preserve characteristic voice features even at cost of detector score.",
    )
    target_register: Optional[str] = Field(
        default=None,
        description="Override the preset register: casual, neutral, academic, formal.",
    )
    skip_passes: List[int] = Field(
        default_factory=list,
        description="Pass numbers (1-9) to skip entirely.",
    )
    pass_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-pass configuration overrides keyed by 'pass_NN_name'.",
    )
    seed: int = Field(default=42, description="Random seed for deterministic runs.")
    return_diff: bool = Field(default=False)
    return_per_pass_log: bool = Field(default=True)


class HumanizeResult(BaseModel):
    """Output of a humanization run."""

    text: str
    passes_applied: List[str] = Field(default_factory=list)
    tells_removed_count: int = 0
    processing_time_ms: float = 0.0
    pass_log: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PipelinePass(ABC):
    """Abstract base for every transformation pass.

    Concrete passes should be small, deterministic, and log every change
    they make so the diff/explain output is faithful.
    """

    pass_id: int = 0
    pass_name: str = "abstract"

    @abstractmethod
    def apply(self, text: str, config: Dict[str, Any]) -> str:
        """Transform ``text`` using ``config`` and return the new text."""

    def changes(self) -> List[Dict[str, Any]]:
        """Return the list of changes made during the most recent ``apply``.

        Default implementation: empty list. Subclasses should override and
        track changes as they go.
        """
        return getattr(self, "_changes", [])

    def reset_changes(self) -> None:
        self._changes: List[Dict[str, Any]] = []

    def log_change(
        self,
        kind: str,
        before: str,
        after: str,
        offset: Optional[int] = None,
    ) -> None:
        record: Dict[str, Any] = {"kind": kind, "before": before, "after": after}
        if offset is not None:
            record["offset"] = offset
        self.changes().append(record)


class HumanizationPipeline:
    """Orchestrates the 9-pass humanization pipeline.

    Each pass receives the current text and returns a transformed version.
    Passes can be skipped via ``HumanizeConfig.skip_passes``. Per-pass
    configuration is merged from preset defaults + caller overrides.
    """

    def __init__(
        self,
        passes: Optional[List[PipelinePass]] = None,
        style_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> None:
        if passes is None:
            passes = self._default_passes()
        self.passes = passes
        self._style_loader = style_loader

    @staticmethod
    def _default_passes() -> List[PipelinePass]:
        # Imported lazily to avoid a circular import at module load.
        from .pass_01_em_dash import EmDashPass
        from .pass_02_lexical import LexicalPass
        from .pass_03_structural import StructuralPass
        from .pass_04_rhythm import RhythmPass
        from .pass_05_contractions import ContractionsPass
        from .pass_06_voice_injection import VoiceInjectionPass
        from .pass_07_punctuation import PunctuationPass
        from .pass_08_register_shift import RegisterShiftPass
        from .pass_09_paraphrase import ParaphrasePass

        return [
            EmDashPass(),
            LexicalPass(),
            StructuralPass(),
            RhythmPass(),
            ContractionsPass(),
            VoiceInjectionPass(),
            PunctuationPass(),
            RegisterShiftPass(),
            ParaphrasePass(),
        ]

    def run(self, text: str, config: HumanizeConfig) -> HumanizeResult:
        """Run every pass that is enabled and return a HumanizeResult."""

        start = time.perf_counter()
        style_cfg = self._load_style(config.style)
        merged_cfg = self._merge_configs(style_cfg, config)

        current_text = text
        records: List[PassRecord] = []
        applied_names: List[str] = []
        total_changes = 0
        warnings: List[str] = []

        enabled_set = set(merged_cfg.get("passes_enabled", list(range(1, 10))))

        for p in self.passes:
            if p.pass_id in config.skip_passes or p.pass_id not in enabled_set:
                records.append(
                    PassRecord(
                        pass_id=p.pass_id,
                        pass_name=p.pass_name,
                        enabled=False,
                        skipped=True,
                        skipped_reason="disabled by config",
                    )
                )
                continue

            p.reset_changes()
            pass_cfg = self._pass_config(merged_cfg, config, p)
            t0 = time.perf_counter()
            try:
                current_text = p.apply(current_text, pass_cfg)
            except Exception as exc:
                logger.exception("pass %s failed: %s", p.pass_name, exc)
                warnings.append(f"pass {p.pass_name} failed: {exc}")
                records.append(
                    PassRecord(
                        pass_id=p.pass_id,
                        pass_name=p.pass_name,
                        enabled=True,
                        skipped=True,
                        skipped_reason=f"exception: {exc}",
                    )
                )
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            records.append(
                PassRecord(
                    pass_id=p.pass_id,
                    pass_name=p.pass_name,
                    enabled=True,
                    changes=list(p.changes()),
                    elapsed_ms=elapsed_ms,
                )
            )
            applied_names.append(p.pass_name)
            total_changes += len(p.changes())

        # v0.1.1 cleanup pass: fix a/an agreement broken by lexical/paraphrase
        # substitutions. Without this we leave artifacts like "a integrated
        # approach" which is itself a fluent-text tell humans never produce.
        current_text = self._fix_a_an(current_text)

        # v0.1.1 cleanup: collapse double spaces left by parens/em-dash passes.
        current_text = re.sub(r"  +", " ", current_text)

        elapsed_total_ms = (time.perf_counter() - start) * 1000.0

        return HumanizeResult(
            text=current_text,
            passes_applied=applied_names,
            tells_removed_count=total_changes,
            processing_time_ms=elapsed_total_ms,
            pass_log=[self._record_to_dict(r) for r in records] if config.return_per_pass_log else [],
            warnings=warnings,
        )


    @staticmethod
    def _fix_a_an(text: str) -> str:
        """Fix a/an article agreement after substitutions.

        Rules:
        - "a apple" -> "an apple"
        - "an car"  -> "a car"
        Honor case (A/An).
        Skip when next "word" starts with a digit, hyphen, or other non-alpha.
        v0.1.1 patch.
        """
        # a/an before vowel-initial word
        def _fix_a(match: re.Match[str]) -> str:
            article, sep, word = match.group(1), match.group(2), match.group(3)
            new_article = "An" if article == "A" else "an"
            return f"{new_article}{sep}{word}"

        def _fix_an(match: re.Match[str]) -> str:
            article, sep, word = match.group(1), match.group(2), match.group(3)
            new_article = "A" if article == "An" else "a"
            return f"{new_article}{sep}{word}"

        # a -> an before vowel sound (rough heuristic; "u" can be "a" or "an"
        # depending on the actual sound; default to vowel rule, accept some misses)
        text = re.sub(r"\b(a|A)(\s+)([aeiouAEIOU]\w*)", _fix_a, text)
        # an -> a before consonant
        text = re.sub(r"\b(an|An)(\s+)([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w*)", _fix_an, text)
        return text

    def _load_style(self, style: str) -> Dict[str, Any]:
        if self._style_loader is not None:
            return self._style_loader(style)
        from styles import load_style

        return load_style(style)

    def _merge_configs(
        self, style_cfg: Dict[str, Any], config: HumanizeConfig
    ) -> Dict[str, Any]:
        merged = dict(style_cfg)
        merged.setdefault("passes_enabled", list(range(1, 10)))
        merged["intensity"] = config.intensity
        merged["preserve_voice"] = config.preserve_voice
        if config.target_register:
            merged["target_register"] = config.target_register
        merged["seed"] = config.seed
        return merged

    @staticmethod
    def _pass_config(
        merged: Dict[str, Any], config: HumanizeConfig, p: PipelinePass
    ) -> Dict[str, Any]:
        key = f"pass_{p.pass_id:02d}_{p.pass_name}"
        out: Dict[str, Any] = {
            "intensity": merged.get("intensity", "balanced"),
            "preserve_voice": merged.get("preserve_voice", True),
            "target_register": merged.get("target_register", "neutral"),
            "seed": merged.get("seed", 42),
            "style": merged,
        }
        out.update(merged.get("pass_configs", {}).get(key, {}))
        out.update(config.pass_configs.get(key, {}))
        return out

    @staticmethod
    def _record_to_dict(r: PassRecord) -> Dict[str, Any]:
        return {
            "pass_id": r.pass_id,
            "pass_name": r.pass_name,
            "enabled": r.enabled,
            "skipped": r.skipped,
            "skipped_reason": r.skipped_reason,
            "change_count": r.change_count,
            "elapsed_ms": round(r.elapsed_ms, 3),
            "changes": r.changes,
        }
