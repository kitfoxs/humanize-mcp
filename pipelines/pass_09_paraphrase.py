"""Pass 9: paraphrase pass.

The DIPPER-style paraphrase pass (research/04 section 1.2). Two
implementations live behind one ``mode`` switch:

* **Lightweight** (default): rule-based synonym substitution with a
  per-paragraph cap. Pure Python, fast, deterministic. Pulls no model
  weights and does no I/O.
* **Heavy** (opt-in via ``mode="heavy"`` or ``mode="auto"``): batched
  paraphrase via ``humarin/chatgpt_paraphraser_on_T5_base`` with a
  cosine-similarity drift gate (sentence-transformers MiniLM). Falls
  back to the lightweight path if the model fails to load (no network,
  no disk space, transformers missing).

Styles can opt into heavy mode per-intensity via ``mode_by_intensity``,
e.g. ``{"aggressive": "heavy"}`` keeps balanced and minimal on the cheap
path. See ``styles/blog.json`` for an example.

Reference: docs/REVIEW_v0.1.0.md sections 3.2, 5 (Bet 1).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .base import PipelinePass

logger = logging.getLogger(__name__)


# Curated synonym map. Conservative: each entry has a single safe alternative
# that preserves register. The lexical pass (pass 2) handles the heavier
# domain-specific table.
LIGHT_SYNONYMS = {
    r"\butilize\b": "use",
    r"\bUtilize\b": "Use",
    r"\bsubsequently\b": "later",
    r"\bSubsequently\b": "Later",
    r"\bnumerous\b": "many",
    r"\bNumerous\b": "Many",
    r"\bvarious\b": "several",
    r"\bVarious\b": "Several",
    r"\badditional\b": "more",
    r"\bsignificantly\b": "noticeably",
    r"\benable\b": "let",
    r"\bdemonstrate\b": "show",
    r"\bDemonstrate\b": "Show",
    r"\bcommence\b": "start",
    r"\bobtain\b": "get",
    r"\brequire\b": "need",
    r"\bassist\b": "help",
    r"\bpurchase\b": "buy",
    r"\bterminate\b": "end",
    r"\binitiate\b": "start",
    r"\bconstitute\b": "make up",
    r"\bsufficient\b": "enough",
}


class ParaphrasePass(PipelinePass):
    pass_id = 9
    pass_name = "paraphrase"

    def __init__(
        self,
        paraphraser: Optional[Any] = None,
        gate: Optional[Any] = None,
    ) -> None:
        # Both backends are injectable so tests can mock without touching
        # the network or HF cache. The real classes are lazy-imported when
        # a heavy run is first requested.
        self._paraphraser = paraphraser
        self._gate = gate
        self._defaults_instantiated: bool = False

    # ----- public entry point -----------------------------------------

    def apply(self, text: str, config: Dict[str, Any]) -> str:
        self.reset_changes()
        intensity: str = str(config.get("intensity", "balanced"))
        mode: str = self._effective_mode(config, intensity)

        if mode in {"heavy", "auto"}:
            self._ensure_default_models(config)
            assert self._paraphraser is not None  # set by _ensure_default_models
            if self._paraphraser.load():
                heavy_out = self._heavy_paraphrase(text, config)
                if heavy_out is not None:
                    return heavy_out
            else:
                logger.info(
                    "heavy paraphrase requested but model unavailable; "
                    "falling back to light mode"
                )

        return self._light_paraphrase(text, intensity)

    # ----- candidate API for the iteration agent ----------------------

    def paraphrase_candidates(
        self,
        text: str,
        n: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Return up to ``n`` stochastic paraphrase variants of ``text``.

        Used by the detector-guided iteration loop (Bet 3). Falls back to
        ``[text]`` if the heavy model is unavailable so callers can
        always treat the return value as non-empty.
        """

        cfg: Dict[str, Any] = config or {}
        self._ensure_default_models(cfg)
        assert self._paraphraser is not None
        if not self._paraphraser.load():
            return [text]
        return self._paraphraser.paraphrase_candidates(text, n=n)

    # ----- mode resolution --------------------------------------------

    @staticmethod
    def _effective_mode(config: Dict[str, Any], intensity: str) -> str:
        """Resolve ``mode`` honoring per-intensity overrides.

        Style files can ship ``"mode_by_intensity": {"aggressive": "heavy"}``
        which lets a single preset stay cheap on light/balanced runs and
        only pay the model cost when the caller explicitly asks for an
        aggressive rewrite.
        """

        mode = str(config.get("mode", "light"))
        by_intensity = config.get("mode_by_intensity") or {}
        if isinstance(by_intensity, dict) and intensity in by_intensity:
            mode = str(by_intensity[intensity])
        return mode

    def _ensure_default_models(self, config: Dict[str, Any]) -> None:
        if self._defaults_instantiated:
            return
        # Lazy import keeps the cold-start cost off the import path.
        from ._paraphrase_models import SemanticGate, T5Paraphraser

        if self._paraphraser is None:
            model_name = config.get("heavy_model")
            self._paraphraser = T5Paraphraser(model_name=model_name)
        if self._gate is None:
            threshold = float(config.get("similarity_threshold", 0.85))
            self._gate = SemanticGate(threshold=threshold)
        self._defaults_instantiated = True

    # ----- light path --------------------------------------------------

    def _light_paraphrase(self, text: str, intensity: str) -> str:
        max_subs_per_para = {
            "minimal": 1,
            "balanced": 3,
            "aggressive": 6,
        }.get(intensity, 3)

        paragraphs = re.split(r"(\n\s*\n)", text)
        new_paras: List[str] = []
        for chunk in paragraphs:
            if not chunk.strip() or chunk.startswith("\n"):
                new_paras.append(chunk)
                continue
            new_paras.append(self._paraphrase_paragraph_light(chunk, max_subs_per_para))
        return "".join(new_paras)

    def _paraphrase_paragraph_light(self, paragraph: str, max_subs: int) -> str:
        out = paragraph
        subs_done = 0
        for pattern, replacement in LIGHT_SYNONYMS.items():
            if subs_done >= max_subs:
                break
            compiled = re.compile(pattern)
            new_out, n = compiled.subn(replacement, out, count=1)
            if n > 0:
                self.log_change("paraphrase_synonym", pattern, replacement)
                subs_done += n
                out = new_out
        return out

    # ----- heavy path --------------------------------------------------

    def _heavy_paraphrase(self, text: str, config: Dict[str, Any]) -> Optional[str]:
        """Batched heavy paraphrase with semantic-drift gating.

        Splits on paragraph boundaries (preserving the separators), runs
        a single batched ``model.generate`` call across all substantive
        paragraphs, then per-paragraph checks cosine similarity against
        the original. Below-threshold candidates are dropped in favor of
        the original, which protects technical content from rewrites
        that change meaning.
        """

        assert self._paraphraser is not None
        assert self._gate is not None

        chunks = re.split(r"(\n\s*\n)", text)
        substantive_indices: List[int] = []
        substantive_texts: List[str] = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip() or chunk.startswith("\n"):
                continue
            substantive_indices.append(i)
            substantive_texts.append(chunk)

        if not substantive_texts:
            return text

        try:
            paraphrased = self._paraphraser.paraphrase_batch(substantive_texts)
        except Exception as exc:
            logger.warning("heavy paraphrase generation failed (%s)", exc)
            return None

        if len(paraphrased) != len(substantive_texts):
            logger.warning(
                "heavy paraphrase returned %d outputs for %d inputs; aborting heavy pass",
                len(paraphrased),
                len(substantive_texts),
            )
            return None

        for idx, original, candidate in zip(
            substantive_indices, substantive_texts, paraphrased
        ):
            if not candidate or candidate.strip() == original.strip():
                continue
            try:
                accepted = self._gate.passes(original.strip(), candidate.strip())
            except Exception as exc:
                logger.warning(
                    "semantic gate raised (%s); accepting candidate by default", exc
                )
                accepted = True

            if accepted:
                chunks[idx] = self._reattach_whitespace(original, candidate.strip())
                self.log_change("paraphrase_heavy", original[:60], candidate[:60])
            else:
                self.log_change(
                    "paraphrase_rejected_low_similarity",
                    original[:60],
                    candidate[:60],
                )

        return "".join(chunks)

    @staticmethod
    def _reattach_whitespace(original: str, replacement: str) -> str:
        """Preserve the original chunk's leading/trailing whitespace.

        The paragraph splitter keeps the inter-paragraph ``\\n\\n`` runs as
        their own chunks, but a paragraph chunk can still carry leading
        spaces (markdown indentation, etc.). We keep those intact so the
        document layout survives the rewrite.
        """

        lead_len = len(original) - len(original.lstrip())
        trail_len = len(original) - len(original.rstrip())
        lead = original[:lead_len]
        trail = original[len(original) - trail_len:] if trail_len else ""
        return f"{lead}{replacement}{trail}"
