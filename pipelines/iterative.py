"""Detector-guided iterative humanization.

Implements the Cheng et al. 2025 algorithm (research/04 section 1.3): given
an already-humanized text that still scores above the target AI-probability,
locate the worst paragraph, generate N stochastic paraphrase candidates,
score each, and replace the paragraph with the lowest-scoring candidate.
Repeat until the target is met or ``max_iterations`` is exhausted.

Why this is meaningfully different from v0.1.0's ``humanize_and_verify``
=======================================================================
v0.1.0 looped the *deterministic* 9-pass pipeline at progressively higher
intensities. As documented in ``docs/REVIEW_v0.1.0.md`` section 2.9, every
pass except 9-heavy is idempotent on its own output: after iteration 1, the
em-dash pass has no em dashes left to replace, the lexical pass has no
"delve" to swap, etc. The loop therefore did no useful work past iteration 1
and reported the same scores at balanced and aggressive intensity for every
shipped style.

This module replaces that no-op loop with the Cheng et al. 2025 prescription:
*stochastic* per-paragraph paraphrasing guided by detector feedback. Each
iteration generates N novel candidates and keeps the one the detector likes
least. That is the only way iteration can converge on a lower score: the
search has to be non-deterministic.

References
----------
* research/04_humanization_techniques.md section 1.3 (Cheng et al. 2025)
* docs/REVIEW_v0.1.0.md sections 2.9, 3.3, 3.4, and 5 (Bet 3)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from .base import HumanizationPipeline, HumanizeConfig, HumanizeResult

logger = logging.getLogger(__name__)


# Paragraph splitter shared with the rest of the pipeline (matches the
# pattern used by LexicalPass and ParaphrasePass: blank line as boundary).
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")

# Maximum wall-clock time budget per iteration. The contract documents a
# soft 30-second budget for a typical 5-paragraph essay; if exceeded we log
# and continue rather than abort, so the caller still gets a result.
_PER_ITERATION_BUDGET_S = 30.0


class IterativeResult(BaseModel):
    """Output of :meth:`IterativeHumanizer.humanize_and_verify`.

    Attributes
    ----------
    text : str
        The final humanized text. May equal the baseline humanization if no
        iteration improved the score.
    iterations : int
        How many improvement iterations ran. ``0`` means the baseline
        humanization already scored at or below the target.
    final_score : float
        The detector score the loop converged at (lower is better; ``-1.0``
        if scoring was unavailable).
    target_reached : bool
        True iff ``final_score <= target_ai_score``.
    per_iteration_scores : list[float]
        Whole-text score after iteration 0 (baseline) then after each
        improvement iteration, in order. The first entry is always present;
        subsequent entries appear only for iterations that actually ran.
    target_detector : str
        Which detector field we optimized against. One of
        ``"trusted_mean"``, ``"raw_mean"``, or a specific detector name as
        returned by the benchmark suite.
    target_ai_score : float
        Echo of the input target, for round-tripping into reports.
    candidates_per_iteration : int
        Echo of the input N, for round-tripping into reports.
    notes : list[str]
        Human-readable trace, one entry per significant decision (baseline
        score, paragraph chosen, candidate scores, fallback reasons).
    total_time_ms : int
        Total wall-clock time including baseline humanization, in
        milliseconds.
    """

    text: str
    iterations: int = 0
    final_score: float = -1.0
    target_reached: bool = False
    per_iteration_scores: list[float] = Field(default_factory=list)
    target_detector: str = "trusted_mean"
    target_ai_score: float = 0.15
    candidates_per_iteration: int = 3
    notes: list[str] = Field(default_factory=list)
    total_time_ms: int = 0


class IterativeHumanizer:
    """Detector-guided iterative humanization.

    Run the base 9-pass pipeline once to get a baseline. Score it against
    the configured detector. If still above the target, find the worst
    paragraph, ask the paraphrase pass for N stochastic candidates, score
    each, and replace the paragraph with the best one. Repeat until the
    whole-text score drops below the target or ``max_iterations`` is hit.

    The detector suite (and any models it lazy-loads) is held as an
    instance attribute so a single ``IterativeHumanizer`` reused across
    requests pays the model-load cost exactly once.
    """

    def __init__(
        self,
        base_pipeline: HumanizationPipeline,
        detectors: Optional[list[str]] = None,
        *,
        suite: Any = None,
    ) -> None:
        """Construct the iterative humanizer.

        Parameters
        ----------
        base_pipeline
            The 9-pass humanization pipeline used for the initial pass.
        detectors
            Optional list of detector names to score against. ``None`` (the
            default) uses ``BenchmarkSuite``'s default detector set, which
            already lazy-loads on first use.
        suite
            Optional pre-built ``BenchmarkSuite``-shaped object. Used by
            tests to inject a mock so we never load real models. If
            provided it overrides ``detectors``.
        """
        self.base_pipeline = base_pipeline
        self._detectors = detectors
        self._suite = suite  # may be None; built lazily in _get_suite

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def humanize_and_verify(
        self,
        text: str,
        style: str = "blog",
        target_ai_score: float = 0.15,
        max_iterations: int = 3,
        candidates_per_iteration: int = 3,
        target_detector: str = "trusted_mean",
        seed: int = 42,
    ) -> IterativeResult:
        """Run the Cheng et al. 2025 detector-guided loop.

        Parameters
        ----------
        text
            Input prose. Empty input returns immediately.
        style
            Style preset name forwarded to the base pipeline.
        target_ai_score
            AI probability at or below which the loop stops. In ``[0, 1]``.
        max_iterations
            Hard upper bound on improvement iterations. Must be ``>= 1``.
            Iteration 0 is always the baseline humanization and does not
            count against this bound.
        candidates_per_iteration
            How many stochastic paraphrase candidates to generate per
            improvement iteration. Must be ``>= 1``. ``1`` reduces the
            algorithm to deterministic per-paragraph paraphrase.
        target_detector
            Which detector field to optimize against. ``"trusted_mean"``
            (default) and ``"raw_mean"`` use the suite's aggregate; any
            other value is treated as a specific detector name.
        seed
            Forwarded to the base pipeline for deterministic baseline
            humanization. The stochastic paraphrase candidates take their
            seed from this value plus the iteration index, which lets the
            test suite reproduce a run end to end.
        """
        if not 0.0 <= target_ai_score <= 1.0:
            raise ValueError("target_ai_score must be in [0.0, 1.0]")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if candidates_per_iteration < 1:
            raise ValueError("candidates_per_iteration must be >= 1")

        t0 = time.perf_counter()

        if not text or not text.strip():
            return IterativeResult(
                text=text,
                iterations=0,
                final_score=-1.0,
                target_reached=False,
                per_iteration_scores=[],
                target_detector=target_detector,
                target_ai_score=target_ai_score,
                candidates_per_iteration=candidates_per_iteration,
                notes=["empty input; nothing to do"],
                total_time_ms=int((time.perf_counter() - t0) * 1000),
            )

        notes: list[str] = []

        # Step 1: baseline humanization at aggressive intensity. This is the
        # surface-tells / rhythm / contractions work that a single humanize()
        # call already does.
        baseline_result: HumanizeResult = self.base_pipeline.run(
            text,
            HumanizeConfig(style=style, intensity="aggressive", seed=seed),
        )
        current_text = baseline_result.text
        notes.append(
            f"baseline humanize: passes={len(baseline_result.passes_applied)}, "
            f"changes={baseline_result.tells_removed_count}"
        )

        # Step 2: score the baseline. If scoring is unavailable we cannot
        # iterate (no signal to optimize against), so we return what we have.
        baseline_score = self._score_whole(current_text)
        per_iteration_scores: list[float] = [baseline_score]
        notes.append(f"iteration 0 score ({target_detector}): {baseline_score:.4f}")

        if baseline_score < 0.0:
            notes.append("scoring unavailable; returning baseline humanization")
            return IterativeResult(
                text=current_text,
                iterations=0,
                final_score=baseline_score,
                target_reached=False,
                per_iteration_scores=per_iteration_scores,
                target_detector=target_detector,
                target_ai_score=target_ai_score,
                candidates_per_iteration=candidates_per_iteration,
                notes=notes,
                total_time_ms=int((time.perf_counter() - t0) * 1000),
            )

        if baseline_score <= target_ai_score:
            notes.append("baseline already below target; no iteration needed")
            return IterativeResult(
                text=current_text,
                iterations=0,
                final_score=baseline_score,
                target_reached=True,
                per_iteration_scores=per_iteration_scores,
                target_detector=target_detector,
                target_ai_score=target_ai_score,
                candidates_per_iteration=candidates_per_iteration,
                notes=notes,
                total_time_ms=int((time.perf_counter() - t0) * 1000),
            )

        # Step 3: iterate. For each iteration: locate the worst paragraph,
        # generate N candidates, score each (cheaply, on the paragraph
        # alone), pick the lowest, splice it back in, re-score the whole.
        current_score = baseline_score
        for iteration in range(1, max_iterations + 1):
            iter_start = time.perf_counter()
            iter_seed = seed + iteration

            paragraphs = self._split_paragraphs(current_text)
            worst_idx, worst_para_score = self._worst_paragraph(paragraphs)
            if worst_idx is None:
                notes.append(
                    f"iteration {iteration}: no scorable paragraph found; stopping"
                )
                break

            worst_para = paragraphs[worst_idx]
            notes.append(
                f"iteration {iteration}: worst paragraph idx={worst_idx} "
                f"score={worst_para_score:.4f} (of {len(paragraphs)} paragraphs)"
            )

            candidates = self._generate_candidates(
                worst_para,
                n=candidates_per_iteration,
                style=style,
                seed=iter_seed,
            )

            if not candidates:
                notes.append(
                    f"iteration {iteration}: paraphrase_candidates produced "
                    "nothing; stopping iteration"
                )
                break

            # Score each candidate on the paragraph alone (fast). Keep the
            # lowest. Tie-break on length-similarity so we don't drift.
            scored: list[tuple[float, str]] = []
            for candidate in candidates:
                score = self._score_whole(candidate)
                scored.append((score, candidate))
            scored.sort(key=lambda pair: pair[0])
            best_score, best_candidate = scored[0]
            notes.append(
                f"iteration {iteration}: candidate scores="
                f"{[round(s, 4) for s, _ in scored]} -> picked {best_score:.4f}"
            )

            # Only accept the swap if it actually improves the worst
            # paragraph; otherwise we are wasting iterations on noise.
            if best_score >= worst_para_score:
                notes.append(
                    f"iteration {iteration}: no candidate improved on the "
                    f"original paragraph score {worst_para_score:.4f}; stopping"
                )
                break

            paragraphs[worst_idx] = best_candidate
            current_text = "\n\n".join(paragraphs)
            current_score = self._score_whole(current_text)
            per_iteration_scores.append(current_score)
            notes.append(
                f"iteration {iteration}: whole-text score now {current_score:.4f}"
            )

            iter_elapsed = time.perf_counter() - iter_start
            if iter_elapsed > _PER_ITERATION_BUDGET_S:
                notes.append(
                    f"iteration {iteration}: exceeded {_PER_ITERATION_BUDGET_S:.0f}s "
                    f"budget ({iter_elapsed:.1f}s); will continue but warn"
                )

            if current_score <= target_ai_score:
                notes.append(
                    f"iteration {iteration}: target {target_ai_score:.4f} reached"
                )
                return IterativeResult(
                    text=current_text,
                    iterations=iteration,
                    final_score=current_score,
                    target_reached=True,
                    per_iteration_scores=per_iteration_scores,
                    target_detector=target_detector,
                    target_ai_score=target_ai_score,
                    candidates_per_iteration=candidates_per_iteration,
                    notes=notes,
                    total_time_ms=int((time.perf_counter() - t0) * 1000),
                )

        return IterativeResult(
            text=current_text,
            iterations=len(per_iteration_scores) - 1,
            final_score=current_score,
            target_reached=current_score <= target_ai_score,
            per_iteration_scores=per_iteration_scores,
            target_detector=target_detector,
            target_ai_score=target_ai_score,
            candidates_per_iteration=candidates_per_iteration,
            notes=notes,
            total_time_ms=int((time.perf_counter() - t0) * 1000),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_suite(self) -> Any:
        """Lazy-build the BenchmarkSuite once and cache it.

        Returns ``None`` if the benchmark package is not importable, in
        which case scoring degrades to "unknown" (-1.0) and the loop will
        return the baseline humanization unchanged.
        """
        if self._suite is not None:
            return self._suite
        try:
            from benchmark.benchmark_suite import BenchmarkSuite
            from benchmark.detectors import default_detectors

            if self._detectors is not None:
                # Caller specified a detector list; honor it.
                all_detectors = default_detectors(include_optional=True)
                by_name = {d.name: d for d in all_detectors}
                picked = [by_name[n] for n in self._detectors if n in by_name]
                if not picked:
                    logger.warning(
                        "none of the requested detectors %s are available; "
                        "falling back to defaults",
                        self._detectors,
                    )
                    self._suite = BenchmarkSuite()
                else:
                    self._suite = BenchmarkSuite(detectors=picked)
            else:
                self._suite = BenchmarkSuite()
            return self._suite
        except Exception as exc:
            logger.warning("BenchmarkSuite unavailable (%s); cannot score", exc)
            self._suite = None
            return None

    def _score_whole(self, text: str) -> float:
        """Score a string against the configured detector.

        Returns ``-1.0`` if scoring is unavailable so the caller can detect
        and degrade. Uses ``trusted_mean_ai_probability`` when available,
        else ``raw_mean_ai_probability``.
        """
        if not text or not text.strip():
            return -1.0
        suite = self._get_suite()
        if suite is None:
            return -1.0
        try:
            report = suite.score(text)
        except Exception as exc:
            logger.warning("scoring failed for text length %d: %s", len(text), exc)
            return -1.0
        return self._extract_score(report)

    @staticmethod
    def _extract_score(report: Any) -> float:
        """Pull a single float out of a BenchmarkReport-shaped object."""
        # Prefer trusted_mean (excludes detectors with documented bias).
        trusted = getattr(report, "trusted_mean_ai_probability", None)
        if trusted is not None:
            return float(trusted)
        raw = getattr(report, "raw_mean_ai_probability", None)
        if raw is not None:
            return float(raw)
        return -1.0

    @staticmethod
    def _call_candidates(
        cands_method: Any,
        paragraph: str,
        n: int,
        seed: int,
        style: str,
    ) -> list[str]:
        """Invoke ``paraphrase_candidates`` across the two known signatures.

        The brief specified ``paraphrase_candidates(text, n=N, **kwargs)``
        but the shipped Bet 1 implementation uses
        ``paraphrase_candidates(text, n=N, config=None)``. We probe via
        :mod:`inspect` so either signature works without a brittle
        try/except cascade. Tests use ``**kwargs``; production uses
        ``config``.
        """
        import inspect

        try:
            sig = inspect.signature(cands_method)
            params = sig.parameters
        except (TypeError, ValueError):
            params = {}  # type: ignore[assignment]

        try:
            if "config" in params:
                # Bet 1 production signature.
                return cands_method(
                    paragraph,
                    n=n,
                    config={"seed": seed, "style": style, "intensity": "aggressive"},
                )
            accepts_var_kw = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if accepts_var_kw:
                # Brief signature / test mock signature.
                return cands_method(paragraph, n=n, seed=seed, style=style)
            # Conservative: only pass what we know fits.
            return cands_method(paragraph, n=n)
        except Exception as exc:
            logger.warning("paraphrase_candidates raised %s; falling back", exc)
            return []

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split on blank lines, dropping empty trailing chunks.

        Mirrors the splitter used by ``LexicalPass`` and ``ParaphrasePass``
        so per-paragraph operations stay consistent across the codebase.
        """
        chunks = _PARAGRAPH_SPLIT.split(text)
        return [c for c in chunks if c.strip()]

    def _worst_paragraph(
        self, paragraphs: list[str]
    ) -> tuple[Optional[int], float]:
        """Find the paragraph with the highest AI probability.

        Returns ``(index, score)``. If no paragraph is scorable returns
        ``(None, -1.0)``. Single-paragraph inputs return ``(0, score)``
        without bothering to compare.
        """
        if not paragraphs:
            return None, -1.0
        if len(paragraphs) == 1:
            return 0, self._score_whole(paragraphs[0])

        worst_idx: Optional[int] = None
        worst_score = -1.0
        for i, para in enumerate(paragraphs):
            score = self._score_whole(para)
            if score < 0.0:
                continue
            if score > worst_score:
                worst_score = score
                worst_idx = i
        return worst_idx, worst_score

    def _generate_candidates(
        self,
        paragraph: str,
        *,
        n: int,
        style: str,
        seed: int,
    ) -> list[str]:
        """Ask the paraphrase pass for N stochastic candidates.

        The paraphrase agent (Bet 1) is adding
        ``ParaphrasePass.paraphrase_candidates(text, n, **kwargs) -> list[str]``
        which returns N stochastic variants. If that method is missing at
        runtime (Bet 1 not yet shipped, or a stub instance is in use) we
        fall back gracefully: we run the deterministic light paraphrase
        once and return ``[result]`` as a single-candidate list. The loop
        will then exit on the next "no improvement" check.
        """
        # Find the paraphrase pass on the base pipeline.
        para_pass = None
        for p in self.base_pipeline.passes:
            if getattr(p, "pass_name", "") == "paraphrase":
                para_pass = p
                break

        if para_pass is None:
            logger.warning(
                "no paraphrase pass found on base pipeline; cannot generate candidates"
            )
            return []

        cands_method = getattr(para_pass, "paraphrase_candidates", None)
        if callable(cands_method):
            candidates = self._call_candidates(cands_method, paragraph, n, seed, style)
            cleaned = [c for c in (candidates or []) if c and c.strip()]
            if cleaned:
                return cleaned
            logger.warning(
                "paraphrase_candidates returned empty list; falling back to "
                "deterministic single-candidate"
            )

        # Fallback: one deterministic light paraphrase. Better than nothing,
        # and lets the loop exit cleanly on the no-improvement check rather
        # than crashing.
        try:
            single = para_pass.apply(
                paragraph,
                {
                    "intensity": "aggressive",
                    "preserve_voice": True,
                    "seed": seed,
                    "mode": "light",
                },
            )
        except Exception as exc:
            logger.warning("fallback paraphrase apply raised %s", exc)
            return []
        if single and single.strip() and single != paragraph:
            return [single]
        return []


__all__ = ["IterativeHumanizer", "IterativeResult"]
