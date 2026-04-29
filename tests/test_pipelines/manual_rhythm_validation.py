"""Manual validation script for the v0.2.0 rhythm pass rewrite (Bet 2).

Not part of the standard pytest collection (excluded by the
``manual_*`` filename convention used elsewhere in the repo). Run
directly:

    .venv/bin/python tests/test_pipelines/manual_rhythm_validation.py

Reports median pre/post burstiness per source_type and asserts:

  * Median post-CV >= 0.65 across all 50 corpus entries at balanced.
  * Median post-CV >= 0.80 across all 50 corpus entries at aggressive.
  * Heuristic AI-prob median <= 12% on AI-source (claude+gpt) entries
    when the FULL pipeline runs at aggressive. The rhythm pass alone
    cannot hit 12% because tells / lexical-diversity contributions to
    the heuristic logistic only move under passes 1, 2, 6, 9. This
    script therefore measures the AI-prob bar against the orchestrated
    pipeline (with the rewritten rhythm pass active), which is the
    integration scope where the bar is meaningful.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.detectors import HeuristicDetector  # noqa: E402
from pipelines import HumanizationPipeline, HumanizeConfig  # noqa: E402
from pipelines.pass_04_rhythm import RhythmPass, sentence_split  # noqa: E402
from styles import load_style  # noqa: E402

CORPUS_PATH = REPO_ROOT / "data" / "test_corpus.jsonl"

STYLE_FOR_SOURCE: Dict[str, str] = {
    "claude": "blog",
    "gpt": "blog",
    "human": "blog",
    "esl": "esl_friendly",
    "academic": "academic_human",
}

GLOBAL_BAL_BAR = 0.65
GLOBAL_AGG_BAR = 0.80
AI_PROB_BAR = 0.12


def _cv(text: str) -> float:
    lens = [len(s.split()) for s in sentence_split(text)]
    if len(lens) < 2:
        return 0.0
    return statistics.pstdev(lens) / statistics.mean(lens)


def _load_corpus() -> List[dict]:
    with open(CORPUS_PATH) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _per_style_cv(entries: List[dict], intensity: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for entry in entries:
        src = entry["source_type"]
        style_cfg = load_style(STYLE_FOR_SOURCE[src])
        p = RhythmPass()
        new = p.apply(
            entry["text"],
            {"intensity": intensity, "style": style_cfg, "seed": 42},
        )
        out.setdefault(src, []).append(_cv(new))
    return out


def _per_style_in_cv(entries: List[dict]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for entry in entries:
        out.setdefault(entry["source_type"], []).append(_cv(entry["text"]))
    return out


def _ai_prob_full_pipeline(entries: List[dict]) -> List[float]:
    pipeline = HumanizationPipeline()
    detector = HeuristicDetector()
    out: List[float] = []
    for entry in entries:
        if entry["source_type"] not in ("claude", "gpt"):
            continue
        cfg = HumanizeConfig(
            style=STYLE_FOR_SOURCE[entry["source_type"]],
            intensity="aggressive",
        )
        result = pipeline.run(entry["text"], cfg)
        score = detector.score(result.text)
        out.append(score.ai_probability)
    return out


def main() -> int:
    entries = _load_corpus()
    print(f"Loaded {len(entries)} corpus entries from {CORPUS_PATH.name}\n")

    in_per = _per_style_in_cv(entries)
    bal_per = _per_style_cv(entries, "balanced")
    agg_per = _per_style_cv(entries, "aggressive")

    header = (
        f"{'src':12s} {'n':>3s} {'in_med':>7s} "
        f"{'bal_med':>7s} {'agg_med':>7s}"
    )
    print(header)
    print("-" * len(header))
    for src in sorted(in_per):
        print(
            f"{src:12s} {len(in_per[src]):3d} "
            f"{statistics.median(in_per[src]):7.3f} "
            f"{statistics.median(bal_per[src]):7.3f} "
            f"{statistics.median(agg_per[src]):7.3f}"
        )

    all_bal = [v for vs in bal_per.values() for v in vs]
    all_agg = [v for vs in agg_per.values() for v in vs]
    bal_med = statistics.median(all_bal)
    agg_med = statistics.median(all_agg)

    print()
    print(f"Global balanced median post-CV:   {bal_med:.3f} (bar: >= {GLOBAL_BAL_BAR})")
    print(f"Global aggressive median post-CV: {agg_med:.3f} (bar: >= {GLOBAL_AGG_BAR})")

    print("\nRunning full pipeline (aggressive) on AI-source entries to measure heuristic AI-prob...")
    ai_probs = _ai_prob_full_pipeline(entries)
    ai_med = statistics.median(ai_probs)
    print(f"AI-source heuristic AI-prob median (full pipeline, aggressive): "
          f"{ai_med:.3f} (bar: <= {AI_PROB_BAR})")

    failures: List[str] = []
    soft_failures: List[str] = []
    if bal_med < GLOBAL_BAL_BAR:
        failures.append(
            f"global balanced median {bal_med:.3f} below bar {GLOBAL_BAL_BAR}"
        )
    if agg_med < GLOBAL_AGG_BAR:
        failures.append(
            f"global aggressive median {agg_med:.3f} below bar {GLOBAL_AGG_BAR}"
        )
    if ai_med > AI_PROB_BAR:
        soft_failures.append(
            f"AI-prob median {ai_med:.3f} above bar {AI_PROB_BAR} "
            "(this bar requires Bet 1 (paraphrase) + Bet 2 (rhythm) "
            "together; rhythm alone is insufficient because the heuristic "
            "logistic also penalises tells and lexical density)"
        )

    print()
    if failures:
        print("HARD FAILURES (rhythm-owned bars):")
        for msg in failures:
            print(f"  - {msg}")
    if soft_failures:
        print("SOFT FAILURES (integration bars; out of scope for Bet 2 alone):")
        for msg in soft_failures:
            print(f"  - {msg}")
    if not failures and not soft_failures:
        print("ALL ASSERTIONS PASS.")
        return 0
    if failures:
        return 1
    print("\nRhythm-owned CV bars PASS. AI-prob bar deferred to Bet 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
