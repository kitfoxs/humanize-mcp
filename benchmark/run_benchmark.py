"""CLI: run the benchmark suite over the test corpus and emit a JSON report.

Usage:
    python benchmark/run_benchmark.py [--quick] [--include-optional]
        [--output PATH] [--limit N] [--source-types TYPE,TYPE]

The runner:

1. Loads the test corpus from ``data/test_corpus.jsonl`` (writing it on
   demand if missing).
2. For each entry: scores the raw text with :class:`BenchmarkSuite`.
3. Attempts to import a humanization pipeline from ``pipelines.run`` (or
   ``humanize.pipelines.run`` etc.). If any pipeline ``humanize(text)`` is
   available, runs it on the raw text and scores the result; otherwise
   logs a one-time skip notice and continues with raw scores only.
4. Writes the full report as JSON to
   ``data/benchmark_results_<timestamp>.json``.
5. Prints a markdown summary table to stdout suitable for pasting into
   the README.

``--quick`` runs against a 5-entry subset and uses only the heuristic
plus the RoBERTa base classifier (no zero-shot scorers, no Desklib).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# Make `from benchmark import ...` work when this script is invoked directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.benchmark_suite import (  # noqa: E402
    BenchmarkReport,
    BenchmarkSuite,
    ComparisonReport,
)
from benchmark.detectors import (  # noqa: E402
    HeuristicDetector,
    RobertaOpenAIDetector,
    default_detectors,
)
from benchmark.test_corpus import CORPUS_PATH, CorpusEntry, load_corpus  # noqa: E402

LOGGER = logging.getLogger("benchmark.run")

DATA_DIR = ROOT / "data"


def _try_load_pipeline() -> Optional[Callable[[str], str]]:
    """Best-effort import of a humanization pipeline.

    Looks at a small set of likely module paths. If none provides a
    callable named ``humanize`` (or a class ``Pipeline`` with a
    ``humanize`` method), returns ``None`` and the caller should skip
    the after-comparison portion of the run.
    """
    candidates = (
        ("humanize_mcp.pipelines", "humanize"),
        ("pipelines", "humanize"),
        ("pipelines.run", "humanize"),
        ("pipelines.default", "humanize"),
    )
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            continue
        fn = getattr(module, attr, None)
        if callable(fn):
            LOGGER.info("Loaded humanization pipeline from %s.%s", module_name, attr)
            return fn  # type: ignore[return-value]
        # Also accept a Pipeline class with humanize() method.
        cls = getattr(module, "Pipeline", None)
        if cls is not None:
            try:
                inst = cls()
                method = getattr(inst, "humanize", None)
                if callable(method):
                    LOGGER.info(
                        "Loaded humanization pipeline class from %s.Pipeline", module_name
                    )
                    return method  # type: ignore[return-value]
            except Exception:  # noqa: BLE001
                continue
    return None


@dataclass
class _EntryResult:
    entry: CorpusEntry
    before: BenchmarkReport
    after: Optional[BenchmarkReport] = None
    comparison: Optional[ComparisonReport] = None
    pipeline_error: Optional[str] = None


@dataclass
class _RunSummary:
    timestamp_iso: str
    detectors: list[str]
    n_entries: int
    pipeline_used: bool
    pipeline_skip_reason: Optional[str] = None
    per_source_summary: dict[str, dict] = field(default_factory=dict)
    overall_mean_before: float = 0.0
    overall_mean_after: Optional[float] = None
    overall_mean_delta: Optional[float] = None
    duration_seconds: float = 0.0


def _summarize(results: list[_EntryResult]) -> dict[str, dict]:
    """Compute per-source-type aggregate statistics."""
    by_source: dict[str, list[_EntryResult]] = {}
    for r in results:
        by_source.setdefault(r.entry.source_type, []).append(r)
    summary: dict[str, dict] = {}
    for source, items in sorted(by_source.items()):
        before_scores = [
            r.before.trusted_mean_ai_probability
            if r.before.trusted_mean_ai_probability is not None
            else r.before.raw_mean_ai_probability
            for r in items
        ]
        after_scores = [
            (
                r.after.trusted_mean_ai_probability
                if r.after.trusted_mean_ai_probability is not None
                else r.after.raw_mean_ai_probability
            )
            for r in items
            if r.after is not None
        ]
        block = {
            "n": len(items),
            "mean_ai_before": round(sum(before_scores) / len(before_scores), 4),
            "max_ai_before": round(max(before_scores), 4),
            "min_ai_before": round(min(before_scores), 4),
        }
        if after_scores:
            block["mean_ai_after"] = round(sum(after_scores) / len(after_scores), 4)
            block["mean_delta"] = round(block["mean_ai_before"] - block["mean_ai_after"], 4)
        summary[source] = block
    return summary


def _markdown_summary(run: _RunSummary) -> str:
    """Render the run as a markdown table for stdout / README."""
    lines: list[str] = []
    lines.append(f"### Benchmark run {run.timestamp_iso}")
    lines.append("")
    lines.append(f"- Detectors: {', '.join(run.detectors)}")
    lines.append(f"- Corpus entries: {run.n_entries}")
    if run.pipeline_used:
        lines.append("- Pipeline: in use")
    else:
        reason = run.pipeline_skip_reason or "no pipeline available"
        lines.append(f"- Pipeline: skipped ({reason})")
    lines.append(f"- Duration: {run.duration_seconds:.1f}s")
    lines.append("")
    if run.pipeline_used:
        header = "| Source | N | Mean AI (before) | Mean AI (after) | Mean delta |"
        sep = "|---|---:|---:|---:|---:|"
    else:
        header = "| Source | N | Mean AI | Min | Max |"
        sep = "|---|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for source, block in run.per_source_summary.items():
        if run.pipeline_used:
            lines.append(
                f"| {source} | {block['n']} | {block['mean_ai_before']:.3f} | "
                f"{block.get('mean_ai_after', float('nan')):.3f} | "
                f"{block.get('mean_delta', float('nan')):+.3f} |"
            )
        else:
            lines.append(
                f"| {source} | {block['n']} | {block['mean_ai_before']:.3f} | "
                f"{block['min_ai_before']:.3f} | {block['max_ai_before']:.3f} |"
            )
    lines.append("")
    if run.pipeline_used and run.overall_mean_delta is not None:
        lines.append(
            f"**Overall mean AI score:** {run.overall_mean_before:.3f} -> "
            f"{run.overall_mean_after:.3f} ({run.overall_mean_delta:+.3f})"
        )
    else:
        lines.append(f"**Overall mean AI score (before):** {run.overall_mean_before:.3f}")
    return "\n".join(lines)


def _result_to_dict(result: _EntryResult) -> dict:
    """Serialize an :class:`_EntryResult` to a JSON-friendly dict."""
    payload = {
        "id": result.entry.id,
        "source_type": result.entry.source_type,
        "source_attribution": result.entry.source_attribution,
        "before": result.before.model_dump(),
    }
    if result.after is not None:
        payload["after"] = result.after.model_dump()
    if result.comparison is not None:
        payload["comparison"] = result.comparison.model_dump()
    if result.pipeline_error:
        payload["pipeline_error"] = result.pipeline_error
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a 5-entry subset using only the heuristic + RoBERTa "
            "classifier. Used for smoke validation."
        ),
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include zero-shot detectors (Fast-DetectGPT, Binoculars) and Desklib.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_PATH,
        help=f"Path to the corpus JSONL (default: {CORPUS_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to write the JSON results. Defaults to "
            "data/benchmark_results_<timestamp>.json"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run on at most N entries (for debugging).",
    )
    parser.add_argument(
        "--source-types",
        type=str,
        default=None,
        help=(
            "Comma-separated source types to include "
            "(claude,gpt,human,esl,academic). Default: all."
        ),
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Do not attempt to load a humanization pipeline; only score raw text.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.quick:
        suite = BenchmarkSuite(
            detectors=[HeuristicDetector(), RobertaOpenAIDetector()],
        )
    else:
        suite = BenchmarkSuite(
            detectors=default_detectors(include_optional=args.include_optional)
        )

    source_types = (
        tuple(s.strip() for s in args.source_types.split(",") if s.strip())
        if args.source_types
        else None
    )
    entries = load_corpus(args.corpus, source_types=source_types)
    if args.quick and not args.limit:
        # Take a representative slice (one per source type, up to 5).
        seen_types: set[str] = set()
        slice_: list[CorpusEntry] = []
        for e in entries:
            if e.source_type not in seen_types:
                slice_.append(e)
                seen_types.add(e.source_type)
            if len(slice_) >= 5:
                break
        entries = slice_
    if args.limit is not None:
        entries = entries[: args.limit]

    pipeline_fn: Optional[Callable[[str], str]] = None
    pipeline_skip_reason: Optional[str] = None
    if args.no_pipeline:
        pipeline_skip_reason = "disabled by --no-pipeline"
    else:
        pipeline_fn = _try_load_pipeline()
        if pipeline_fn is None:
            pipeline_skip_reason = "no humanize() callable found in pipelines/"

    results: list[_EntryResult] = []
    t0 = time.time()
    for i, entry in enumerate(entries, start=1):
        print(
            f"[{i}/{len(entries)}] {entry.id} ({entry.source_type})",
            file=sys.stderr,
        )
        before = suite.score(entry.text)
        result = _EntryResult(entry=entry, before=before)
        if pipeline_fn is not None:
            try:
                after_text = pipeline_fn(entry.text)
                if isinstance(after_text, str) and after_text.strip():
                    after = suite.score(after_text)
                    comparison = suite.compare(entry.text, after_text)
                    result.after = after
                    result.comparison = comparison
                else:
                    result.pipeline_error = (
                        "pipeline returned empty / non-string result"
                    )
            except Exception as exc:  # noqa: BLE001
                result.pipeline_error = repr(exc)
        results.append(result)

    duration = time.time() - t0
    per_source = _summarize(results)
    pipeline_used = bool(pipeline_fn) and any(r.after for r in results)

    overall_before = sum(
        b
        for r in results
        for b in [
            r.before.trusted_mean_ai_probability
            if r.before.trusted_mean_ai_probability is not None
            else r.before.raw_mean_ai_probability
        ]
    ) / max(1, len(results))
    overall_after: Optional[float] = None
    overall_delta: Optional[float] = None
    if pipeline_used:
        afters = [
            (
                r.after.trusted_mean_ai_probability
                if r.after.trusted_mean_ai_probability is not None
                else r.after.raw_mean_ai_probability
            )
            for r in results
            if r.after is not None
        ]
        if afters:
            overall_after = sum(afters) / len(afters)
            overall_delta = overall_before - overall_after

    summary = _RunSummary(
        timestamp_iso=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        detectors=suite.detector_names,
        n_entries=len(results),
        pipeline_used=pipeline_used,
        pipeline_skip_reason=pipeline_skip_reason if not pipeline_used else None,
        per_source_summary=per_source,
        overall_mean_before=round(overall_before, 4),
        overall_mean_after=round(overall_after, 4) if overall_after is not None else None,
        overall_mean_delta=round(overall_delta, 4) if overall_delta is not None else None,
        duration_seconds=round(duration, 2),
    )

    output_path = args.output
    if output_path is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = DATA_DIR / f"benchmark_results_{ts}.json"

    output_payload = {
        "summary": {
            "timestamp_iso": summary.timestamp_iso,
            "detectors": summary.detectors,
            "n_entries": summary.n_entries,
            "pipeline_used": summary.pipeline_used,
            "pipeline_skip_reason": summary.pipeline_skip_reason,
            "per_source_summary": summary.per_source_summary,
            "overall_mean_before": summary.overall_mean_before,
            "overall_mean_after": summary.overall_mean_after,
            "overall_mean_delta": summary.overall_mean_delta,
            "duration_seconds": summary.duration_seconds,
        },
        "entries": [_result_to_dict(r) for r in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False))

    print(_markdown_summary(summary))
    print(f"\nFull report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
