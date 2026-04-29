# HumanizeMCP Benchmark Module

The benchmark module scores text against multiple AI-text detectors and
produces structured before/after comparison reports for the rest of the
HumanizeMCP project (the MCP server, the pipeline harness, and the
README's published numbers).

It is deliberately conservative: every detector reports its own score
along with documented bias caveats, and aggregates are computed two ways
so a single biased detector cannot dominate the headline number.

## Quickstart

```bash
cd /Users/kit/projects/humanize-mcp
source .venv/bin/activate

# Smoke check — uses heuristic + RoBERTa, runs over a 5-entry subset.
python benchmark/run_benchmark.py --quick

# Full run on the whole corpus.
python benchmark/run_benchmark.py

# Full run including optional zero-shot scorers (downloads more models).
python benchmark/run_benchmark.py --include-optional

# Score a single string from Python.
python -c "from benchmark import score_text; print(score_text('your text here').model_dump_json(indent=2))"
```

Reports land in `data/benchmark_results_<timestamp>.json` and a markdown
summary is printed to stdout.

## What's wrapped

| Detector | Model | Tier | Notes |
|---|---|---|---|
| `HeuristicDetector` | none (pure Python) | heuristic | Burstiness + lexical diversity + curated tells list. Always available. |
| `RobertaOpenAIDetector` | `roberta-base-openai-detector` | classifier | Canonical academic baseline. Documented bias against modern LLM output. |
| `ChatGPTRobertaDetector` | `Hello-SimpleAI/chatgpt-detector-roberta` | classifier | HC3-trained; useful counterpart to the RoBERTa-base one. |
| `DesklibAIDetector` | `desklib/ai-text-detector-v1.01` | classifier | Optional. Custom architecture; may need `trust_remote_code`. |
| `FastDetectGPT` | reference LM (default `gpt2`) | zero-shot | CPU-friendly variant of Bao et al. 2023; not the canonical impl. |
| `BinocularsDetector` | observer/performer pair (default `gpt2`/`distilgpt2`) | zero-shot | CPU-friendly variant of Hans et al. 2024; calibration approximate. |

By default `BenchmarkSuite()` loads only the always-on tier (heuristic +
the two small RoBERTa classifiers). Pass `include_optional=True` (or
`--include-optional` on the CLI) to add the zero-shot detectors and
Desklib.

All detectors load their models lazily on first `score()` call so
`from benchmark import ...` is cheap.

## Public API

```python
from benchmark import BenchmarkSuite, score_text, pre_commit_check

# Quick one-shot
report = score_text("Some text to score.")
print(report.verdict)
print(report.raw_mean_ai_probability)

# Suite with persistent detector state (avoids reloading models)
suite = BenchmarkSuite(include_optional=False)
report = suite.score("text")
comparison = suite.compare(before_text="raw", after_text="humanized")

# Pre-commit lint
result = pre_commit_check("text", threshold=0.6)
print(result.passed, result.suggestions)
```

## Calibration ethics

The research dossier (`research/01_detector_landscape.md`,
`research/03_human_diversity.md`) documents that several detectors —
including `roberta-base-openai-detector` — have severe false positive
rates on real human writing, ESL writing, and text that has been even
trivially paraphrased. The benchmark suite handles this in three ways:

1. Per-detector scores are kept separate and surfaced individually;
   nothing is silently averaged into a single "verdict" at the
   detector layer.
2. `BenchmarkReport.bias_warnings` aggregates every detector's
   documented caveats so any UI presenting the result can show them.
3. Aggregate AI-probability is computed two ways:
   * `raw_mean_ai_probability` — unweighted mean of all detectors.
   * `trusted_mean_ai_probability` — mean restricted to detectors with
     no documented bias caveats. This is the recommended headline number.

The heuristic detector is included as a transparent baseline. It uses
public, inspectable features (sentence-length variance + a curated tells
list); it makes no peer-reviewed claims about itself.

## Adding a new detector

1. Subclass `benchmark.detectors.Detector`.
2. Set `name`, `tier` (`TIER_HEURISTIC`, `TIER_CLASSIFIER`, or
   `TIER_ZERO_SHOT`), and a `bias_notes: ClassVar[list[str]]` listing any
   documented calibration caveats.
3. Implement `_score_impl(text) -> DetectorScore`. Use
   `DetectorScore.from_ai_prob(...)` to build the return value; it will
   handle verdict thresholds and confidence calculation for you.
4. Load any heavy model lazily inside `_score_impl` (or a `_ensure_loaded`
   helper) — the import-time cost of `benchmark` should stay near zero.
5. Add the detector to `default_detectors()` if it should run by default.

## Test corpus

`benchmark.test_corpus` writes a 50-entry JSONL to
`data/test_corpus.jsonl`:

* 10 synthetic Claude-style paragraphs
* 10 synthetic GPT-style paragraphs
* 10 short pre-1924 public-domain excerpts (a few are 20th-century
  fair-use samples flagged in their attribution; replace before any
  redistribution)
* 10 synthetic ESL-style paragraphs
* 10 synthetic academic-abstract paragraphs

Synthetic samples are labeled as such in their `source_attribution`. The
synthetic ground-truth is good enough for calibration smoke tests but is
not the same as gold-standard human-vs-AI provenance and reports note
this.

## License notes for wrapped HuggingFace models

| Model | License |
|---|---|
| `roberta-base-openai-detector` | MIT |
| `Hello-SimpleAI/chatgpt-detector-roberta` | Apache-2.0 (per HF model card) |
| `desklib/ai-text-detector-v1.01` | Check upstream model card before redistribution |
| `gpt2`, `distilgpt2` | MIT / Apache-2.0 |
| `EleutherAI/gpt-neo-2.7B` | MIT (used as optional Fast-DetectGPT reference) |

The benchmark module itself does not redistribute model weights; it
downloads them via `transformers`/`huggingface_hub` on first use.
