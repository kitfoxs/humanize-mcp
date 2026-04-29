# HumanizeMCP

> An open-source Model Context Protocol server that rewrites AI-generated prose as human-authored, with first-class support for the writers detectors get wrong.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastMCP](https://img.shields.io/badge/MCP-FastMCP-purple.svg)](https://github.com/jlowin/fastmcp)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#roadmap)

---

## Why this exists

Stanford researchers tested seven major AI detectors on TOEFL essays written by non-native English speakers and **61.3% were classified as AI-generated**. Native US 8th-graders writing the same kind of essays were classified correctly. That single result, published in *Patterns* (Cell Press, 2023), is the reason this project exists.

The same statistical signature that detectors call "AI" is also the signature of:

- **Neurodivergent writers**, especially autistic adults, whose prose tends toward lower lexical diversity, more concrete vocabulary, and repeated phrasal structures.
- **ESL writers**, whose textbook-derived grammar reads as unnaturally regular.
- **Academics in formal disciplines**, whose register is by design uniform and discourse-marker dense.

A 2026 Springer study measured Originality.ai's real-world false positive rate on academic writing at 14%, an order of magnitude above the company's stated rate. Independent testing of ZeroGPT in 2026 measured 26.4% false positives on student essays and over 21% on ESL writers.

These detectors are deployed in academic, employment, publishing, and visa workflows. HumanizeMCP gives the affected writers a tool to defend themselves, built on the published peer-reviewed evasion literature rather than the closed-source heuristics shipped by commercial humanizers.

The full evidence base is in [`research/`](research/). Read [`docs/ETHICS.md`](docs/ETHICS.md) for the line we draw between legitimate accessibility use and academic fraud.

---

## What it is

HumanizeMCP is a [Model Context Protocol](https://modelcontextprotocol.io/) server. Any MCP-aware client (Claude Code, GitHub Copilot CLI, Continue, Zed, Cursor, and others) can call its tools. It runs locally, it ships no telemetry, and the only network calls it makes are the optional Hugging Face model downloads on first run.

**Core capabilities:**

- A **9-pass pipeline** (preprocess, surface-tell substitution, watermark scrub, stylometric smoothing, controlled paraphrase, detector-guided iteration, optional back-translation, voice-preservation, verification) following the design in `research/06_implementation_recommendations.md`.
- **Style presets** (`casual`, `blog`, `academic`, `esl`, `neurodivergent`, `preserve`, ...) configurable per call.
- **Local detector benchmarking** against open models (RoBERTa-OpenAI-detector, Fast-DetectGPT, Binoculars) plus optional adapters for commercial APIs when the user supplies their own key.
- A **verify-and-iterate loop** (`humanize_and_verify`) that rehumanizes until a target detector score is met or a budget is exhausted.

---

## Quickstart

### Install

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e .

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The first call that touches a Hugging Face model will download weights to `~/.cache/huggingface/`. The base detector (`roberta-base-openai-detector`) is ~500MB.

### Register with your MCP client

**Claude Code** (`~/.config/claude-code/mcp.json` or equivalent):

```json
{
  "mcpServers": {
    "humanize": {
      "command": "humanize-mcp"
    }
  }
}
```

**GitHub Copilot CLI** (`~/.copilot/mcp-config.json`):

```json
{
  "mcpServers": {
    "humanize": {
      "command": "humanize-mcp"
    }
  }
}
```

If you installed into a virtualenv, point `command` at the absolute path of the script (`/path/to/.venv/bin/humanize-mcp`).

### First call

From any MCP client:

```
> use humanize on the following text with style="academic":
>
> In this paper, we delve into the multifaceted intricacies of consensus
> mechanisms — leveraging robust theoretical frameworks to underscore the
> pivotal role of decentralized governance in fostering trust.
```

The server returns a rewrite that drops the `delve` cluster, breaks up the em dash, varies sentence length, and (if `humanize_and_verify` is used) confirms with a local detector that the rewrite scores below the requested AI threshold.

---

## Tool reference

All tools are MCP-callable. Type signatures are enforced by Pydantic.

### `humanize(text, style="default", preserve_voice=True, intensity=0.7) -> str`

Run the full humanization pipeline on `text`. `intensity` is in [0, 1] and maps to the minimal/balanced/aggressive levers from the research dossier.

```python
humanize(
    text="The implementation leverages a robust framework...",
    style="academic",
    preserve_voice=True,
    intensity=0.7,
)
# -> "The implementation uses a sturdy framework..."
```

### `detect_tells(text) -> TellsReport`

Locate AI tells without rewriting. Returns line numbers, character offsets, severity (1-5 stars matching `research/02_ai_tells_catalog.md`), and substitution suggestions.

```python
detect_tells("We delve into the intricate tapestry of...")
# -> TellsReport(
#      tell_count=3,
#      tells=[TellLocation(category="excess_vocabulary", fragment="delve",
#                          line=1, char_start=3, char_end=8, severity=5,
#                          suggestion="examine"),
#             ...],
#      summary={"excess_vocabulary": 3},
#    )
```

### `score_humanity(text, detectors=["roberta-base"]) -> HumanityReport`

Run one or more local detectors and return per-detector probabilities plus an aggregate. Verdict is one of `"human"`, `"uncertain"`, `"ai"`, `"unknown"`.

```python
score_humanity(text, detectors=["roberta-base", "fast_detect_gpt"])
# -> HumanityReport(
#      detector_scores=[DetectorScore(detector="roberta-base",
#                                     probability_ai=0.87, ...),
#                       DetectorScore(detector="fast_detect_gpt",
#                                     probability_ai=0.91, ...)],
#      aggregate_probability_ai=0.89,
#      verdict="ai",
#    )
```

### `apply_style(text, style) -> str`

Pure register translation, no humanization. Useful when you want academic-to-blog or formal-to-casual without touching AI tells.

### `list_styles() -> list[str]`

Enumerate the style presets currently registered in `styles/`.

### `humanize_and_verify(text, style="default", target_ai_score=0.3, max_iterations=3) -> VerifyResult`

Humanize, score, rehumanize at higher intensity if the score is above target, repeat until target is reached or `max_iterations` is hit. Returns the final text along with before/after scores and a per-iteration log.

```python
result = humanize_and_verify(text, target_ai_score=0.2, max_iterations=5)
print(result.final_score.aggregate_probability_ai)  # e.g. 0.18
print(result.target_reached)                        # True
```

---

## Style presets

Defined in `styles/`. The set evolves as the project matures; call `list_styles()` for the live list. The MVP ships:

| Preset | Designed for | Notes |
|---|---|---|
| `default` | General-purpose rewriting | Balanced intensity, conservative voice preservation |
| `casual` | Personal email, social posts | Strips formality, raises burstiness |
| `blog` | Long-form personal writing | Maintains paragraph rhythm |
| `academic` | Conference papers, theses | Preserves discourse markers, drops `delve` cluster |
| `esl` | L2 English writers | Preserves simpler syntax, removes detector-flagged surface tells |
| `neurodivergent` | Autistic / ND writers | Preserves repeated phrasal structures, low lexical diversity |
| `preserve` | "Don't change my voice" | Surface tells only, no paraphrase passes |

The preset schema and how to add your own is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Architecture

```
+------------------------+
|   MCP Client (Claude   |
|   Code, Copilot, ...)  |
+-----------+------------+
            | stdio (MCP)
            v
+------------------------+
|  server.py (FastMCP)   |    <-- this package
+-----+-----+-----+------+
      |     |     |
      v     v     v
+--------+ +--------+ +----------+
| pipe-  | | styles | | bench-   |
| lines/ | |        | | mark/    |
+--------+ +--------+ +----------+
      |                 |
      v                 v
+----------------------------+
| 9-pass pipeline:           |
| 1. preprocess              |
| 2. surface tells           |
| 3. watermark scrub         |
| 4. stylometric smoothing   |
| 5. DIPPER paraphrase       |
| 6. detector-guided loop    |
| 7. back-translation (opt)  |
| 8. style transfer          |
| 9. verification            |
+----------------------------+
```

Full architecture and extension points: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Benchmarks

The benchmark harness lives in `benchmark/` and is currently being built out. The headline numbers below are placeholders that the Benchmark Engineer will replace with measured values from the RAID test set.

| Detector | Pre-humanization (mean P(AI)) | Post-humanization (mean P(AI)) | Reduction |
|---|---|---|---|
| RoBERTa-OpenAI | _TBD_ | _TBD_ | _TBD_ |
| Fast-DetectGPT | _TBD_ | _TBD_ | _TBD_ |
| Binoculars | _TBD_ | _TBD_ | _TBD_ |
| GPTZero (API) | _TBD_ | _TBD_ | _TBD_ |
| Originality.ai (API) | _TBD_ | _TBD_ | _TBD_ |

Reproduce locally:

```bash
python -m benchmark.run --dataset raid_subset --pipeline default
```

---

## Ethical use

This tool is built for, and documented around, legitimate accessibility and equity use:

- A neurodivergent writer whose own writing is being misclassified.
- An ESL writer whose academic submission is being flagged by a tool that has been independently measured at 21-61% false positive rates on their cohort.
- An academic whose formal-discipline prose is triggering a brittle detector that should not have been deployed in their institution's workflow.
- A journalist or essayist trying to escape detector-driven deplatforming.

This tool is **not** built for, and the maintainers will not assist with:

- Submitting LLM-generated work as your own writing in a graded academic assignment that requires original authorship.
- Generating disinformation, spam, or content designed to deceive at scale.
- Defeating detectors in safety-critical contexts (medical advice, legal filings, security-relevant disclosure).

The technical pipelines support both populations identically; that is unavoidable, and is the same ethical posture taken by Tor, Signal, and every paraphrasing tool that has ever existed. Read the full statement in [`docs/ETHICS.md`](docs/ETHICS.md).

We do **not** ship homoglyph attacks, watermark stripping for closed models, zero-width character injection, or any technique that breaks accessibility tooling such as screen readers. Those are exploits, not humanization.

---

## Citation

If you use HumanizeMCP in research, please cite the dossier and the project:

```bibtex
@software{humanize_mcp_2026,
  author       = {Olivas, Kit},
  title        = {HumanizeMCP: an open-source MCP server for accessibility-first
                  AI-text humanization},
  year         = {2026},
  url          = {https://github.com/kitfoxs/humanize-mcp},
  version      = {0.1.0},
  note         = {Research dossier in research/, evidence base in
                  research/references.bib}
}
```

The research dossier in `research/` is itself a citable document and contains the full bibliography of the peer-reviewed work this project builds on.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Acknowledgments

Built by **Kit Olivas** ([@kitfoxs](https://github.com/kitfoxs)) and **Ada Marie**.

The research dossier draws on work by Liang et al. (Stanford, 2023), Dugan et al. (RAID, 2024), Krishna et al. (DIPPER, 2023), Cheng et al. (Adversarial Paraphrasing, 2025), David & Gervais (AuthorMist, 2025), Kobak et al. (vocabulary shift study, 2024), Juzek & Ward (focal-word analysis, 2024), and many others. Full citations in `research/references.bib`.

Special thanks to the writers, students, and academics whose published accounts of false-positive harm motivated this work.
