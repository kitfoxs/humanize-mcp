# 06 — Implementation Recommendations for the Pipeline Builder Agent

This file is the bridge between the research dossier (sections 00-05) and code. It specifies which techniques to implement, in what order, with which Python libraries, what HuggingFace models to use, and how to wire the components together. Decisions in this file should be revised only with explicit reference to the supporting research in earlier sections.

---

## 1. Architectural overview

HumanizeMCP exposes a single MCP tool family with the following structure:

```
humanize(
  text: str,
  preset: Optional[Literal["casual","blog","literary","academic","email","esl","neurodivergent","preserve"]] = None,
  intensity: Literal["minimal","balanced","aggressive"] = "balanced",
  preserve_features: Optional[List[Feature]] = None,
  target_detectors: Optional[List[str]] = None,
  style_corpus_path: Optional[str] = None,
  return_diff: bool = False,
  return_scores: bool = True,
) -> HumanizeResult
```

Internally the call dispatches to a configurable pipeline. The pipeline is a sequence of named passes. Each pass:
- Takes the current text + a context object (preset, intensity, preserve list, scores so far).
- Returns transformed text + a record of what it changed.
- Can be skipped by configuration.
- Has its own per-preset defaults.

This makes the pipeline observable, testable, and ablatable. The MCP `humanize_dry_run` tool returns the pipeline plan without executing it; `humanize_explain` returns the full diff and per-pass scores.

---

## 2. The pipeline passes (recommended order)

Implement in this order. Each pass should be independently shippable, so the project can ship MVP at pass 4 and grow.

### Pass 1: Preprocess and normalize

**Purpose:** Strip artifacts that would corrupt downstream passes.

- Unicode NFKC normalization.
- Strip zero-width characters (U+200B, U+200C, U+200D, U+FEFF).
- Strip bidi-override characters (U+202A-202E, U+2066-2069).
- Normalize curly quotes to straight quotes if `intensity != "minimal"` (so re-stylization passes can re-introduce them naturally).
- Strip Markdown formatting (headers, bold, italic, list markers) if pasted from chat output.
- Detect and unwrap "Great question!" / "Certainly!" / "I'd be happy to..." opening preambles.
- Detect and strip "Let me know if you have follow-up questions!" / "I hope this helps!" closing offers.
- Detect and strip "As an AI..." disclosures.

**Library:** Python stdlib `unicodedata`, `re`, `regex`. Optionally `mistune` or `markdown-it-py` for robust Markdown parsing.

**Cost:** Microseconds; deterministic.

**Tests required:** Round-trip safety on text that doesn't contain these patterns. Idempotency.

### Pass 2: Surface-tell substitution

**Purpose:** Remove the cosmetic AI fingerprint at the lexical and punctuation level. This is the single highest-ROI pass — cheap, fast, and removes the most-recognizable AI tells.

- **Em-dash substitution.** Replace em dashes (—, U+2014) according to context: parenthetical use → comma pair or parentheses; explanation-introducer use → colon or period; range use (rare in prose) → en dash. Keep one or two if the text already had varied punctuation, to avoid uniformly stripped output.
- **Excess-vocabulary substitution.** Maintain a curated list of the "delve cluster" with synonyms, scored by register. The list should be configurable per preset (academic preset preserves "underscore" and "robust" because those are discipline-legitimate). At default intensity, substitute 1-2 instances per paragraph rather than every occurrence (preserves naturalness).
- **Discourse-marker thinning.** Detect sentence-initial "Moreover," "Furthermore," "Additionally," "However," etc. Substitute with: removal (merge into prior sentence), conjunction ("And," "But"), or replacement ("Also," "Yet," "Still"). At minimal intensity, substitute every other instance only.
- **Conversational scaffolding strip** (already partially done in Pass 1; this is the literal-text pass).
- **Parallel-structure detection and dampening.** Detect "X, Y, and Z" lists where X, Y, Z share grammatical structure. At balanced/aggressive intensity, vary one element's structure or break the list across two sentences.
- **"It's not X, it's Y" detection and rewrite.** Pattern-match this construction; rewrite as a single positive statement.
- **"X is a Y of Z" copular template detection.** Pattern-match and rewrite ~50% of instances.

**Libraries:** `re`/`regex` for pattern matching; `spacy` (en_core_web_sm or md) for POS-aware substitution and parallel-structure detection. A custom configurable dictionary file shipped in `humanize-mcp/data/lexical_substitutions.json`.

**Cost:** Milliseconds per paragraph; deterministic given the dictionary.

**Tests required:** Per-substitution unit tests with before/after examples. Idempotency check. Negative tests: substitution does not corrupt text where the trigger word has technical meaning (e.g., don't substitute "underscore" in a code-formatting context).

### Pass 3: Watermark scrubber (defensive)

**Purpose:** Remove statistical or covert watermarks that may have been embedded during generation.

- Already partially handled by Pass 1 (Unicode normalization).
- Optional: light per-token paraphrase of high-impact tokens (rare-word substitution) to perturb green-list watermarks. Defer this to Pass 5 (heavy paraphrasing) unless watermarking is specifically suspected.

**Cost:** Microseconds (passive normalization only).

**Tests required:** Round-trip on watermarked text from Kirchenbauer et al.'s reference implementation; verify watermark detection score drops.

### Pass 4: Burstiness and stylometric smoothing

**Purpose:** Adjust the sentence-length distribution and other measurable stylometric features toward a target distribution. Defeats GPTZero / ZeroGPT / perplexity-burstiness detectors directly.

- Compute current text's stylometric feature vector using **StyloMetrix**.
- Load target stylometric feature vector for the active preset.
- For burstiness specifically: compute current sentence-length variance; if below target, identify candidate sentences for splitting (compound sentences, sentences with multiple clauses) and merging (very short adjacent sentences). Apply edits until variance is within target range.
- Optional: compute perplexity under a small reference LM (GPT-2 or similar) and identify lowest-perplexity tokens for substitution if perplexity is below target.

**Libraries:**
- `StyloMetrix` for feature computation.
- `spacy` for sentence segmentation and dependency parsing.
- `transformers` + `gpt2` for perplexity scoring (optional).

**Cost:** Tens of milliseconds per paragraph for stylometric analysis; substitution edits are fast.

**Tests required:** Verify that target burstiness is achieved within tolerance; verify semantic preservation (sentence-transformers cosine similarity > 0.9 vs. input).

### Pass 5: DIPPER paraphrase (single controlled pass)

**Purpose:** Restructure sentences while preserving semantics. Defeats DetectGPT-class zero-shot detectors.

- Use the open-source `kalpeshk2011/dipper-paraphraser-xxl` model (T5-XXL, 11B params).
- Configure lexical-diversity and content-reordering control codes per preset/intensity:
  - `minimal`: lexical=20, reorder=0.
  - `balanced`: lexical=40, reorder=20.
  - `aggressive`: lexical=60, reorder=60.
- Operate paragraph-by-paragraph with adjacent-paragraph context.

**Library:** `transformers` for the model load; `torch` runtime; optionally `vllm` for higher-throughput serving.

**Cost:** ~1-3 seconds per paragraph on a single A10/RTX 4090; significantly faster batched.

**Hardware note:** This is the heaviest pass. Should be optional (skippable) for users without GPU. CPU inference is feasible but slow (~30s/paragraph). Provide a smaller fallback model option (e.g., a fine-tuned T5-base paraphraser) for CPU users.

**Tests required:** Semantic similarity preservation; verify detector scores improve vs. pre-pass baseline on HC3-style test pairs.

### Pass 6: Detector-guided iterative paraphrasing

**Purpose:** The state-of-the-art evasion technique (Cheng et al. 2025). Defeats most modern detectors.

- Loop:
  1. Score current text with one or more open-source detectors (Fast-DetectGPT, Binoculars).
  2. If aggregate AI-score is below threshold (e.g., 0.3 with appropriate calibration), exit.
  3. Otherwise, prompt a paraphraser LLM with the current text + the detector feedback ("This text scored 0.85 AI; rewrite to sound more human, especially these segments: [highlighted high-score sentences]").
  4. Sample N candidate paraphrases.
  5. Score each candidate; keep the one with the lowest detector score that maintains semantic similarity above threshold.
  6. Repeat until below detector threshold or max iterations (default: 5).

**Models:**
- Detectors: `Fast-DetectGPT` (https://github.com/baoguangsheng/fast-detect-gpt) via local install; `Binoculars` (https://github.com/ahans30/Binoculars).
- Paraphraser: any reasonable instruction-tuned LLM. Recommend Llama-3.1-8B-Instruct or Qwen-2.5-7B-Instruct for local use; OpenAI/Anthropic API with user key for cloud use.

**Cost:** N detector evaluations + N paraphrase candidates per iteration, up to max iterations. With N=3, max=5, on local hardware: ~30-90 seconds per text.

**Tests required:** End-to-end test on RAID benchmark subset; verify median detector score reduction matches Cheng et al. paper's reported magnitudes (within tolerance).

### Pass 7: Optional back-translation

**Purpose:** Final perturbation; particularly useful against retrieval-based defenses and translation-sensitive watermarks.

- Translate text → pivot language (default French or German for English source) → back to English.
- Use a quality MT model: NLLB-200, Helsinki-NLP OPUS-MT, or commercial API (DeepL, Google) with user key.
- Disabled by default (this can introduce noticeable awkwardness); enable for `aggressive` intensity or by explicit flag.

**Library:** `transformers` for NLLB; or commercial MT APIs.

**Cost:** Two MT calls per text.

### Pass 8: Style-transfer (preserve-voice mode)

**Purpose:** Restore user voice that earlier passes may have flattened. Only active when user supplies a `style_corpus_path`.

- Load user's style corpus (their past writing, 1000+ words minimum, preferably 5000+).
- Compute the corpus's stylometric feature vector and a few-shot prompt set (10-20 representative passages).
- Re-prompt the paraphraser LLM with: current humanized text + few-shot examples + instruction "rewrite to match the style of these examples."
- Optionally: if a fine-tuned LoRA adapter exists for the user, apply it instead.

**Libraries:** `peft` for LoRA; `sentence-transformers` for representative-passage selection.

**Cost:** One paraphrase call per text. LoRA training is one-time, ~10-30 minutes on consumer GPU.

### Pass 9: Final stylometric verification and report

**Purpose:** Verify the pipeline did what it was supposed to and report to the user.

- Re-score with all configured detectors.
- Re-compute stylometric feature vector; compare to pre-pipeline.
- If `return_diff=True`: produce a unified diff between input and output.
- If `return_scores=True`: produce per-detector before/after scores and per-pass time/cost breakdown.

**Output structure:**
```python
@dataclass
class HumanizeResult:
    text: str                              # final humanized text
    diff: Optional[str] = None             # unified diff if requested
    detector_scores: Dict[str, ScorePair]  # before/after per detector
    stylometric_delta: Dict[str, float]    # per-feature change
    pipeline_log: List[PassRecord]         # what each pass did
    warnings: List[str]                    # quality concerns
    semantic_similarity: float             # final vs. input
```

---

## 3. Dependency stack

### Required (MVP, passes 1-4)
```
python>=3.10
spacy>=3.7  + en_core_web_sm
nltk>=3.8
regex>=2024.1
textstat>=0.7
StyloMetrix>=0.2  (or similar)
sentence-transformers>=2.2
mcp>=1.0  (Model Context Protocol Python SDK)
pydantic>=2.0
```

### Required for full pipeline (passes 5-9)
```
transformers>=4.40
torch>=2.1
accelerate>=0.27
peft>=0.10
```

### Optional / dev
```
vllm  (for higher-throughput paraphrasing)
fast-detect-gpt  (clone from upstream, install as dev dependency)
binoculars-detector  (same)
huggingface-hub  (for downloading reference models)
```

### Detector adapter optional dependencies
```
openai  (for GPTZero adapter — wait, GPTZero is its own API, not OpenAI)
httpx  (for commercial-detector REST adapters: GPTZero, Originality.ai, Sapling)
```

---

## 4. Suggested HuggingFace models

| Role | Model | Size | License | Notes |
|------|-------|------|---------|-------|
| Paraphraser (DIPPER) | `kalpeshk2011/dipper-paraphraser-xxl` | 11B | apache-2.0 | Reference implementation; ~24GB VRAM |
| Paraphraser (light) | `humarin/chatgpt_paraphraser_on_T5_base` | 220M | mit | CPU-feasible fallback |
| Paraphraser (LLM) | `meta-llama/Llama-3.1-8B-Instruct` | 8B | llama license | Local instruction-tuned paraphraser |
| Paraphraser (LLM, alt) | `Qwen/Qwen2.5-7B-Instruct` | 7B | apache-2.0 | Alternative instruction-tuned paraphraser; fewer license restrictions |
| Detector (Fast-DetectGPT) | reference: `gpt-neo-2.7B` + `gpt-j-6B` | 2.7B+6B | mit/apache-2.0 | Both LMs needed for the curvature computation |
| Detector (Binoculars) | reference: `tiiuae/falcon-7b` + `tiiuae/falcon-7b-instruct` | 7B+7B | apache-2.0 | Observer + performer pair |
| Detector (RoBERTa) | `openai-community/roberta-base-openai-detector` | 125M | mit | Legacy reference; useful as one signal among many |
| Embedding (semantic check) | `sentence-transformers/all-mpnet-base-v2` | 110M | apache-2.0 | Quality embedding for similarity check |
| Embedding (light) | `sentence-transformers/all-MiniLM-L6-v2` | 22M | apache-2.0 | Fast option |
| Translation (back-translation) | `facebook/nllb-200-distilled-600M` | 600M | cc-by-nc-4.0 | Multilingual MT; note non-commercial license |
| Translation (alt) | `Helsinki-NLP/opus-mt-en-fr` etc. | small | apache-2.0 | Pair-specific, more permissive license |
| Perplexity scorer | `gpt2` | 124M | mit | Cheap reference LM |

**License notes:** NLLB-200 is CC-BY-NC, which restricts commercial use. For a permissively-licensed back-translation chain, use Helsinki-NLP OPUS-MT pair models (en-fr, fr-en) which are Apache-2.0.

---

## 5. Configuration: presets

Each preset is a YAML/JSON file defining: which passes are enabled, with what intensity, what stylometric target, and what preservation defaults.

Example (`presets/esl.yaml`):
```yaml
name: esl
description: |
  Preserve features common in ESL writing (simpler syntax, limited vocabulary,
  formal register) while removing the most-flagged AI surface tells.
passes:
  preprocess: enabled
  surface_tells:
    enabled: true
    em_dash_substitution: aggressive
    excess_vocabulary: aggressive
    discourse_markers: minimal      # preserve formal academic markers
    parallel_structure: minimal     # preserve learned templates
    copular_templates: disabled
  watermark_scrubber: enabled
  stylometric_smoothing:
    enabled: true
    burstiness_target: 0.6          # below native baseline
    perplexity_target: 8.0          # below native baseline
    operate_on: paragraph_only      # don't touch within-paragraph structure
  dipper:
    enabled: false                  # too aggressive; would erase voice
  detector_guided:
    enabled: true
    intensity: balanced
    paraphraser_instruction: |
      Rewrite the following to sound more like a confident,
      well-educated non-native English speaker. Maintain the
      author's vocabulary choices and sentence structures where
      reasonable.
  back_translation: disabled
  style_transfer: enabled_if_corpus_supplied
preserve_features:
  - lexical_diversity_floor: 0.4    # don't compress vocabulary further
  - discourse_marker_floor: 0.7     # preserve formal connectives
  - sentence_length_mean: tolerance(0.15)  # don't shift mean
```

Similar files for `casual`, `blog`, `literary`, `academic`, `email`, `neurodivergent`, `preserve` presets.

---

## 6. Detector adapter contract

Each detector adapter implements:

```python
class DetectorAdapter(Protocol):
    name: str
    requires_api_key: bool
    cost_per_call: Optional[float]   # estimated $ if commercial
    
    def score(self, text: str) -> DetectorScore:
        """
        Returns a DetectorScore with:
          - probability_ai: float in [0, 1]
          - confidence: float in [0, 1]
          - sentence_scores: Optional[List[float]]
          - raw_response: dict (provider-specific)
        """
```

Adapters to ship:
- **Built-in (no API key):** `fast_detect_gpt`, `binoculars`, `roberta_openai`, `stylometric_xgboost` (a small custom classifier we train on HC3 features).
- **Optional (user-supplied API key):** `gptzero`, `originality_ai`, `sapling`, `copyleaks`. Pangram if/when API access becomes broadly available.

The pipeline's `target_detectors` parameter selects which adapters drive the detector-guided pass and which appear in the score report.

---

## 7. Testing strategy

- **Unit tests per pass.** Idempotency, semantic preservation, deterministic outputs given fixed inputs.
- **Integration tests** running the full pipeline on RAID benchmark samples; assert detector-score reduction matches published Cheng et al. magnitudes within tolerance.
- **Regression tests on preserve-voice mode.** For each preset, assert that flagged preserve-features remain within tolerance after pipeline execution.
- **Quality regression suite.** Curated set of ~50 input texts with human-rated quality; CI flags pipeline changes that drop quality below threshold.
- **Bias check.** Run pipeline on subset of ESL learner essays (ICLE) and verify the pipeline does not flatten lexical diversity or other ESL features below preservation thresholds.
- **Watermark stripping.** Generate watermarked text via Kirchenbauer reference implementation; verify watermark detection score drops below threshold after pipeline.

Test data should live in `humanize-mcp/tests/fixtures/` with documented provenance and licensing.

---

## 8. Non-goals (explicit list of things NOT to ship)

- **Homoglyph substitution.** Documented in section 04; this is an exploit, not a humanization technique, and ships only harm.
- **Zero-width character insertion.** Same reason.
- **Pirated training corpora.** No Books3, no scraped Substacks, no Twitter dumps acquired post-2023 API closure.
- **Blanket "remove all AI features" mode.** Always preserve some signal of voice; the goal is to defeat detectors *while keeping the writer's identity*, not to maximally generic-ify the text.
- **Plausible-deniability features.** No "claim this was written by [name]" features. The tool transforms text; it doesn't lie about authorship.
- **Built-in commercial-detector evasion *guarantees*.** We make empirical reductions in detection probability, not guarantees. Marketing language must reflect this honestly.
- **Bundled commercial-detector API keys.** All commercial-detector access requires the user to supply their own key. Avoid the legal exposure of operating those APIs on user behalf at scale.

---

## 9. Phased roadmap

**v0.1 (MVP):** passes 1-4 (preprocess, surface tells, watermark scrub, stylometric smoothing). Built-in scorers: `roberta_openai` only. Three presets: `casual`, `academic`, `preserve`. CLI + MCP server. Defeats GPTZero / ZeroGPT for typical inputs.

**v0.2:** Add pass 5 (DIPPER). Add `fast_detect_gpt` and `binoculars` adapters. Add `esl`, `neurodivergent` presets. Defeats DetectGPT-class detectors.

**v0.3:** Add pass 6 (detector-guided iterative paraphrasing). Add commercial-detector adapters (user-key). Defeats most current detectors. This is the "production-ready" milestone.

**v0.4:** Add pass 7 (back-translation), pass 8 (style transfer). LoRA fine-tuning for committed users.

**v0.5+:** Continuous improvement: track new detectors as they release; track new evasion techniques in the literature; refresh excess-vocabulary list; add language coverage beyond English.

---

## 10. Final notes for the Pipeline Builder

- **Build the observability first.** The diff/scores output is what gives users (and the project) confidence the pipeline is doing the right thing. Implement `humanize_dry_run` and the report structure before optimizing any pass for speed.
- **Default to conservatism.** "Minimal" intensity should be the default for any new preset until empirical testing shows balanced/aggressive is needed for that population.
- **Document every removed feature.** When the pipeline strips an em dash, the diff should show it. When it substitutes "delve" → "examine," the diff should show that. Trust requires legibility.
- **Cite the research.** Each pass should have a docstring referencing the paper it implements. Each preset should reference the population research that motivates it. The codebase should read like an applied translation of this dossier.
