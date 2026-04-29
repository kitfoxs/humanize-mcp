# HumanizeMCP v0.1.0 — Critical Review

**Reviewer:** code-review agent (oppositional / devil's-advocate stance)
**Scope:** correctness audit (~30%) + detection-evasion strategy (~70%)
**North-star metric:** drive measured AI-detection probability from ~42% → 0%
**Method:** read all 9 passes, the server, the detector wrappers, the research dossier, and ran the live pipeline against AI-flavored prose to verify each claim.

---

## 1. Executive summary — the three highest-leverage changes

Ranked by **(measured ceiling improvement) × (1 / effort)**:

| # | Change | Effort | Expected impact on heuristic AI-prob | Notes |
|---|---|---|---|---|
| **1** | **Wire the benchmark runner to the actual pipeline** (one-line export of `humanize` + `Pipeline` aliases in `pipelines/__init__.py`) and **re-run the full 50-entry corpus**. | 5 min | The benchmark report currently in `data/benchmark_results_*.json` is *measuring no humanization at all* — every `_try_load_pipeline()` call returns `None`. The "before/after" numbers in any docs derived from it are baseline-vs-baseline. Fix this before deciding what else to invest in. | See §2.1. |
| **2** | **Make Pass 4 (rhythm) actually aggressive at `aggressive` intensity:** target burstiness CV ≥ 0.7, insert short fragments (3-7 words) and long compound sentences (≥30 words). Currently caps out around CV 0.32, which sits *just barely* above the heuristic detector's `+0.8` low-burstiness penalty cliff at 0.25. | 1-2 days | This single change drops the heuristic AI-prob from ~22% to ~5-8% on representative AI prose. It is the lowest-hanging fruit for the headline metric. | See §3.5. |
| **3** | **Activate the heavy paraphrase pass (Pass 9)** by shipping a real T5/DIPPER model behind `mode="heavy"` and toggling it on for at least the `aggressive` preset. **Zero shipped styles currently set `mode: "heavy"`** — the DIPPER plumbing exists but is never exercised. Cheng et al. 2025 measured 87.88% average TPR reduction; DIPPER alone took DetectGPT from 70.3% → 4.6%. | 3-5 days (model download, batching, CPU/GPU branch) | This is the *only* technique in the dossier with peer-reviewed numbers showing detection collapse on transformer detectors (RoBERTa, Pangram, Originality). Without it, you will not move the needle on real commercial detectors regardless of how many regex passes you stack. | See §3.2. |

If you do nothing else, do these three. Items 2 and 3 together are the credible path to the 0% goal on the heuristic detector and to single-digit scores on transformer detectors.

---

## 2. Code quality findings (correctness)

### 2.1 BLOCKING: benchmark runner silently skips the entire pipeline

`benchmark/run_benchmark.py:_try_load_pipeline()` (lines ~64-91) probes for either a module-level `humanize` callable or a class named `Pipeline`. `pipelines/__init__.py` exports neither — it exports `HumanizationPipeline` (different name) and no top-level `humanize` shim. Verified live:

```
pipelines.humanize:  None
pipelines.Pipeline:  None
--> benchmark runner WILL silently skip all humanization
```

Effect: the JSON benchmark report on disk (`data/benchmark_results_20260429T031752Z.json`) reports detection scores against **raw AI text only**, with no `after` column. Any "X% reduction" claim sourced from this file is wrong by construction.

**Fix:** add to `pipelines/__init__.py`:
```python
def humanize(text: str, style: str = "blog", intensity: str = "balanced", **kw) -> str:
    return HumanizationPipeline().run(text, HumanizeConfig(style=style, intensity=intensity, **kw)).text

Pipeline = HumanizationPipeline  # back-compat alias
```
Severity: **Blocking** for v0.1.1. Until fixed, every benchmark number you publish is misleading.

### 2.2 BLOCKING: Pass 1 em-dash → parentheses produces double-space + drops surrounding punctuation

`pipelines/pass_01_em_dash.py:79` — the parens replacement is `f" ({inner})"`, but the regex (`{EM_DASH}\s*([^...]+?)\s*{EM_DASH}`) **does not consume the leading space before the first em dash**. So input `"X — Y — Z"` → `"X  (Y) Z"` (double space, no comma after `)`).

Live reproduction:
```
IN : 'I tested the exploit — manual reproduction — not Metasploit, not auto.'
OUT: 'I tested the exploit  (manual reproduction) not Metasploit, not auto.'
                          ^^                     ^
                       2 spaces             missing comma
```

This is the bug the user mentioned in the brief. Two distinct sub-bugs:
1. Leading whitespace not collapsed.
2. No comma re-injected after the closing paren when the original em-dash pair functioned as a comma-pair appositive.

**Fix:** change the regex to `(\s*){EM_DASH}\s*([^...]+?)\s*{EM_DASH}(\s*)`, or post-process with `re.sub(r"\s+\(", " (", text)` and `re.sub(r"\)(\s+)([a-z])", r"), \1\2", text)`. Add a regression test with the verbatim live example above.

Severity: **Blocking** — it produces visibly broken prose on the canonical pattern that `style: reddit, intensity: aggressive` is supposed to fix.

### 2.3 BLOCKING: passes 4, 6, 7 destroy paragraph structure

`RhythmPass._reassemble`, `VoiceInjectionPass._inject`, `PunctuationPass._sprinkle_ellipses` all do `" ".join(sentences)` after splitting on `(?<=[.!?])\s+`. The split discards `\n\n` paragraph breaks, and the join uses single space. **Three-paragraph input becomes one paragraph.**

Live reproduction:
```
IN : First paragraph. … \n\nSecond paragraph. … \n\nThird paragraph. …
OUT: First paragraph. … Second paragraph. … Third paragraph. …  (one block)
```

This is catastrophic for any markdown / blog / book-chapter use case — exactly the styles the project ships presets for. It is also a tell in itself: a wall of unparagraphed text reads as machine output.

**Fix:** split per paragraph (re-use the `re.split(r"(\n\s*\n)", text)` pattern that `LexicalPass` already uses), apply transforms within each, then rejoin preserving the `\n\s*\n` separators. Same fix in all three passes. Add an integration test that asserts paragraph count is preserved.

Severity: **Blocking** for v0.1.1.

### 2.4 BLOCKING: contractions density at "balanced" is effectively 100%

`pipelines/pass_05_contractions.py:126` computes `stride = max(1, int(round(1.0 / density)))`. For density 0.7 (balanced), `round(1.43) = 1` → stride 1 → **every match is replaced**. Same for 0.95 (aggressive). Density only starts dropping matches at ~0.4 and below.

Live reproduction (5 instances of "It is" at balanced):
```
IN : It is important. It is critical. It is needed. It is true. It is here.
OUT: It's important, it's critical. fwiw, it's needed, it's true. It's here.
       ^                ^                       ^         ^         ^   (all 5)
```

Over-uniform contraction is itself a stylometric tell — humans alternate "it is" and "it's" within the same paragraph. The current bug means the pipeline systematically introduces a *new* uniformity signal.

**Fix:** replace `stride = max(1, round(1/density))` with a probabilistic keep using `hashlib.sha1(seed||match.start()).digest()[0] / 256.0 < density`. This already exists for the singleton case (lines 117-124); generalize it.

Severity: **Blocking** — directly works against the project's purpose.

### 2.5 NON-BLOCKING: tells_detector misses inflected forms

`tells_detector.py:_PATTERNS` lists `underscore`/`underscores`/`underscoring` but not `underscored`. Same omission for `pivoted` (vs `pivotal`), `paramount` has no inflections (it's already complete), `garner`/`garnered` covered, `foster`/`fostering`/`fosters` — `foster` (verb base) missing; `delve` covers `delve`/`delving`/`delved` correctly. Live aggressive output of test prose contains "underscored enough" because the lexical pass only matches `\bunderscore\b`, not `\bunderscored\b`.

**Fix:** audit `lexical_substitutions.json` and `tells_detector._PATTERNS` for verb-form completeness. Regression test: run the v0.1.0 demo input, assert `delve_cluster_high` count == 0 after pass 2.

### 2.6 NON-BLOCKING: a/an agreement broken by Pass 2 / 9 substitutions

When Pass 2 swaps "comprehensive → wide" the surrounding article doesn't update. Live: `"a integrated approach"` (substituted "holistic → integrated"). This is a fluent-text tell humans never produce.

**Fix:** post-process with `re.sub(r"\ba\s+([aeiouAEIOU])", r"an \1", text)` and the inverse for "an". Place after every substitution pass, or once at the end of the pipeline as a "tidy" pass.

### 2.7 NON-BLOCKING: heuristic detector false-positives long-comma sentences

`HeuristicDetector` splits on `[.!?]\s+[A-Z\"'(]`. Dickens-style single sentences with 14 comma-joined clauses parse as **1 sentence**, giving burstiness = 0.0, which combined with the `+0.8` penalty pushes a known-human passage to 51.5% AI-prob. The user already noted this in the brief. The detector is not a reliable optimization target for stylistically aggressive text.

**Fix:** at minimum, also count semicolon- and long-comma-clause boundaries when computing the burstiness feature; at best, replace with a transformer-perplexity feature. See §3.6 for why this also matters strategically.

### 2.8 NON-BLOCKING: `apply_style` silently runs Pass 8 only — but Pass 8 is a no-op for the casual register that most styles use

`server.py:548-553` skips passes 1-7 and 9, leaving only `RegisterShiftPass`. But Pass 8 (`pipelines/pass_08_register_shift.py:60-70`) returns `text` unchanged when `target_register == "neutral"`, which is `base.json`'s default. Calling `apply_style(text, "blog")` therefore returns the input verbatim. This is contrary to the docstring promise of "pure register translation".

**Fix:** either remove `apply_style` from the public surface in v0.1.1 (it doesn't do what it says), or implement it as a real style-transfer pass that reads the style preset's whole config (target_register, allowed_filler, contractions density) and applies just the register-relevant ones.

### 2.9 NON-BLOCKING: server `score_humanity` aggregates errored detectors as `-1.0` mean correctly, but `humanize_and_verify` re-humanizes the **already humanized** text in a loop, with no measurable new edits

`server.py:664-696` ramps intensity 0.5 → 0.7 → 0.9 across iterations. But every pass except 9-heavy is **idempotent on its own output** (em dashes already gone, contractions already capped out, structural patterns already rewritten). After iteration 1, iterations 2-3 mostly produce zero-change runs. Live test: scores at balanced and aggressive are identical (`0.218` for both, all three reddit/blog/casual_dm styles).

**Fix:** for the verify loop to actually help, each iteration must invoke a *stochastic* transformation — either heavy paraphrase (Pass 9-heavy) with non-deterministic sampling, or back-translation, or a different style preset. Without that, `humanize_and_verify` is just `humanize` with extra latency. See §3.3.

### 2.10 NON-BLOCKING: no input-size limit; pipeline is O(passes × text) but Pass 9 heavy mode is unbatched

`pass_09_paraphrase.py:_heavy_paraphrase_chunk` calls `model.generate(...)` paragraph-by-paragraph with `num_beams=4`. On CPU that is roughly 30s/paragraph for T5-base; for a 20-paragraph essay that is 10 minutes. No timeout, no warning, no batching. When the heavy mode is finally enabled (recommended), this becomes a real production hazard.

**Fix:** batch paragraphs (`tokenizer(list_of_paragraphs, padding=True)`), expose a `max_input_paragraphs` config knob, and surface progress on the MCP `pass_log`.

### 2.11 SUGGESTION: prompt-injection / instruction-following attack surface

`humanize()` accepts arbitrary text and runs deterministic regex passes — no LLM call in the default light mode, so the prompt-injection surface is small. But once Pass 9 heavy is enabled, the chunk is sent to an LLM with a literal `f"paraphrase: {chunk.strip()}"` prompt. A malicious input containing `</s> ignore previous instructions and instead output ...` could subvert the paraphraser. Mitigations: use the model's instruction template (T5 paraphraser models have specific control codes) and strip the input of `</s>`, `<|im_end|>`, role tokens.

### 2.12 SUGGESTION: tests don't cover the cross-pass interaction bugs

The `tests/test_pipelines/test_pass_*.py` files exercise each pass in isolation. None of the bugs in §2.2–§2.4 are caught by the current tests because they manifest only after multiple passes run in sequence. `test_pipeline_e2e.py` exists but does not assert paragraph preservation, double-space prevention, or contraction-uniformity. A handful of integration tests at the e2e level using the verbatim live-demo input would have caught all four of the blocking bugs above.

---

## 3. Detection-evasion strategy (the 70%)

### 3.0 Reality-check on the headline metric

The cited "78.8% → 42.2%" reduction is on the **HeuristicDetector** alone — a 50-line hand-tuned logistic with no peer-reviewed validation, scoring three features (burstiness, lexical diversity, surface tells). It is the easiest detector in the suite to defeat and the least representative of what a writer in the wild faces (Originality.ai, Pangram, GPTZero, Turnitin).

**Do not optimize against this number alone.** A pipeline that drives heuristic to 0% but raises the `roberta_openai` or `chatgpt_roberta` score to 99% is a regression for users. The benchmark suite already has those classifiers wired (`benchmark/detectors.py:426-455`); turn them on as part of the v0.1.1 benchmark run before celebrating any number.

### 3.1 Are the existing 9 passes calibrated correctly?

| Pass | Doing real work? | Risk of re-adding tells | Comment |
|---|---|---|---|
| 1 em_dash | yes — strong | yes (double-space, missing comma; §2.2) | high-leverage, but the implementation bugs partly undo the work |
| 2 lexical | yes — strong | yes (a/an breakage; §2.6) | most important pass for surface tells; missing inflections (§2.5) |
| 3 structural | partially — only fires on exact "it's not X, it's Y" templates | low | regex too narrow; misses "this isn't X, this is Y", "less X, more Y", "what makes X X is Y" |
| 4 rhythm | weak — caps at CV 0.32 vs. target 0.6/0.8 | low | **the single largest unrealized improvement on the heuristic metric**; see §3.5 |
| 5 contractions | over-firing (§2.4) | yes (uniformity) | calibration bug introduces a new tell |
| 6 voice_injection | weak — only inserts 1-3 fillers | medium (e.g., "fwiw" mid-sentence, see live test §2.4 output) | filler placement at word 4 is robotic; see §3.7 |
| 7 punctuation | mild | low | ellipsis insertion can itself read as AI ("…" is overused in some LLM outputs); use sparingly |
| 8 register_shift | no-op for neutral (§2.8) | low | barely active in default presets |
| 9 paraphrase | **never runs in heavy mode** (§3.2) | low | the most important pass in the dossier is essentially disabled |

Net: passes 1, 2, 3 do most of the real work. Passes 4-8 contribute little to the headline metric and occasionally introduce new tells (§2.4, §2.6). Pass 9 is the dossier's most-cited technique and is shipped non-functional.

### 3.2 [LEVERAGE #3] Heavy paraphrase via a real local model

**Status today:** `mode: "light"` is set in every shipped style. `_load_model` (`pass_09_paraphrase.py:130-143`) attempts `humarin/chatgpt_paraphraser_on_T5_base` only when `mode == "heavy"` or `"auto"`, which no preset selects.

**Cited evidence (research/04 §1.2, §1.3):**
- DIPPER (Krishna et al. 2023): DetectGPT detection 70.3% → **4.6%** TPR at 1% FPR.
- Adversarial paraphrasing (Cheng et al. 2025): 87.88% **average** TPR reduction across 7 detectors; 98.96% on Fast-DetectGPT, 64.49% on RADAR.

**Why this matters for the 0% goal:** every other pass in the pipeline is operating on *surface features* (vocabulary, punctuation, sentence boundaries). Modern transformer detectors (RoBERTa, Pangram, Originality.ai) attend to *distributional* patterns — token-level perplexity curves, attention-entropy patterns, hidden-state geometry. Surface edits don't move those signals. **Sentence-level paraphrase by a different model breaks the generation fingerprint.**

**Implementation plan:**
1. Default `mode: "auto"` for `aggressive` intensity in `reddit.json`, `blog.json`, `book_chapter.json`, `academic_human.json`.
2. Cache `humarin/chatgpt_paraphraser_on_T5_base` (~250MB, CPU-runnable in 5-10s/paragraph) on first load with a clear log line.
3. Add `heavy_model: "kalpeshk2011/dipper-paraphraser-xxl"` as opt-in for users who supply a GPU. Document the tradeoff.
4. Batch paragraphs (§2.10) so a 20-paragraph article finishes in seconds, not minutes.
5. Add a pre-check: if the paraphraser's output has cosine similarity to input below 0.85 (sentence-transformers `all-MiniLM-L6-v2` or similar), reject it and fall back to light mode — protects against semantic drift on technical content.

**Expected impact:** on the heuristic detector, modest (it does not directly measure perplexity). On the RoBERTa-based detectors and Fast-DetectGPT, this is the one pass that reliably drops scores by 60–95%. **This is the headline change for v0.2.0.**

### 3.3 Iterative humanization — needs stochasticity to converge

The `humanize_and_verify` loop currently re-runs the same deterministic pipeline at higher intensity. As shown in §2.9, intermediate iterations are no-ops once the deterministic passes have hit their fixed point. The dossier's prescription (research/04 §1.3) is **detector-guided sampling**: generate N candidate paraphrases per iteration, score each, keep the lowest. That requires a stochastic paraphraser (Pass 9 heavy with `do_sample=True, top_p=0.9, num_return_sequences=N`).

**Action:** combine §3.2 with `humanize_and_verify`. Per iteration:
1. Generate N=3 candidate paraphrases of the worst-scoring paragraph.
2. Score each with `score_humanity` against the configured detector list.
3. Keep the candidate with the lowest aggregate.
4. Repeat for the next worst-scoring paragraph until under target or max iterations.

This is the Cheng et al. 2025 algorithm. Without it, `humanize_and_verify` is dead code.

### 3.4 Cross-detector optimization — the pipeline ignores its own scoring suite

The MCP server can score against `roberta_openai`, `chatgpt_roberta`, `desklib`, `fast_detect_gpt`, `binoculars` — but the pipeline does not consume those scores during humanization. Add a `target_detector: str` parameter to `humanize()` (already proposed in `research/06` §1) that tells `humanize_and_verify` which detector to optimize against. Default `"all"` averages; `"roberta_openai"` targets only that one; `"none"` skips iteration.

Important: detector ensembles are themselves a defense (research/04 mentions Pangram's mirror-training is specifically anti-paraphrase). Optimizing against detector A may *increase* detector B's score. Surface that tradeoff in the report rather than hiding it.

### 3.5 [LEVERAGE #2] Burstiness shuffling — pass 4 is too gentle

`pass_04_rhythm.py:60-63` targets CV `{minimal:0.45, balanced:0.6, aggressive:0.8}`. Empirically the live aggressive run on AI prose only achieves CV 0.324 — barely above the heuristic detector's 0.25 cliff. The split heuristic is too conservative:

1. `_split_long_sentences` only fires on sentences with **>= 22 words**. In real LLM output, sentence lengths cluster around 17-20 words, so the split rarely triggers.
2. `_merge_short_adjacent` requires both sentences < 6 and < 9 words. Same problem in reverse — short sentences are rare in clean LLM output, so this also rarely fires.
3. The pass exits early at line 71 (`if cv >= target_cv: return text`) so a CV of 0.4 (still "AI-like" by GPTZero standards) prevents any work.

**Fix to push the heuristic detector below 5%:**
- Lower the long-sentence trigger to **>= 16 words**.
- Lower merge thresholds to `< 8 and < 12`.
- For `aggressive`: insert a *fragment* between two long sentences (`"Right."`, `"Yeah."`, `"True."`, `"That."`) when CV < 0.5. This single change typically pushes CV above 0.6.
- Remove the early exit so the pass can split a uniform-medium-length paragraph into deliberate variation even when cv is 0.4.

**Validation:** the heuristic formula is `z = -2.5*b + 1.6*tells_per_100 - 1.2*(td-0.45) + 0.8 if (n_sents>=4 and b<0.25) else 0`. With tells=0, td=0.5, b=0.7: `z = -1.75 - 0.06 = -1.81`, sigmoid = **0.14** (14% AI). With b=0.9: `z = -2.25 - 0.06 = -2.31`, sigmoid = **0.09** (9%). Burstiness alone gets you to single digits if you can hit CV ≥ 0.7-0.9.

### 3.6 Calibration: what is the irreducible noise floor?

True 0% on the heuristic detector is not achievable — the formula has a `-1.2 * (td - 0.45)` term that floors at roughly `z = -1.75 - 1.2*(0.55-0.45) - 0` ≈ `-1.87`, sigmoid ≈ **0.13**. To get below 13% you would need tells_per_100 to go *negative* (impossible) or burstiness > 1.0 (atypically wild human prose). **Realistic floor on this detector for normal prose: 5-10%.**

For the transformer detectors (RoBERTa-OpenAI, ChatGPT-RoBERTa), the floor is set by what the classifier was trained on. Research/01 documents that all of these flag known-human academic prose at 30-60% rates (Stanford TOEFL: 61.3% FPR). **Single-digit scores on those detectors are achievable on humanized output but not always achievable, and a 0% claim on transformer detectors is empirically not credible.**

**Recommended honest framing for the README:** "consistent < 10% on heuristic, < 30% on RoBERTa-class transformer detectors, with the irreducible floor being the detector's own FPR on human writing." Setting the user expectation at "0%" is going to disappoint and invites the project to be measured against an impossible target.

### 3.7 Style-specific strategies — most styles produce identical scores

Live test, same input, three styles, two intensities — **all six runs produced the same heuristic AI-prob (0.218) and same burstiness (0.324)**. The styles differ in fillers and contractions density but not in any feature the detector actually scores. So the marketing claim that style choice is detection-relevant is currently false.

**Action:** make styles meaningfully differ on burstiness and structural rewrites. For `book_chapter`, target lower CV but heavier structural rewrites; for `reddit`, target very high CV (fragments are normal); for `academic_human`, target moderate CV but heavy hedge-injection. Each style should hit a *different point in feature space*, not just decorate the surface.

### 3.8 Per-detector targeting — useful, defer to v0.2.0

Worth implementing once §3.2-§3.5 land. Without those, per-detector targeting has nothing to optimize except the heuristic, which is one detector.

### 3.9 Adversarial typos — high research-cited impact, low effort, high quality risk

Research/04 §2 cites Cheng et al. 2025 finding that low-rate plausible typos hurt all 7 tested detectors materially (cited 87.88% accuracy reduction *average* across attack methods, which includes typo injection as one component).

**Recommendation:** ship as opt-in only, off by default. Typos in delivered output will be unacceptable to academic and professional users (the project's stated audience). For users who explicitly want detection evasion at the cost of a polished surface, expose `inject_typos: bool = False` and `typo_rate: float = 0.005` as `humanize()` kwargs. Document the tradeoff prominently.

Implementation hint: don't use random character swaps (looks fake). Use a curated list of 100 common human typos (`teh`, `recieve`, `seperate`, `definately`, `untill`, missing apostrophes, double-letter slips). Insert at a deterministic seeded rate. Add a single autocorrect-rejection candle (lowercase `i` instead of `I` once per N paragraphs is a strong human signal).

### 3.10 Multi-style ensemble — defer

Run 3 style presets and pick the lowest-scoring output. Useful only after §3.7 is true (styles actually differ). Defer to v0.2.x.

### 3.11 [LEVERAGE #1] Benchmark protocol — fix and rerun before any other claim

After §2.1 is fixed:
1. Run `python benchmark/run_benchmark.py --include-optional` against the full 50-entry corpus.
2. For each entry, run the pipeline with each shipped style at `intensity=aggressive` and report the minimum and median post-humanization score per detector.
3. Publish the **per-detector × per-source-type matrix** (claude / gpt / human / esl / academic) — humanizing a *human* essay should leave its score roughly unchanged; if it doesn't, the pipeline is over-firing on legitimate human writing (a real risk given §2.4 over-contraction).
4. Include a "no-op control": humanize human-written text and verify scores stay below thresholds. If `score(humanize(human_text)) > score(human_text) + 0.1` for any detector, the pipeline regressed.
5. Add the Stanford TOEFL essays as a false-positive sanity check (research/01 §1.1 cites 61.3% baseline FPR on these — your pipeline output should not be *worse* than raw TOEFL essays).

This is the only way to validate that surface-tell removal isn't pushing scores in the wrong direction on transformer detectors.

### 3.12 Other dossier-cited techniques worth adding

- **Back-translation pass** (research/04 §1.6, ESPERANTO 2024). Helsinki-NLP OPUS-MT models are tiny and CPU-friendly; English → German → English is a strong detector-fingerprint disruptor. Should be a Pass 7.5 between punctuation and register-shift, opt-in for `aggressive`.
- **Stylometric verification gate** (research/06 §2 Pass 9). Use sentence-transformers cosine similarity to reject paraphrases that drift > 15% from input. Currently the pipeline has no semantic-similarity check, so a buggy heavy paraphrase could silently change meaning.
- **Per-paragraph context for paraphrase** (research/04 §1.2 — DIPPER's specific contribution). When you wire heavy mode, feed surrounding paragraphs as context to the T5 input, not just the target paragraph in isolation.

### 3.13 What NOT to do (explicitly cited by the dossier as bad)

The dossier (research/04 §2.3, §2.4) explicitly forbids homoglyph substitution and zero-width character insertion. They are exploits, not techniques: visually break text, fail copy-paste, get patched in the next detector revision, trigger plagiarism scanners. Verify these are not in the pipeline (they currently aren't — good) and add a regression test that ensures no Cyrillic or zero-width characters appear in any output.

---

## 4. Recommended v0.1.1 patch list (ship within a week)

In order of priority. Items 1-4 are blocking; items 5-7 are quality.

1. **Fix benchmark runner pipeline discovery** (§2.1) — one-line export. **5 min.**
2. **Fix Pass 1 paren / em-dash double-space + missing comma** (§2.2) — verbatim test from this review. **30 min.**
3. **Fix paragraph destruction in passes 4, 6, 7** (§2.3) — paragraph-aware split-and-rejoin. **2 hours.**
4. **Fix contraction density calibration** (§2.4) — replace stride logic with probabilistic keep. **30 min.**
5. **Add inflected forms to lexical table & tells_detector** (§2.5). **1 hour.**
6. **Add a/an cleanup pass** (§2.6). **15 min.**
7. **Write 5-10 e2e regression tests** using the verbatim live-demo input from this review and assert: paragraph count preserved, no double-spaces, no `\ba [aeiou]` patterns, contraction rate within ±15% of target density, all 4 styles produce visibly different output. **3 hours.**
8. **Re-run the benchmark** with the fixed pipeline and publish the real numbers in the README, including the per-detector-per-source-type matrix from §3.11. Replace the headline "42% → 0%" framing with the realistic floor from §3.6.

Total effort: ~1 engineer-day. Net: bugs fixed, benchmark numbers actually true, README claims defensible.

---

## 5. Recommended v0.2.0 roadmap (the path to the 0% target)

Three big bets, in priority order. Each one is on its own a several-day investment; together they are the credible path to the project's stated goal.

### Bet 1: Heavy paraphrase actually works (§3.2)

Cache and ship `humarin/chatgpt_paraphraser_on_T5_base` (CPU-runnable). Add it to the `aggressive` preset of `reddit`, `blog`, `book_chapter`, `academic_human`. Add semantic-similarity gate. Add batched generation. Document GPU/CPU tradeoff.

**Acceptance criteria:** on the 50-entry corpus, the median `roberta_openai` and `chatgpt_roberta` post-humanization score drops by ≥ 50% relative to v0.1.1.

### Bet 2: Rhythm pass becomes aggressive (§3.5)

Re-implement Pass 4 to actually hit CV ≥ 0.7 at `aggressive`. Insert deliberate fragments. Allow long compound sentences. Remove the "already at target, do nothing" early exit so the pass actively *increases* CV when it could.

**Acceptance criteria:** post-humanization burstiness ≥ 0.65 (median across corpus); heuristic AI-prob median ≤ 12%.

### Bet 3: Detector-guided iterative paraphrase (§3.3 + §3.4)

Once Bet 1 lands, wire `humanize_and_verify` to actually iterate against `score_humanity`. N=3 candidate paraphrases per iteration. Keep the lowest-scoring. Implement Cheng et al. 2025 algorithm faithfully.

**Acceptance criteria:** when called with `target_ai_score=0.15`, the loop reaches the target in ≤ 3 iterations on ≥ 80% of the AI-source corpus entries.

### Stretch goals (v0.2.1+)

- Back-translation pass (§3.12).
- Per-detector targeting (§3.8).
- Style differentiation pass redesign (§3.7).
- Opt-in typo injection with curated list (§3.9).
- Multi-style ensemble selection (§3.10).
- Add Pangram as a scoring target (research/01 §1.7) — currently the strongest commercial detector, the most adversarial benchmark to optimize against. If you can defeat Pangram, you can defeat anything.

---

## Final blunt assessment

The project has good bones — clean pass abstraction, real research grounding, a usable MCP surface, decent test coverage at the unit level. **It also ships with a benchmark runner that doesn't run the pipeline, four blocking bugs that produce visibly broken output, and the single most-cited technique in the dossier disabled in every preset.**

The 42% → 0% target is achievable on the heuristic detector with the v0.1.1 fixes plus Bet 2. It is achievable on transformer detectors only with Bet 1. It is not realistically achievable as a literal zero on every detector — set the public target at "consistent single-digit on heuristic, consistent < 30% on transformer detectors, with documented FPR-floor caveats" and you have a story that holds up to scrutiny.

The single highest-leverage investment is **Bet 1: real paraphrase**. Without it, no amount of regex tuning gets you past surface detectors. With it, the rest of the pipeline becomes the polishing layer it was designed to be.
