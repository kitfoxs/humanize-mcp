# 04 — Humanization Techniques: Survey

A survey of every published and commercial technique for transforming AI-generated text such that it reads as human-authored. Organized by family. Each entry includes mechanism, citation or source, measured effectiveness, computational cost, and degradation risk to the underlying text.

The headline finding: **detector-aware adversarial paraphrasing is the empirically dominant technique** (Cheng et al. 2025, David & Gervais 2025), defeating essentially all current detectors with negligible semantic loss. Commercial humanizers have not yet caught up to the academic state of the art.

---

## 1. Direct paraphrasing

### 1.1 Naive single-pass paraphrasing

Pass the AI text through any LLM with a prompt like "Rewrite this to sound human."

- **Mechanism:** zero shot LLM rewrite.
- **Cost:** one API call.
- **Effectiveness:** Modest. Defeats older perplexity-only detectors (DetectGPT-class) but caught by modern supervised detectors (Pangram, Originality.ai). Dugan et al. (RAID, 2024) showed that simple paraphrasing alone increases the True Positive Rate at 1% FPR by only 8.57% on RADAR and 15.03% on Fast-DetectGPT [Dugan2024]. In other words, naive paraphrasing barely moves the needle against modern detectors.
- **Degradation:** Often introduces awkwardness, semantic drift, factual errors.
- **Verdict:** Insufficient as a standalone technique.

### 1.2 DIPPER: controlled multi-paragraph paraphrasing

**Krishna, K., Song, Y., Karpinska, M., Wieting, J., & Iyyer, M. (2023).** "Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense." NeurIPS 2023. arXiv:2303.13408.

- **Mechanism:** An 11B-parameter T5-based paraphrase model fine-tuned to:
  - Paraphrase paragraph-by-paragraph (not sentence-by-sentence).
  - Condition on surrounding context.
  - Accept explicit *control codes* for **lexical diversity** and **content reordering** intensity.
- **Effectiveness (from the paper):** Drops DetectGPT detection accuracy from **70.3% to 4.6%** at 1% FPR; defeats GPTZero, OpenAI's text classifier, and watermarking; preserves semantics measured by automatic and human evaluation.
- **Cost:** A 11B-parameter forward pass per paragraph. Runs locally on consumer GPUs (~24GB VRAM), or via HuggingFace.
- **Degradation:** Minimal. Quality preserved across multiple human evaluations.
- **Verdict:** **Foundational technique.** Open-sourced; available on HuggingFace as `kalpeshk2011/dipper-paraphraser-xxl`. Should be a primary building block in HumanizeMCP.

### 1.3 Adversarial Paraphrasing — the current state of the art

**Cheng, Y., Sadasivan, V. S., Saberi, M., Saha, S., & Feizi, S. (2025).** "Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text." arXiv:2506.07001.

- **Mechanism:** Training-free attack framework. Loop:
  1. Paraphrase the AI text using an off-the-shelf instruction-tuned LLM.
  2. Score the result with an open-source AI detector.
  3. If still classified as AI, prompt the paraphraser with feedback (or sample multiple paraphrases and keep the lowest-detection-score one).
  4. Iterate until below threshold.
- **Effectiveness:** Reduces True Positive Rate at 1% FPR by **64.49% on RADAR** and **98.96% on Fast-DetectGPT**. Average reduction across diverse detectors: **87.88%** [Cheng2025].
- **Cost:** Multiple LLM calls per paragraph. Roughly 5-15 paraphrase iterations per text in the paper's setup.
- **Degradation:** "Slight degradation in text quality" per the paper's quality vs. attack-success analysis. Configurable trade-off.
- **Verdict:** **The HumanizeMCP primary technique.** Empirically dominates against the broadest range of detectors. Implementation requires only a paraphraser LLM (any decent open-source instruction-tuned model: Llama-3.3-70B, Qwen-2.5-72B, Mistral) and a detector signal (Fast-DetectGPT or Binoculars locally).

### 1.4 AuthorMist: reinforcement-learning-trained paraphraser

**David, I., & Gervais, A. (2025).** "AuthorMist: Evading AI Text Detectors with Reinforcement Learning." arXiv:2503.08716.

- **Mechanism:** A 3B-parameter language model fine-tuned with **Group Relative Policy Optimization (GRPO)** using detector APIs (GPTZero, Winston, Originality.ai) as the reward signal. Frames evasion as an RL problem: paraphraser is the policy, detector score is the reward.
- **Effectiveness:** Attack success rates **78.6% to 96.2%** against individual detectors, "significantly outperforming baseline paraphrasing." Maintains semantic similarity above **0.94**.
- **Cost:** One-time fine-tuning cost; inference is cheap (3B params).
- **Verdict:** Promising direction, but requires querying paid detector APIs during training. The Cheng et al. approach (training-free, uses open-source detector for guidance) is more tractable.

### 1.5 Iterative paraphrasing

**Zha, Y., Min, R., & Sushmita, S. (2025).** "PADBen: A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase Attacks." arXiv:2511.00416.

- **Mechanism:** Apply paraphrase repeatedly (3, 5, or more rounds), each time on the previous paraphrase. Creates an "intermediate laundering region" where text has shifted semantically but retained generation patterns.
- **Effectiveness:** Detectors achieve over 90% accuracy on raw AI output but "fail catastrophically against iteratively-paraphrased content." Particularly effective at "authorship obfuscation" (paraphrasing human text) — detectors cannot tell the result is human-originated.
- **Cost:** N paraphrase iterations.
- **Degradation:** Accumulates with iterations. After 5 rounds, semantic drift is noticeable.
- **Verdict:** Use sparingly (2-3 iterations max) when paired with detector-signal guidance.

### 1.6 Back-translation (ESPERANTO)

**Ayoobi, N., Knab, L., Cheng, W., et al. (2024).** "ESPERANTO: Evaluating Synthesized Phrases to Enhance Robustness in AI Detection for Text Origination." arXiv:2409.14285.

- **Mechanism:** Translate AI text into another language (or several pivot languages), then translate back to English. Each translation pass introduces L2-style structural variation and breaks the original generation fingerprint.
- **Effectiveness:** Significantly reduces TPR across nine evaluated detectors. The combined-back-translation variant is more effective than single-pass.
- **Cost:** N+1 translation calls per text.
- **Degradation:** Translation drift; mistranslation of idioms; loss of stylistic flourish.
- **Verdict:** Cheap and effective as a preprocessing step. Cross-lingual watermark removal (He et al. 2024) is essentially the same technique applied to watermarked text. Recommended as an **optional preprocessing pass** in the pipeline.

---

## 2. Substitution-based attacks

### 2.1 Synonym substitution

- **Mechanism:** Replace tokens with synonyms while preserving syntax. Implemented in nearly every commercial humanizer.
- **Effectiveness:** Modest against modern detectors. The vocabulary fingerprint shifts but other features (sentence-length distribution, punctuation, structural patterns) remain.
- **Cost:** Negligible.
- **Verdict:** Use for the specific "delve cluster" excess-vocabulary substitution. Not sufficient as a standalone technique.

### 2.2 Word-substitution adversarial perturbation

**Peng, X., Zhou, Y., He, B., Sun, L., & Sun, Y. (2024).** "Hidding the Ghostwriters: An Adversarial Evaluation of AI-Generated Student Essay Detection." arXiv:2402.00412.

- **Mechanism:** Targeted substitution of specific high-influence tokens identified via detector gradients or feature-importance analysis.
- **Effectiveness:** Effectively circumvents detectors on student essay datasets while maintaining quality.
- **Verdict:** Useful when detector internals are accessible (open-source detectors).

### 2.3 Homoglyph substitution (SilverSpeak)

**Creo, A., & Pudasaini, S. (2024).** "SilverSpeak: Evading AI-Generated Text Detectors using Homoglyphs." arXiv:2406.11239.

- **Mechanism:** Replace ASCII characters with visually identical Cyrillic/Greek codepoints (e.g., Latin "a" U+0061 → Cyrillic "а" U+0430).
- **Effectiveness:** Devastating. Drops Matthews Correlation Coefficient from 0.64 to -0.01 across seven detectors (ArguGPT, Binoculars, DetectGPT, Fast-DetectGPT, Ghostbuster, OpenAI's detector, and watermarking).
- **Verdict:** **Do not use.** This is an exploit, not a humanization technique. It will:
  1. Be trivially detected and patched in next-generation detectors via Unicode normalization.
  2. Visually break the text in many fonts and accessibility tools (screen readers stumble on mid-word codepoint switches).
  3. Trigger anti-fraud systems (CAPTCHAs, plagiarism scanners, search engines).
  4. Produce text that is functionally broken — copy-paste, search, grep all fail.
  
  HumanizeMCP **must explicitly NOT** ship a homoglyph substitution mode. This is in the dossier as a known-bad technique to avoid.

### 2.4 Whitespace and zero-width character insertion

Similar exploit class to homoglyphs. Insert U+200B (zero-width space), U+200C (zero-width non-joiner), or U+FEFF (BOM) between characters. Defeats some detectors. Same objections as homoglyphs: exploit, not technique. Will be normalized out by any defensive preprocessor. Will break copy-paste. Do not ship.

---

## 3. Style-transfer techniques

### 3.1 Few-shot style prompting

- **Mechanism:** Provide the LLM with several examples of the target writing style and ask it to rewrite in that style.
- **Effectiveness:** Modest improvement over naive paraphrasing. Quality depends entirely on the curated style examples.
- **Cost:** Negligible.
- **Verdict:** Useful as a *first* pass in combination with other techniques. The HumanizeMCP `style_corpus` feature should make this easy.

### 3.2 StyleMC: contrastively-trained style representations

**Khan, A., Wang, A., Hager, S., & Andrews, N. (2023).** "Learning to Generate Text in Arbitrary Writing Styles." arXiv:2312.17242.

- **Mechanism:** Combines an author-adapted language model with sequence-level inference using contrastively-trained stylometric representations to generate text in a target author's style.
- **Effectiveness:** Effective for both unconditional generation and style transfer; also works as an anonymization method (mask authorship while preserving meaning).
- **Verdict:** Worth implementing as a reference style-transfer baseline if voice-preservation mode demands fine-grained style matching.

### 3.3 LoRA fine-tuning on user style corpus

- **Mechanism:** User provides a corpus of their own past writing. A LoRA adapter is fine-tuned on that corpus and used to bias the paraphraser toward their voice.
- **Effectiveness:** Highly effective for voice preservation but requires the user to have a substantial corpus of their own writing (~5000+ words minimum).
- **Cost:** One-time training (10-30 minutes on consumer GPU); inference is cheap.
- **Verdict:** Recommended as an advanced feature for committed users. Use PEFT library, target a small base model like Llama-3.2-3B.

---

## 4. Statistical-feature smoothing

### 4.1 Burstiness injection

- **Mechanism:** Detect low sentence-length variance, then deliberately split or merge sentences to increase variance. Uses a target burstiness distribution drawn from a reference human corpus.
- **Effectiveness:** Defeats burstiness-based detectors (GPTZero, ZeroGPT). Less effective against neural detectors.
- **Cost:** Fast, deterministic; no LLM call.
- **Verdict:** Cheap win. Should be a built-in mid-pipeline pass.

### 4.2 Perplexity smoothing

- **Mechanism:** Identify low-perplexity tokens (the most "predictable" words), substitute with higher-perplexity alternatives (less common synonyms). Optionally add controlled rare-word insertions.
- **Effectiveness:** Defeats DetectGPT-class detectors. Less effective against modern detectors that have moved past pure perplexity.
- **Cost:** Requires a reference LM for perplexity scoring; one forward pass per token.
- **Verdict:** Modest contributor; combine with other techniques.

### 4.3 Stylometric feature targeting

- **Mechanism:** Compute the target text's stylometric feature vector (function-word frequencies, POS tag distribution, dependency-tree depth, punctuation distribution) and the AI-text feature vector. Optimize edits to move the AI text's vector toward the target.
- **Effectiveness:** Defeats interpretable stylometric classifiers; mixed against neural detectors.
- **Cost:** Moderate; requires StyloMetrix or comparable library.
- **Verdict:** Important for preserve-voice mode where the user has supplied a target style corpus.

---

## 5. Detector defenses (so we know what we're fighting)

### 5.1 Retrieval-based defense

Krishna et al. 2023 [Krishna2023] propose: maintain a database of all generations the LLM API has produced. To check whether a candidate text is AI-generated, retrieve semantically similar entries from the database. Robust to paraphrase (the semantic retrieval still finds matches).

- **Detection of paraphrased generations:** 80-97% in their experiments.
- **False positive rate on human writing:** ~1%.
- **Limitation:** Requires the model provider to maintain and expose the retrieval database. OpenAI, Anthropic, and Google have not done this publicly. **Unimplementable in practice.**

### 5.2 Paraphrase inversion defense

**Rivera Soto, R., Chen, B., & Andrews, N. (2024).** "Mitigating Paraphrase Attacks on Machine-Text Detectors via Paraphrase Inversion." arXiv:2410.21637.

- Train a model to reverse paraphrasing (translate paraphrased text back toward the original AI text). Then run the detector on the inverted text. Improves detector AUROC by ~22% on average across seven detectors.
- **Implication:** As detectors deploy paraphrase-inversion preprocessing, single-pass paraphrasing will become weaker. Iterative and adversarial paraphrasing remain robust because the inversion target is ambiguous.

### 5.3 Synthetic-mirror training

Pangram's training method [Emi2024]: generate a paraphrased "mirror" of every AI training example and use it as a hard negative. This is precisely a defense against the dominant attack technique.

- **Implication:** Pangram is unusually resilient to paraphrasing-based attacks. To defeat Pangram specifically, the pipeline likely needs detector-guided iterative paraphrasing (Cheng et al. 2025) with Pangram itself as the guidance signal — which requires API access.

---

## 6. Commercial humanizers: a survey

The commercial market is large and the methodology is opaque. The following entries are based on independent reviews [Beebom2024, Lifehacker2024, HumanizerPro2025] and vendor documentation. Pricing reflects approximate 2024-2025 levels.

### 6.1 Undetectable.ai

- **Tagline:** "Bypass AI detectors."
- **Mechanism (inferred):** LLM paraphraser with prompt-engineered "human style" instructions. Some evidence of detector-feedback loop (the company likely tests against multiple detectors during training).
- **Pricing:** From $9.99/mo for 10k words.
- **Effectiveness:** Independent reviews report partial success against GPTZero, Originality.ai. Caught reliably by Pangram. Quality degradation noticeable in aggressive mode.

### 6.2 StealthGPT

- **Tagline:** "AI content that bypasses detection."
- **Mechanism:** LLM paraphrasing with style imitation. Markets itself as offering a "humanizer" that mimics human imperfections (varied sentence length, occasional minor grammar quirks).
- **Pricing:** From $14.99/mo.
- **Effectiveness:** Comparable to Undetectable.ai. Works on simpler detectors, struggles with advanced ones.

### 6.3 Humbot

- **Mechanism:** Multi-mode rewriting (basic / enhanced / aggressive). Bundles humanizer with plagiarism checker, translator, summarizer. API available.
- **Pricing:** Free tier (~150 words); paid plans from ~$10/mo.
- **Effectiveness:** Variable. Passes Quillbot, ZeroGPT, Originality.ai in some tests; fails GPTZero in others. Inconsistent run-to-run.

### 6.4 Quillbot Humanizer

- **Mechanism:** Specialized output mode of the Quillbot paraphraser. Synonym substitution + sentence restructuring + tone variation.
- **Pricing:** Quillbot Premium from $9.95/mo.
- **Effectiveness:** Improves readability over raw AI text but unreliable at fooling modern detectors.

### 6.5 GPTinf

- **Tagline:** "SEO-friendly undetectable AI content."
- **Mechanism:** Synonym swap + sentence restructuring + syntactic randomization.
- **Pricing:** Subscription model.
- **Effectiveness:** Among the better simple humanizers for short SEO copy. Quality and coherence sometimes suffer.

### 6.6 Phrasly, BypassGPT, WriteHuman, AI Humanize, etc.

Long tail of similar products. All use the same basic technique stack: LLM paraphrase + synonym substitution + light style imitation. Pricing $5-30/mo. Effectiveness clusters around "defeats older detectors, struggles with new ones, sometimes degrades quality."

### 6.7 Cross-cutting observations on commercial humanizers

- **Closed-source.** No published methodology.
- **Variable per-run output.** Many produce different output for the same input on different days, making them unreliable for high-stakes use.
- **Behind the academic state of the art.** None implement detector-guided iterative paraphrasing (Cheng et al. 2025) in any documented way.
- **Aggressive humanization mode degrades quality.** A consistent reviewer complaint.
- **Subscription-locked.** Even basic functionality is paywalled.
- **No voice preservation.** None of them treat the writer's existing voice as something to keep.
- **No transparency.** The user cannot see what the tool changed, why, or against which detectors it was tested.

The HumanizeMCP value proposition in this market: open source, MCP-native (use from any AI tool you already use), implements the published academic state of the art, transparent (shows diffs and detector scores before and after), preserves voice, free.

---

## 7. Open-source code that exists

A grep through public GitHub for relevant repositories:

- **DIPPER (Krishna et al.):** Released by the authors. Available as `kalpeshk2011/dipper-paraphraser-xxl` on HuggingFace. Strong starting point.
- **Fast-DetectGPT (Bao et al.):** https://github.com/baoguangsheng/fast-detect-gpt — the detector but useful as an evasion-testing oracle.
- **Ghostbuster (Verma et al.):** https://github.com/vivek3141/ghostbuster — detector code and datasets.
- **Binoculars (Hans et al.):** https://github.com/ahans30/Binoculars — detector code.
- **HC3 / chatgpt-comparison-detection:** https://github.com/Hello-SimpleAI/chatgpt-comparison-detection — datasets, detectors, evaluation code.
- **PADBen:** https://github.com/JonathanZha47/PadBen-Paraphrase-Attack-Benchmark — paraphrase-attack benchmark.
- **StealthRL:** https://github.com/suraj-ranganath/StealthRL — RL framework for adversarially paraphrasing AI text.
- **AuthorMist:** referenced in the paper; likely on GitHub.

A search for explicit "humanize AI text" repositories returned 125+ hits, mostly small individual projects of variable quality. None we surveyed implement the full pipeline (detector-guided + voice preservation + MCP interface) that HumanizeMCP proposes.

---

## 8. Synthesis: the recommended technique stack

In dependency-implementation order (build them in this sequence):

1. **Surface-tell remover.** Cosmetic substitutions: em dash → comma/period, "delve cluster" → curated synonyms, conversational AI scaffolding ("Great question!", "Let me know if...") → strip, Markdown bleed → strip. Fast, deterministic, no LLM call. *Defeats casual reader inspection.*

2. **Burstiness injector.** Statistical pass: detect low sentence-length variance, split or merge sentences to inject variance using a target distribution. *Defeats GPTZero, ZeroGPT.*

3. **Watermark scrubber.** Unicode normalization (NFKC), zero-width character removal, homoglyph normalization. *Removes embedded watermark signals (passive defense, not active attack).*

4. **DIPPER paraphrase pass.** Single controlled paraphrase, preserves semantics. *Defeats DetectGPT-class detectors.*

5. **Detector-guided iterative paraphrasing.** Cheng et al. 2025 pattern: paraphrase → score with Fast-DetectGPT and/or Binoculars → re-paraphrase if above threshold. *Defeats most modern detectors.*

6. **Optional back-translation pass.** Through one or more pivot languages. *Adds robustness against retrieval-based defenses and watermarks.*

7. **Style-transfer pass (preserve-voice mode).** Few-shot prompted from user-supplied style corpus, optionally fine-tuned LoRA adapter. *Restores user voice that earlier passes may have flattened.*

8. **Stylometric feature smoothing.** Optional final pass: compute target stylometric feature vector from user corpus or chosen preset, optimize for closeness. *Defeats interpretable stylometric detectors.*

This stack composes the published state-of-the-art techniques into a single pipeline. Each step is independently tested and ablatable. The pipeline can be run end-to-end or partially (e.g., just steps 1-3 for an autistic writer who wants to preserve everything except the most-flagged surface tells).

The empirical claim from the literature, particularly Cheng et al. 2025, is that this stack defeats every detector that is available at scale and exposes a score signal. Pangram and Turnitin (which do not expose meaningful internals) will be harder. The honest projection is "high reduction in detection probability, not zero, and the situation is an arms race that will continue."
