# 01 — Detector Landscape

A comprehensive map of the AI-text detection ecosystem as of late 2025 / early 2026. Each entry documents methodology (where known), peer-reviewed and independent accuracy data, false positive rates, pricing, and API availability. Where vendor claims diverge from independent benchmarks, both are reported with the gap explicitly flagged.

The single most important framing fact in this section: every commercial detector advertises 95%+ accuracy and "less than 1% false positives." Every independent peer-reviewed benchmark of those same detectors reports substantially higher false positive rates on formal academic writing, ESL writing, and any text that has been subject to even trivial paraphrase. The vendor numbers are derived from in-distribution test sets that look like their training data; the academic numbers are derived from out-of-distribution writing that resembles what real users actually submit.

---

## 1. Commercial detectors

### 1.1 GPTZero

- **Vendor:** GPTZero, Inc. (Edward Tian et al., founded 2023).
- **Public methodology:** GPTZero analyzes two statistical properties: **perplexity** (how surprising each token is to a reference language model — AI text tends to have low, uniform perplexity) and **burstiness** (variance in sentence-to-sentence perplexity and length — humans alternate between simple and complex constructions, AI does not). The classifier returns a sentence-level probability score and an aggregate document score [GPTZero2024methodology]. Newer versions claim to incorporate "context-aware" features and additional stylometric signals, but the underlying public mechanism remains perplexity + burstiness.
- **Vendor accuracy claim:** ~98%.
- **Independent measurement:** ~88% accuracy, ~9% false positive rate on mixed corpora [QuickCreator2025]. The Stanford TOEFL study found GPTZero misclassified 61.3% of non-native-English-speaker essays as AI-generated [Liang2023].
- **API pricing (as of 2025):** Tiered subscription. Basic API ~$12.99/mo for 300,000 words/mo (~$0.000043/word). Mid tier ~$24.99/mo for 500,000 words/mo. Enterprise ~$100/mo for 2M words/mo [GPTZeroPricing].
- **API availability:** Yes, REST API with sentence-level highlighting.
- **Notes:** Most-recognized brand in the space. Founded explicitly for academic integrity use. The default starting point for any commercial detector evaluation.

### 1.2 Originality.ai

- **Vendor:** Originality.AI Inc.
- **Public methodology:** Closed. Reportedly a fine-tuned transformer classifier with proprietary training data. Marketed for SEO/publishing audiences first, academia second.
- **Vendor accuracy claim:** 96% overall; FPR < 1%.
- **Independent measurement:**
  - On the November 2025 "Falsely Accused" benchmark (arXiv:2511.16690), 96% overall accuracy, the highest of 4 commercial tools tested across 16,000 samples [FalselyAccused2025].
  - On the RAID benchmark (Dugan et al. 2024), 99%+ on raw AI text, 97% on humanized text — the strongest commercial result in that suite.
  - On formal academic writing, the 2026 Springer study found a real-world false positive rate of 14%, an order of magnitude above the vendor claim [SpringerEduIntegrity2026].
- **Pricing (2024-2026):** ~$0.01 per 100 words on credits ($0.0001/word). Subscription from $14.95/mo. The most cost-efficient commercial API at scale.
- **API availability:** Yes, well-documented REST API.
- **Notes:** Generally regarded as the strongest commercial detector for unedited LLM output and the most resistant to light human editing. Its weakness is that its absolute accuracy on legitimate formal human writing is overstated by a factor of ~10-30x in marketing materials.

### 1.3 Turnitin AI Detector

- **Vendor:** Turnitin LLC.
- **Public methodology:** Proprietary, but per Turnitin's own documentation, "the model was trained to detect content from GPT-3 and GPT-3.5" [TurnitinDocs]. It applies machine-learning analysis of word choice, sentence structure, and predictability. Results are presented to instructors as a percentage of the document estimated to be AI-written.
- **Vendor accuracy claim:** "Less than 1% false positive rate" with caveat that results "should not be used as the sole basis" for academic misconduct allegations [TurnitinSupport].
- **Independent measurement:**
  - The 2026 Springer educational-integrity study found Turnitin lagging Originality.ai on raw AI-text accuracy (0.61 vs. 0.69) and noted high false-positive rates on technical and formal genres [SpringerEduIntegrity2026].
  - Multiple educator-reported cases of human-written work flagged as AI-generated, particularly for international and ESL students.
- **Pricing:** Institutional only, no individual or pay-as-you-go API. School pricing typically $3-8 per student per year, AI detection bundled into broader plagiarism-detection contract.
- **API availability:** Limited; LMS integration via institutional contract only. Not available for direct developer integration.
- **Notes:** The most consequential detector by deployment surface area, since it is built into the dominant academic plagiarism-detection product. Its non-availability as a developer API means HumanizeMCP cannot include it as a built-in scorer; the only viable approach is to evaluate against detector families that correlate with it.

### 1.4 Copyleaks

- **Vendor:** Copyleaks Ltd.
- **Public methodology:** Neural network classifier trained on multi-source human/AI text. Combines perplexity, burstiness, "linguistic fingerprinting," and database lookup against reference corpora. Provides sentence-level confidence scores. Supports 30+ languages [CopyleaksMethodology].
- **Vendor accuracy claim:** 99% on AI-generated text, 99.4% on human text, 0.2% FPR (lab conditions).
- **Independent measurement:** ~91% real-world accuracy, 6-7% false positive rate. Drops sharply on humanized AI output and on hybrid (human+AI) content.
- **Pricing:** From ~$13.99/mo. Credit-based scans (1 credit per 250 words). API plans custom for enterprise.
- **API availability:** Yes, with strong developer documentation.
- **Notes:** Among the better detectors for multilingual content. Aggressive on hybrid texts, often classifying lightly-edited AI as fully AI to remain conservative.

### 1.5 ZeroGPT

- **Vendor:** ZeroGPT (small startup).
- **Public methodology:** Perplexity + burstiness, very similar to GPTZero. Threshold of >80% AI score flags text as "likely AI." No published technical paper.
- **Vendor accuracy claim:** Up to 98% [ZeroGPTSelf].
- **Independent measurement:** The most-criticized detector in the independent literature. Multiple 2025-2026 studies report:
  - Overall accuracy 73-85% [SupWriter2026].
  - False positive rate 10-26% across studies [Rewritify2026, Phrasly2026, NaturalRewrite2026].
  - 21% FPR on ESL writers vs. 9% on native English writers [HumanizeAIPro2026] — among the most biased detectors measured.
  - One mega-study of 30,000 student essays measured a 26.4% false positive rate [Phrasly2026].
- **Pricing:** Free tier; paid plans for higher quota.
- **API availability:** Yes, lightweight API.
- **Notes:** Cited here primarily as a cautionary example. ZeroGPT is widely used by educators because it is free, but its false-positive rate makes it materially harmful as a high-stakes decision aid.

### 1.6 Sapling AI Content Detector

- **Vendor:** Sapling Intelligence.
- **Public methodology:** Transformer-based neural network. Outputs a probability and sentence-level highlighting. API supports document types (PDF, DOCX) and bulk analysis [SaplingDocs].
- **Vendor accuracy claim:** 97%.
- **Independent measurement:** ~92% on third-party benchmarks. Editorial false-positive rates 3-15% depending on text type. Some academic studies found false-positive rates up to 35% on highly formal human writing [TexasAMSapling]. Detection drops to 81% after light human editing, 65-78% after paraphrase.
- **Pricing:** Free tier; paid plans starting around $25/mo.
- **API availability:** Yes.
- **Notes:** Among the strictest detectors — high recall on raw AI text, but at the cost of high false positives. Frequently flagged in reviews as among the most biased against formulaic human writing.

### 1.7 Pangram

- **Vendor:** Pangram Labs (Bradley Emi & Max Spero).
- **Public methodology:** **The only commercial detector with a peer-reviewed technical report** [Emi2024, arXiv:2402.14873]. Pangram is a transformer classifier trained with a novel "hard negative mining with synthetic mirrors" algorithm: for each candidate AI text, generate a paraphrased "mirror" text and use it as a hard negative example in training. This explicitly targets the paraphrase-evasion attack class.
- **Vendor accuracy claim:** 99.98%, 0.01% FPR (1 in 10,000).
- **Independent measurement:**
  - The technical report itself reports >38x lower error rates than GPTZero, Originality.ai, and DetectGPT across 10 text domains and 8 LLMs [Emi2024].
  - On the COLING/RAID shared task, Pangram achieved 99.3% accuracy, the highest score [Pangram2026].
  - Independent academic comparison: best-in-class on ESL bias as well — the technical report explicitly tests for and finds no significant bias against non-native English speakers.
- **Pricing:** Subscription model; enterprise pricing.
- **API availability:** Yes.
- **Notes:** Currently the strongest commercial detector by independent measurement. The synthetic-mirror training approach is also the most direct counter-attack against paraphrase-based humanization. Any humanization technique that does not specifically target Pangram's training distribution is unlikely to defeat it.

### 1.8 Other commercial detectors (briefly)

- **Winston AI:** Marketed for publishers; perplexity-based; broadly similar profile to Originality.ai but with weaker independent benchmarks.
- **Content at Scale:** Hybrid AI/plagiarism tool; mid-tier accuracy.
- **Crossplag:** EU-focused, multilingual.
- **AI Text Classifier (OpenAI, deprecated 2023):** OpenAI's own first-party detector. Withdrawn due to "low rate of accuracy" — a notable admission given OpenAI built the model being detected.

---

## 2. Open-source / academic detectors

### 2.1 RoBERTa-base OpenAI Detector (HuggingFace)

- **Citation:** Solaiman et al. 2019, OpenAI release.
- **Methodology:** RoBERTa-base fine-tuned on outputs from GPT-2 1.5B vs. WebText human samples.
- **Reported accuracy:** ~95% on its original test set of 510-token GPT-2 outputs [HFModelCard].
- **Real-world:** Useless against modern LLMs (GPT-3.5+, Claude, Llama 3+, Gemini). The most-cited "open-source detector," but the field has moved past it.
- **Use case for HumanizeMCP:** Historical baseline; potentially useful as a reward signal in adversarial paraphrasing because of how stylistically distinct GPT-2 output is.

### 2.2 DetectGPT

- **Citation:** Mitchell et al. 2023 [Mitchell2023, arXiv:2301.11305].
- **Methodology:** Zero-shot. Computes log-probability of a passage under a reference LLM, then perturbs the passage (mask-and-fill with T5) and re-computes. AI text occupies negative-curvature regions of the log-prob landscape — small perturbations decrease the probability. Human text does not have this property.
- **Reported accuracy:** 0.95 AUROC on detecting GPT-NeoX 20B fake news, vs. 0.81 for the strongest zero-shot baseline.
- **Limitation:** Computationally expensive (multiple perturbations per passage). DIPPER drops it to 4.6% TPR at 1% FPR.

### 2.3 Fast-DetectGPT

- **Citation:** Bao et al. 2023 [Bao2023, arXiv:2310.05130].
- **Methodology:** Replaces DetectGPT's perturbation step with a sampling step using a "conditional probability curvature" formulation. Same zero-shot principle, ~340x faster, ~75% better accuracy in white-box and black-box settings.
- **Use case for HumanizeMCP:** **Recommended primary built-in scorer.** Open-source, runs locally, fast enough for iterative paraphrasing loops.

### 2.4 Binoculars

- **Citation:** Hans et al. 2024 [Hans2024, arXiv:2401.12070].
- **Methodology:** Zero-shot. Uses the ratio between two LLMs' perplexities ("observer" and "performer" model) as the detection statistic. The intuition is that AI text has consistent perplexity under both observers, while human text diverges.
- **Reported accuracy:** State-of-the-art zero-shot method on its release; defeated by paraphrase attacks like every other zero-shot method.
- **Use case for HumanizeMCP:** **Recommended secondary built-in scorer.** Complementary to Fast-DetectGPT; together they cover the dominant zero-shot detection approaches.

### 2.5 Ghostbuster

- **Citation:** Verma et al. 2023 [Verma2023, arXiv:2305.15047].
- **Methodology:** Pass document through several "weaker" LMs (smaller GPT-2 variants), extract token probabilities, run a structured search over feature combinations, train a classifier. Crucially does not require access to the target LLM — useful for black-box settings.
- **Reported accuracy:** 99.0 F1 cross-domain, 5.9 F1 above the next-best detector at release. Generalizes well to unseen prompting strategies and writing domains.
- **Notes:** Released with three new datasets (student essays, creative writing, news). Among the most rigorously evaluated open-source detectors.

### 2.6 GLTR (Giant Language Model Test Room)

- **Citation:** Gehrmann, Strobelt, Rush 2019 (HarvardNLP & MIT-IBM).
- **Methodology:** Visualization tool. Color-codes each token by its rank in the predicted distribution under GPT-2: green (top 10), yellow (11-100), red (101-1000), purple (>1000). Human text exhibits more rainbow; AI text is mostly green/yellow.
- **Use case for HumanizeMCP:** Diagnostic / debugging tool — useful for visualizing which tokens of a generated text the detector is "reading" as suspicious.

### 2.7 RADAR

- **Citation:** Hu, Chen, Ho 2023.
- **Methodology:** Adversarially trained: a paraphraser and a detector are trained jointly, where the paraphraser tries to evade the detector and the detector tries to catch the paraphrase. The resulting detector is more robust to paraphrase attacks than DetectGPT-class methods.
- **Use case for HumanizeMCP:** Important to be aware of — RADAR is the detector that adversarial paraphrasing in [Cheng2025] still beat (64.49% TPR reduction), so it is a meaningful baseline.

### 2.8 Other open-source detectors

- **Ghostbuster, Binoculars, Fast-DetectGPT** are the three most-cited open-source detectors in the 2024-2026 literature.
- **DeBERTa-based supervised classifiers** (Sarang at DEFACTIFY 4.0, Trivedi & Sivanesan 2025 [Trivedi2025]) achieve F1 = 1.0 in shared-task conditions but degrade out-of-domain.
- **MHFD** (Multi-Hierarchical Feature Detection) [Zhang2025]: integrates DeBERTa semantic + syntactic + statistical features. Important negative result: multi-feature integration provides only 0.4-0.5% improvement over single-neural-model baselines while costing 4.2x compute. Suggests modern neural detectors already capture most of the available signal.

---

## 3. Watermarking approaches

Watermarking is fundamentally different from detection: it embeds a signal at generation time and recovers it later. It only works when the model provider cooperates.

- **Kirchenbauer et al. 2023** [Kirchenbauer2023, arXiv:2301.10226]: The seminal "green-list" watermark. Before generating each token, the LM hashes the previous token to produce a pseudo-random green/red split of the vocabulary, then biases sampling toward green tokens. Detected via a statistical test on token frequencies.
- **Adaptive watermarking** [Liu2024]: Adds watermark only to high-entropy tokens, preserving quality.
- **Ensemble watermarks** [Niess2024]: Combines acrostica, sensorimotor norms, and green-red watermarks. Achieves 95% detection after paraphrase vs. 49% for green-red alone.
- **Cross-lingual watermark removal** [He2024]: Translating watermarked text into a pivot language and back removes most watermarks.
- **Unicode watermarking** [Hellmeier2025]: Invisible Unicode tags detected by reasoning models (GPT-5, Claude Sonnet 4) but not extractable.

**Implication for HumanizeMCP:** Watermark stripping should be a built-in step. Strip zero-width and bidirectional Unicode; normalize homoglyphs; consider light paraphrase as defense against statistical watermarks.

---

## 4. Benchmarks

### 4.1 RAID — Robust AI-text Detection

- **Citation:** Dugan et al. 2024 [Dugan2024, arXiv:2405.07940].
- **Scale:** 6 million generations, 11 LLMs, 8 domains, 11 adversarial attacks, 4 decoding strategies.
- **Key finding:** "Current detectors are easily fooled by adversarial attacks, variations in sampling strategies, repetition penalties, and unseen generative models."
- **Use case:** **The primary detector benchmark.** HumanizeMCP should evaluate its output against detectors using a RAID-style protocol.

### 4.2 PADBen — Paraphrase Attack Benchmark

- **Citation:** Zha et al. 2025 [Zha2025, arXiv:2511.00416].
- **Scale:** Evaluates 11 SOTA detectors against iteratively paraphrased text across 5 text types.
- **Key finding:** Detectors successfully identify "plagiarism evasion" (paraphrasing AI text to evade) but fail at "authorship obfuscation" (paraphrasing human text — they cannot tell whether human-paraphrased text is human or AI).

### 4.3 SemEval-2024 Task 8

- **Citation:** Wang et al. 2024 [Wang2024, arXiv:2404.14183].
- **Scope:** Multidomain, multimodel, multilingual machine-generated text detection. Three subtasks: binary detection (mono and multilingual), source attribution, change-point detection in hybrid texts. 285 participating teams across all subtasks.

### 4.4 MAGE / M4 / MultiSocial / CEAID

Multilingual machine-generated text detection benchmarks: M4 covers 10+ languages with 6 LLMs; MultiSocial [Macko2024] covers 22 languages and 5 social media platforms; CEAID [Macko2025] covers Central European languages. Important because most commercial detectors are English-only or perform sharply worse on other languages.

### 4.5 HC3 — Human ChatGPT Comparison Corpus

- **Citation:** Guo et al. 2023 [Guo2023, arXiv:2301.07597].
- **Scale:** Tens of thousands of paired human/ChatGPT responses across open-domain, finance, medicine, law, psychology.
- **Use case:** Reference corpus for detector training and evaluation. Available on HuggingFace as `Hello-SimpleAI/HC3`.

### 4.6 "Falsely Accused" benchmark

- **Citation:** arXiv:2511.16690 (November 2025).
- **Scope:** 18 detectors (14 LLMs, 4 commercial), 16,000+ samples.
- **Key finding:** Originality.ai best of commercial tools at 96%, lowest FP rate; multilingual content (Arabic) particularly challenging.

---

## 5. Cross-cutting findings from the benchmark literature

1. **Vendor-claimed accuracy substantially overstates real-world false-positive rates**, particularly on formal academic prose, ESL writing, and technical writing. The gap is often 10-30x.
2. **Zero-shot detectors (DetectGPT family) are obsolete against paraphrase attacks.** Modern adversarial paraphrasing reduces their TPR by 87-99%.
3. **Supervised neural detectors (Pangram, Ghostbuster, fine-tuned RoBERTa/DeBERTa) generalize poorly** out of training distribution. Cross-domain accuracy can drop 10-20 percentage points [Pudasaini2026].
4. **"In-the-wild" detector accuracy is materially worse** than in published benchmarks. The benchmarks themselves are typically in-distribution; real student writing, ESL writing, and writing by writers with stylistic idiosyncrasies (autistic prose, formal academic prose) is out-of-distribution.
5. **Watermarking is the only reliable detection mechanism**, but it requires generator cooperation and is defeated by translation and homoglyph attacks.
6. **Polarity inversion is real:** modern instruction-tuned LLMs sometimes produce text with *higher* perplexity than human text, breaking the foundational assumption of perplexity-based detectors [Baidya2026].
7. **The detection-evasion arms race is structurally unwinnable for the detector.** Cheng et al. (2025) and David & Gervais (2025) demonstrate that detector-aware adversarial paraphrasing achieves near-100% evasion against any detector that exposes a score, including the most accurate commercial detectors. Detectors that do not expose scores (Pangram-class) are harder to attack but not immune.

---

## 6. Implications for HumanizeMCP

1. **Built-in scorers should include Fast-DetectGPT and Binoculars** as zero-shot baselines, plus a stylometric XGBoost classifier as a third interpretable signal. These cover the three open-source detector families.
2. **Optional adapters for commercial detectors** (GPTZero, Originality.ai, Sapling) when the user supplies an API key. Pangram should be supported when its API becomes available; Turnitin should be acknowledged as un-targetable directly and documented as such.
3. **Self-report a confidence interval** rather than a single "AI score." Detectors disagree, and the user should see that disagreement.
4. **Test against detector-aware paraphrasing as a quality benchmark.** If output passes Fast-DetectGPT but fails GPTZero, the pipeline needs more iterations. If it passes both but fails an interpretable stylometric classifier, the surface text has changed but the structural fingerprint hasn't.
5. **Default to detector-guided iterative paraphrasing** (Cheng et al. 2025 pattern). It is the empirically dominant evasion technique and works against the broadest range of detectors.
