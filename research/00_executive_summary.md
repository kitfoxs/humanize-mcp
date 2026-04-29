# HumanizeMCP — Research Dossier: Executive Summary

**Project:** HumanizeMCP, an open-source Model Context Protocol server that rewrites AI-generated prose into text that reads as human-authored, while preserving (when the user requests it) idiosyncratic features of neurodivergent, ESL, and academic writers that current detectors systematically misclassify.

**Document status:** Internal research dossier, version 1.0. Compiled from peer-reviewed literature (2023-2026), independent benchmark studies, and primary source documentation. Citations are gathered in `references.bib`.

---

## 1. Problem statement

Large language model (LLM) detectors are now embedded in academic, employment, publishing, and commercial gatekeeping workflows. Every major detector on the market — GPTZero, Originality.ai, Turnitin, Copyleaks, ZeroGPT, Sapling, Pangram — markets accuracy figures above 95%, and several claim "less than 1% false positive rate." Independent peer-reviewed benchmarks tell a different story:

- The Stanford study by Liang et al. (2023), published in *Patterns* (Cell Press), found that seven widely deployed GPT detectors classified **61.3% of TOEFL essays written by non-native English speakers as AI-generated**, while correctly classifying essays by US-born 8th graders [Liang2023].
- The RAID benchmark (Dugan et al., 2024), the largest open benchmark for machine-generated text detection (6 million generations, 11 models, 8 domains, 11 adversarial attacks), found that "current detectors are easily fooled by adversarial attacks, variations in sampling strategies, repetition penalties, and unseen generative models" [Dugan2024].
- A 2026 Springer study on academic integrity contexts found Originality.ai's real-world false positive rate on formal academic writing reached 14%, an order of magnitude above the company's stated 0.5-1.5% [SpringerEduIntegrity2026].
- Independent retesting of ZeroGPT in 2026 measured a 26.4% false positive rate on student essays [Phrasly2026], and over 21% on ESL writers versus 9% on native speakers [HumanizeAIPro2026].

The detectors are simultaneously **brittle against trivial paraphrase attacks** and **systematically biased against marginalized writers** whose prose happens to share the surface statistical signature that detectors associate with LLM output — low perplexity, restricted vocabulary, formal register, consistent sentence structure.

This is the gap HumanizeMCP fills.

## 2. Strategic positioning

HumanizeMCP is framed and built as an **accessibility and equity tool**, not as an academic-fraud enabler. The framing is grounded in the empirical record:

- **Neurodivergent writers**, particularly autistic adults, produce prose with reduced lexical diversity, more concrete vocabulary, repeated phrasal structures, and lower burstiness than the neurotypical norm — exactly the features detectors flag as "AI-like" [autistic-prose-overview].
- **ESL writers**, and particularly L2 academic writers, produce prose with simpler syntax, lower type-token ratio, and "textbook" grammatical constructions because they were taught from textbooks. The Stanford 2023 study's 61.3% false positive rate is the canonical citation; ChatGPT-as-Linguistic-Equalizer (Lin et al., 2025) shows the inverse problem from 2.8M biomedical articles — that NNES authors who use LLMs to polish their writing now get misclassified [Lin2025].
- **Academics in formal disciplines** produce prose that is by design uniform in tone, low in burstiness, and high in discourse-marker density — the same signature.

The user the system is designed for is the writer who needs to **defeat a detector that should never have been deployed in the first place**, in order to prevent a harmful real-world outcome (failing grade, retracted paper, denied job application, deplatformed essay). The system is not designed for a student who wrote nothing and pasted ChatGPT output into a graded assignment. The technical pipelines support both populations identically — that is unavoidable, and is the same situation faced by Tor, Signal, VPN providers, and every paraphrasing tool that has ever existed — but the framing, documentation, defaults, and example presets are oriented toward the former.

## 3. The state of the art

### 3.1 Detection

Modern detectors fall into four families:

1. **Perplexity / log-probability based** (DetectGPT [Mitchell2023], Fast-DetectGPT [Bao2023], Binoculars [Hans2024], GLTR). Compute the likelihood of a passage under a reference LM; AI text occupies negative-curvature regions of the log-probability function. These are now partially obsolete — modern instruction-tuned LLM output sometimes has *higher* perplexity than human text (polarity inversion) [Baidya2026].
2. **Supervised neural classifiers** (RoBERTa-OpenAI-detector, Pangram [Emi2024], Ghostbuster [Verma2023], commercial classifiers like Originality.ai, GPTZero, Copyleaks). Fine-tuned transformers on labeled human/AI corpora. Highest in-domain accuracy, brittle out-of-domain.
3. **Stylometric / linguistic feature** (XGBoost-on-stylometric-features baselines, MHFD [Zhang2025], LIWC-based studies). Hand-engineered features over lexical/syntactic/punctuation patterns. Interpretable but matched in accuracy by neural methods on in-domain data; recent work (MHFD) found multi-feature integration provides only 0.4-0.5% lift over single neural models.
4. **Watermarking** (Kirchenbauer et al. 2023 green-list; ensemble watermarks [Niess2024]; adaptive watermarks [Liu2024]). Embed signals at generation time. Only available when the model provider cooperates; defeated by translation [He2024], paraphrasing, and homoglyph attacks [Creo2024].

### 3.2 Evasion

The peer-reviewed evasion literature is substantially more advanced than commercial humanizers. Key results:

- **DIPPER** (Krishna et al., 2023): an 11B parameter paraphrase model that drops DetectGPT accuracy from 70.3% to 4.6% at 1% FPR while preserving semantics [Krishna2023].
- **Adversarial Paraphrasing** (Cheng et al., 2025): a training-free framework using an off-the-shelf instruction-tuned LLM guided by a detector reward signal; reduces true positive rate by 87.88% on average across diverse detectors, including a striking 98.96% on Fast-DetectGPT [Cheng2025].
- **AuthorMist** (David & Gervais, 2025): a 3B parameter model fine-tuned with Group Relative Policy Optimization (GRPO) using detector APIs as the reward signal; achieves 78.6-96.2% attack success against individual detectors while keeping semantic similarity above 0.94 [David2025].
- **Back-translation attacks** (ESPERANTO, Ayoobi et al., 2024): translating through pivot languages drops detector TPR substantially [Ayoobi2024].
- **Homoglyph attacks** (SilverSpeak, Creo & Pudasaini, 2024): substituting visually identical Cyrillic/Greek codepoints for ASCII drops Matthews Correlation Coefficient from 0.64 to -0.01 across seven detectors [Creo2024].
- **Iterative paraphrasing** (PADBen, Zha et al., 2025): creates an "intermediate laundering region" that detectors fail to handle [Zha2025].

### 3.3 Commercial humanizers

The commercial market (Undetectable.ai, StealthGPT, Humbot, Quillbot Humanizer, GPTinf, Phrasly, BypassGPT) primarily relies on:
1. LLM paraphrasing with prompts like "rewrite to sound human."
2. Synonym substitution.
3. Sentence-length variation.
4. Injecting controlled imperfections.

Most are closed-source, charge subscription fees ($10-30/month), have no published methodology, and produce inconsistent results across detectors. Independent testing repeatedly finds that "humanized" output still triggers detectors a substantial fraction of the time, and that aggressive humanization mode degrades readability [Beebom2024, Lifehacker2024].

## 4. The HumanizeMCP gap

There is currently **no open-source MCP server** that:

1. Implements the published peer-reviewed evasion techniques (DIPPER-style controlled paraphrase, adversarial paraphrasing, back-translation, perplexity smoothing) in a composable pipeline.
2. Exposes them through a Model Context Protocol interface, so any MCP-aware tool (Claude Code, Copilot CLI, Continue, Zed, Cursor, etc.) can call them.
3. Is explicitly framed and documented around accessibility for neurodivergent, ESL, and formal-academic writers.
4. Includes a **preserve-voice mode** that maintains identifying stylistic features the user wants to keep while modifying only the surface features the detectors latch onto.
5. Ships with **detector self-evaluation** (a benchmark harness against open-source detectors and, optionally, commercial APIs the user supplies a key for) so users can measure rather than guess whether the rewrite worked.

HumanizeMCP fills that gap.

## 5. Headline recommendations

Concrete recommendations for the Pipeline Builder agent are in `06_implementation_recommendations.md`. Briefly:

- **Primary technique:** detector-guided iterative paraphrasing (Cheng et al. 2025 pattern), implemented as a loop over a local LLM and an open-source detector signal (Binoculars or Fast-DetectGPT). This is the empirically dominant approach.
- **Secondary techniques:** stylometric feature smoothing (sentence-length variance injection, vocabulary substitution from a target-style corpus), discourse-marker thinning (delve/leverage/intricate/tapestry/moreover removal), em-dash and parallel-structure suppression, controlled punctuation imperfection.
- **Voice preservation:** few-shot prompting from a user-supplied style corpus, optional fine-tuned LoRA adapter per user.
- **Self-evaluation:** ship Binoculars + Fast-DetectGPT + a stylometric XGBoost baseline as built-in scorers; provide adapters for commercial detector APIs (GPTZero, Originality.ai, Sapling) when the user supplies credentials.
- **Library stack:** `transformers` + `nltk` + `spacy` + `textstat` + `sentence-transformers` + Hugging Face `datasets` for corpora.
- **Calibration corpora:** HC3, RAID, MultiSocial for AI-vs-human reference; Project Gutenberg, blog_authorship_corpus, pushshift-reddit subsets, and Enron for human-style reference.

The remaining files in this dossier expand each of these threads.
