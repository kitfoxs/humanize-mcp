# 03 — Human Writing Diversity That Detectors Wrongly Flag

The detectors in section 01 are trained to distinguish "AI text" from "human text." In practice, the "human text" their training corpora draw from is overwhelmingly **native-English, neurotypical, native-medium-register prose written by US/UK undergraduates and journalists**. Writers who diverge from that distribution, in either direction, get systematically misclassified.

This section documents the specific human-writing populations that current detectors falsely flag, with citable evidence. The HumanizeMCP design imperative that emerges from this section is the **preserve-voice mode**: a default that distinguishes between "remove the AI-coded surface features" and "erase the writer's identity." We want the former, never the latter.

---

## 1. Non-native English (ESL / EAL) writers

### 1.1 The Stanford 2023 study

The single most-cited piece of evidence in this entire dossier:

**Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., & Zou, J. (2023).** "GPT detectors are biased against non-native English writers." *Patterns* (Cell Press), 4(7), 100779. arXiv:2304.02819.

Key findings [Liang2023]:

- The team evaluated **seven widely-used GPT detectors** (including GPTZero, Originality.AI, Quil.org, OpenAI's then-active classifier, and academic open-source detectors) on:
  1. **TOEFL essays written by Chinese students** (human, non-native English).
  2. **8th-grade essays from US students** (human, native English).
  3. **GPT-generated essays** as a control.
- Result: the detectors classified the TOEFL essays as AI-generated at a rate of **61.3%** (one detector was as high as 97.8%). The same detectors classified the US 8th-grade essays correctly at 95%+.
- Mechanism: TOEFL essays exhibit lower vocabulary diversity, simpler syntax, and lower perplexity (because non-native writers reach for the safest, most common word choices that they are confident are correct). These are exactly the features detectors associate with LLM output.
- The team showed that asking ChatGPT to "elevate the language" of the TOEFL essays *reduced* their AI-detection score, because the now-native-sounding text no longer triggered the detector. Inversely, a "simplify the language" prompt on the US 8th-grade essays raised their AI-detection score.
- The team's conclusion: "GPT detectors may unintentionally penalize writers with constrained linguistic expressions."

**This is the foundational equity case for HumanizeMCP.** A tool that allows ESL writers to defend against this misclassification is an accessibility tool, not a fraud tool. The detector is the malfunctioning actor; the writer is the wronged party.

### 1.2 The follow-up evidence

**Lin, D., Zhao, N., Tian, D., & Li, J. (2025).** "ChatGPT as Linguistic Equalizer? Quantifying LLM-Driven Lexical Shifts in Academic Writing." arXiv:2504.12317 [Lin2025].

- Analyzed **2.8 million articles from OpenAlex (2020-2024)** using Measure of Textual Lexical Diversity (MTLD) with a difference-in-differences design.
- Finding: ChatGPT significantly *raised* the lexical complexity of NNES-authored abstracts, particularly in lower-tier journals, preprints, and technology/biology fields.
- Implication: The very NNES authors who would benefit most from LLM polishing are now disproportionately likely to be flagged by detectors trained to recognize that polishing.

**Al Ali, A., Helcl, J., & Libovický, J. (2026).** "Different Time, Different Language: Revisiting the Bias Against Non-Native Speakers in GPT Detectors." arXiv:2602.05769 [AlAli2026].

- A 2026 follow-up in the Czech-language setting. The team find that **the perplexity gap between native and non-native Czech writers is no longer present** in their corpus, and that contemporary detectors operate without relying on perplexity as the primary signal.
- Caveat: this is a single language (Czech), with a specific sample of writers, and the result is two years downstream of Liang et al. The authors do *not* claim the bias has been resolved in English or generally. They claim the situation has shifted.
- The honest summary: the bias was severe in 2023; some 2026 detectors have improved on it; commercial detectors (per the Springer 2026 study cited in section 01) still exhibit substantial bias on formal and technical writing, which heavily overlaps with NNES academic writing.

### 1.3 Patterns to preserve in ESL writing

Empirically observed patterns in ESL writing that detectors flag and that **HumanizeMCP must preserve when the user wants to keep them**:

| ESL pattern | Why detectors flag it | Why a human writer might keep it |
|-------------|----------------------|-----------------------------------|
| Lower type-token ratio (vocabulary repetition) | Maps to LLM "predictability" | The writer chooses words they trust; substitution would risk error. |
| Shorter mean sentence length | Maps to formulaic AI prose | The writer is being clear; long L2 sentences risk grammatical errors. |
| Limited use of phrasal verbs | Doesn't match L1 idiomatic prose | Writer learned from textbook English. |
| "Textbook" grammatical correctness (no contractions, formal register) | Maps to AI's clean grammar | Writer is being deliberate and respectful. |
| Direct, low-implication sentences | Maps to AI's literal style | Reflects the rhetorical norms of the writer's L1. |
| Specific transitional phrases learned in EAP coursework ("Furthermore," "In conclusion") | Maps to AI tells | Learned templates that work. |

**Preservation strategy:** Allow the user to mark these features as "preserve." The pipeline should reduce *only* the features that overlap with the AI signature without being part of the writer's deliberate voice. Burstiness can be increased without changing vocabulary; em dashes can be removed without changing sentence structure.

---

## 2. Autistic and otherwise neurodivergent writers

### 2.1 Documented patterns

Autistic writing is less well-studied at the corpus-stylometric level than ESL writing, but the patterns are well-documented in clinical and qualitative literature. Salient features include:

- **Lexical precision and concreteness.** Autistic writers tend to choose specific over general terms, avoid metaphor where literal description suffices, and use technical vocabulary in non-technical contexts.
- **Repetition for emphasis or accuracy.** What neurotypical readers see as "repetitive" prose may be the writer ensuring no ambiguity.
- **Topic perseveration.** A narrow set of interests treated in depth, sometimes within a single document.
- **Reduced burstiness.** A tendency toward more uniform sentence length, particularly in technical/expository registers.
- **Direct, unhedged statements.** Less softening, less politeness scaffolding, fewer "I think" / "you might say" insertions.
- **Heavy use of explicit logical connectives.** "Because," "therefore," "however," used at higher density than neurotypical baseline, to make reasoning chains explicit.
- **Unusual use of punctuation, including em dashes** for explicit clarification (interestingly, the same punctuation LLMs overuse, but for different reasons).

These features map almost directly onto the AI-detector signature: low perplexity (precise vocabulary), low burstiness (uniform sentences), explicit discourse markers, dense parallel structure. Autistic writers are at structurally elevated risk of being flagged.

### 2.2 Evidence base

We could not locate, in the searches conducted, a peer-reviewed corpus-scale study specifically on autistic prose vs. AI-detector classification. This is a genuine gap in the literature. What exists is:

- Clinical and qualitative literature on autistic writing style (Tager-Flusberg's work on autistic narrative; King et al. 2014 in *Language and Linguistics Compass* on language in ASD).
- Anecdotal and self-reported evidence from autistic writers who report being flagged (notable: a 2024 wave of autistic students publicly reporting Turnitin/Copyleaks false positives on social media; not yet aggregated into peer-reviewed research).
- Persona-Augmented Benchmarking (Le Truong et al. 2025 [LeTruong2025, arXiv:2507.22168]) showed that even semantically identical prompts in different writing styles produce wildly different LLM-as-judge outputs — confirming that detectors and evaluators are extremely sensitive to writing style. By extension, writing styles that diverge from the neurotypical norm are at elevated misclassification risk.

The honest framing in any HumanizeMCP documentation: "There is qualitative and anecdotal evidence that autistic prose patterns share statistical features with AI output that current detectors flag. Rigorous corpus-level confirmation does not yet exist, and we encourage academic work in this area." This is a research gap worth flagging publicly.

### 2.3 Patterns to preserve in autistic writing

**Critical:** these are features the writer values and that may be diagnostic of identity. They must not be erased.

| Autistic pattern | Preservation rationale |
|------------------|------------------------|
| Concrete, technical vocabulary | This is precision, not deficit. |
| Explicit logical connectives | These make reasoning followable. |
| Repeated phrasings for emphasis | Often deliberate. |
| Direct, unhedged statements | Clarity is a feature. |
| Long uninterrupted information dumps in topic of interest | This is *the prose voice.* |

The HumanizeMCP pipeline should let neurodivergent users opt out of:
- Sentence-length variance injection (which would chop or merge sentences they wrote deliberately).
- Hedge insertion (which would soften statements they meant to be direct).
- Vocabulary substitution toward general/idiomatic terms (which would lose precision).

It should opt them *in* to:
- Em-dash reduction (because em dashes are a distinct AI tell, even though autistic writers also use them).
- "Delve" / "tapestry" excess-vocabulary substitution (because these specific words are rare in genuine autistic writing and their presence is itself an LLM-edit signal).
- Burstiness adjustment via *paragraph* structure variance rather than sentence-level variance (preserves sentence-level voice).

---

## 3. Formal academic writers (any background)

### 3.1 The structural problem

Academic writing in the humanities, sciences, and social sciences is *constructed* to look like the AI signature:

- **Discipline-enforced uniformity of register.** Style guides (APA, Chicago, MLA, IEEE) eliminate burstiness on purpose.
- **Formulaic transitional vocabulary.** "Furthermore," "however," "in this section," "as discussed previously" are not optional, they are required.
- **Long sentences with low variance.** Driven by sentence-combining norms in formal writing.
- **High discourse-marker density.** "Therefore," "consequently," "thus," "as a result" are scaffolding required to make arguments followable.
- **Hedging is explicitly trained.** "These results suggest," "the data indicate," "it appears that" are the polite voice of scientific tentativeness, *and* the polite voice of LLMs trying not to overclaim.
- **Excess vocabulary is in the discipline lexicon.** "Underscore," "delve," "intricate" are not LLM-isms in academic writing — they were academic-isms first, which is why they ended up in RLHF reward signals.

### 3.2 Evidence

The 2026 Springer International Journal for Educational Integrity study found Originality.ai's false positive rate on formal academic writing reached **14%**, against a vendor claim of 0.5-1.5%. Both Originality.ai and Turnitin showed sharp accuracy declines on technical and formal genres and on longer documents [SpringerEduIntegrity2026].

Pudasaini et al. 2026 [Pudasaini2026, arXiv:2603.23146] used SHAP-based explanations to show that detectors that achieve F1 = 0.97 in-domain rely heavily on dataset-specific features, and that the features most discriminative on training data are also the most susceptible to domain shift. Formal academic writing is a domain shift relative to the open-domain training corpora most detectors use.

### 3.3 Patterns to preserve

| Academic pattern | Preservation rationale |
|------------------|------------------------|
| Discipline-required hedging | Mandated by editorial norms. |
| Formal discourse markers | Required for argument structure. |
| Discipline jargon (including "underscore," "leverage," "robust") | These have technical meanings in many fields. |
| Citation density and signal phrases ("Smith argues that...") | Genre constraint. |
| Long, complex sentences | Often required for precision. |

The HumanizeMCP "academic preservation" preset should:
- Suppress only the most-flagged surface tells (em dashes, "delve" cluster) without restructuring prose.
- Allow burstiness adjustment via sentence-merging at paragraph boundaries rather than within argumentative passages.
- Preserve all hedging.
- Optionally substitute one or two flagship excess-vocabulary words per paragraph (just enough to break the lexical fingerprint) without rewriting the whole document.

---

## 4. Other affected populations

### 4.1 Translators and bilingual writers

Translated text — even by professional human translators — exhibits "translationese": simplified syntax, normalized vocabulary, under-translation of idiom, source-language structural interference. These features overlap with the AI signature. There is no specific peer-reviewed study on translator misclassification rates by current detectors that we located, but the structural prediction is straightforward and is a known concern in translation studies.

### 4.2 Writers with cognitive impairments

Stroke survivors, traumatic brain injury survivors, and writers with specific dementias often produce prose with reduced lexical diversity, simplified syntax, and repetitive phrasing for memory-aid reasons. These map directly onto the detector signature. No corpus-scale studies on detector misclassification of these populations exist that we located. Another genuine research gap.

### 4.3 Children and adolescents

The Stanford study used US 8th-grade essays as the "human" baseline, and detectors classified them correctly. But that was the baseline they were trained on. ESL adolescents, neurodivergent adolescents, and adolescents writing in formal genres they have not yet mastered are at elevated risk. The deployment of detectors in K-12 contexts is therefore particularly fraught.

### 4.4 Writers using assistive technologies

Writers using speech-to-text, predictive-text keyboards, autocorrect, and grammar checkers (Grammarly, Microsoft Editor, ProWritingAid) are receiving LLM-style polishing on their prose passively. Whether the resulting text is "AI-written" is genuinely unclear, and detectors offer no nuance.

### 4.5 Writers who use LLMs for *legitimate* assistance

The most policy-relevant population. A writer who composes a draft and uses an LLM for grammar polishing, translation assistance, or section reorganization has done nothing wrong by any reasonable academic standard. The detector cannot distinguish them from a writer who pasted the LLM output unchanged. The Springer 2026 study found both Originality.ai and Turnitin "struggled with hybrid (AI/human mixed) texts," with accuracy dropping sharply.

---

## 5. The design imperative: preserve, don't erase

The throughline across every population in this section: **detector false positives concentrate on writers whose prose is precise, formal, repeated for emphasis, simple, or otherwise structurally regular.** That covers a large fraction of the world's writers. Building a humanizer that solves this by erasing all that structural regularity solves the detector problem at the cost of erasing the writer.

The HumanizeMCP design must therefore:

1. **Default to a "minimum-disruption" mode.** Remove the most-flagged surface tells (em dashes, excess vocabulary, conversational AI scaffolding) and adjust statistical surface features (burstiness, perplexity-curvature) without rewriting prose voice.

2. **Expose a "preserve" annotation system.** Let users explicitly mark passages, vocabulary patterns, sentence-structure preferences, and register choices that the pipeline must not modify.

3. **Ship presets per affected population:**
   - `preset.esl` — preserves simpler syntax and limited vocabulary, removes em dashes and the "delve" cluster, adjusts only burstiness and discourse-marker density.
   - `preset.autistic` — preserves precision and explicit logical connectives, allows paragraph-level burstiness adjustment only.
   - `preset.academic` — preserves all hedging and formal register, removes only flagship lexical and punctuation tells, allows light excess-vocabulary substitution.
   - `preset.casual` — full transformation toward a casual conversational style; for writers whose target voice is genuinely casual.

4. **Document the framing publicly.** The README, the MCP tool descriptions, and any associated essays should make explicit that this is a tool for writers being unjustly flagged, not a tool for academic fraud, and that the empirical case for the former is overwhelming.

5. **Acknowledge the dual-use reality honestly.** Any tool that defeats detectors can be used by anyone, including bad actors. The same is true of paraphrasing software, thesaurus apps, translation tools, and writing tutors. The right response is honest framing, not pretending the tool can be locked to only "deserving" users.
