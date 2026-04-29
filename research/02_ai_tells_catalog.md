# 02 — AI Writing Tells: Exhaustive Catalog

A taxonomy of features that current LLM output (Claude 3+/4, GPT-4/5, Gemini 1.5+/2, Llama 3+) overproduces relative to baseline human writing. Each entry includes evidence base, severity (how strong a signal it is to current detectors), frequency (how often it appears in unedited LLM output), and removal difficulty (cosmetic vs. structural).

The list is organized by category. Within each category, items are ordered roughly by signal strength.

---

## A. Lexical tells

### A.1 Excess vocabulary — the "delve" cluster

**The single most rigorously documented AI tell.** Kobak et al. 2024 [Kobak2024, arXiv:2406.07016] analyzed 15 million PubMed biomedical abstracts from 2010-2024 and identified an abrupt frequency increase in dozens of "style words" coinciding with ChatGPT's release. They estimate at least **13.5% of 2024 biomedical abstracts were processed with LLMs** (40%+ in some subfields), and the inference is grounded in the same statistical method used to track the COVID pandemic's effect on scientific vocabulary.

Juzek & Ward 2024 [Juzek2024, arXiv:2412.11385] catalogued **21 focal words whose increased occurrence is attributable to LLMs** and investigated *why* — finding that RLHF (Reinforcement Learning from Human Feedback) is the most likely cause, since these words appear preferentially in RLHF-tuned variants of the same base model.

The canonical excess-vocabulary list, in approximate order of severity:

| Word | Severity | Notes |
|------|----------|-------|
| delve / delving / delved | ★★★★★ | The flagship marker. Frequency increased 25x post-ChatGPT in scientific writing. |
| intricate / intricacies | ★★★★★ | Among the most-cited markers. |
| underscore / underscores / underscoring | ★★★★★ | Verb form is particularly AI-coded; the noun form less so. |
| tapestry | ★★★★ | Often metaphorical: "rich tapestry of...". |
| meticulous / meticulously | ★★★★ | High-frequency post-ChatGPT. |
| leverage / leveraging | ★★★★ | Pre-existed in business jargon; LLMs amplified across domains. |
| navigate / navigating | ★★★★ | Used metaphorically for non-spatial concepts. |
| realm / realms | ★★★★ | "In the realm of..." is a near-perfect AI signature. |
| pivotal | ★★★★ | Especially "play(s) a pivotal role." |
| paramount | ★★★★ | "X is paramount" / "of paramount importance." |
| nuanced / nuances | ★★★ | High frequency in academic writing. |
| multifaceted | ★★★ | Common in introductions/conclusions. |
| holistic | ★★★ | Especially "holistic approach/understanding." |
| robust | ★★★ | Pre-existing in technical writing; LLMs extended to general use. |
| comprehensive | ★★★ | "Comprehensive understanding/analysis." |
| crucial / crucially | ★★★ | Adverb form in particular. |
| seamless / seamlessly | ★★★ | Especially in product / business contexts. |
| garner / garnered | ★★★ | "Garnered attention/interest." |
| fosters / fostering | ★★★ | "Fosters innovation/collaboration/growth." |
| testament | ★★★ | "A testament to..." |
| myriad | ★★★ | "A myriad of." |
| plethora | ★★★ | "A plethora of." |
| paradigm | ★★ | "Paradigm shift." |
| facet / facets | ★★ | "Various facets of." |
| underscores the importance of | ★★★ | Whole-phrase signature. |
| sheds light on | ★★★ | Whole-phrase signature. |
| dive deep / dive into / deep dive | ★★ | Casual register of "delve." |
| at the forefront | ★★ | Common in tech/business. |

**Why these words?** Juzek & Ward's preferred hypothesis: RLHF reward models, trained largely by underpaid annotators in low-cost-labor countries (notably Nigeria, Kenya, India), inadvertently rewarded these specific lexical choices because the annotators themselves used them — they are common in the English-language training materials of professional writing in those educational systems. So the signature is partly a fingerprint of the *RLHF labor pipeline*, not the model architecture. This has the implication that the signature should weaken as RLHF datasets become more diverse, but for the moment it is the dominant single-feature signal.

**Removal difficulty:** Cosmetic. A simple word-substitution pass with a curated synonym list defeats this completely. **This should be the first step of any HumanizeMCP pipeline.**

### A.2 Discourse markers and connectives

LLM output overproduces formal logical connectives:

- **moreover / furthermore** — by far the most over-used. Human academic prose uses these too, but at much lower density than LLM output.
- **however / nevertheless / nonetheless** — particularly sentence-initial.
- **additionally** — sentence-initial form especially.
- **consequently / therefore / thus** — sentence-initial.
- **in conclusion / to conclude / to summarize / in summary** — overwhelmingly AI-coded as section-closing phrases.
- **it is important to note (that) / it should be noted (that) / it is worth noting** — the most-mocked AI hedging phrase.
- **that said / having said that** — frequent transitional.
- **on the other hand** — overproduced.
- **ultimately** — overused as a paragraph-closer.

Severity: ★★★★. Together they create the "essay-template" feel that human readers recognize even before detectors do.

**Removal difficulty:** Mostly cosmetic, but harder than vocabulary swaps because removal requires sentence restructuring. Replace "Moreover, X" with simply "X" or "And X" or merge into the prior sentence.

### A.3 Hedging vocabulary

LLMs hedge constantly. The hedge cluster:

- **may / might / could / can be** — modal pile-up.
- **arguably / potentially / possibly / perhaps**.
- **tends to / often / sometimes / generally / typically** — generalizing qualifiers.
- **it appears / it seems / it would seem**.
- **various / numerous / a variety of / several**.
- **a wide range of / a diverse set of**.

Severity: ★★★. Individually mild; cumulatively distinctive.

### A.4 Emotional / evaluative adjective clusters

LLMs reach for the same emotional descriptors:

- **fascinating / intriguing / compelling / captivating / remarkable**.
- **groundbreaking / revolutionary / unprecedented / transformative**.
- **profound / striking**.
- **vital / essential / critical** — the "importance" cluster, often paired with "is X to."

Severity: ★★★. Especially in introductions, conclusions, and topic sentences.

### A.5 First-person retreat

LLMs often substitute "one" for "I" or "you," producing prose with no clear first-person voice. When they do use first person, it is often "we" (royal/editorial). Genuine human casual writing is much more "I"-heavy.

Severity: ★★. Easy to fix by re-personification.

---

## B. Punctuation tells

### B.1 The em dash

**The most famous AI tell, and a real one.** LLM output overproduces the em dash (—) by an order of magnitude relative to most human writing. This is not a myth — it has been reproducibly measured and is now widely known among writers (to the point that some humans now avoid em dashes specifically to signal "I am not AI"). Even one peer-reviewed paper (Kilictas & Alpay 2025 [Kilictas2025, arXiv:2506.18129]) was specifically dedicated to suppressing em dashes in LLM output.

Severity: ★★★★★. A single document with high em-dash density is a near-certain AI tell to a human reader, even when detectors miss it.

**Removal difficulty:** Trivial. Replace with comma, semicolon, period, or parentheses.

### B.2 The Oxford comma in lists of three

LLMs are aggressive Oxford-comma users. Many human writers (especially British, journalistic, or stylistically casual writers) skip the Oxford comma. Consistent Oxford-comma use is a mild signal. Severity: ★.

### B.3 Numbered/bulleted lists

LLM output reaches for bulleted and numbered lists at much higher rates than human prose, especially in instructional or expository contexts. A document where every section enumerates 3-5 bullet points is a structural fingerprint. Severity: ★★★.

### B.4 Curly / smart quotes

LLM output produces typographically correct curly quotes ("X" 'X'); typed-by-human prose often uses straight quotes ("X" 'X') because the writer is using a code editor or chat box that doesn't auto-substitute. This is a *weak inverse* signal — straight quotes weakly suggest human, curly quotes weakly suggest AI or autocorrect. Severity: ★.

### B.5 Hyphenation consistency

LLMs hyphenate compound modifiers consistently ("AI-generated text"). Human writers vary, especially across drafts. Severity: ★.

### B.6 Sentence-initial conjunctions

LLMs avoid sentence-initial "And" and "But" because they were trained on prescriptive grammar guides that prohibit them. Human casual and creative prose uses sentence-initial conjunctions liberally. Adding them is a strong human-coded signal. Severity: ★★★ (in the favorable direction for humanization).

---

## C. Structural / rhythmic tells

### C.1 Burstiness deficit (sentence-length uniformity)

Human prose alternates between very short and very long sentences ("burstiness"). LLM output produces sentences clustered around a mean length, with low variance.

This is the single most-used statistical feature in commercial detectors (it is half of GPTZero's published methodology). Severity: ★★★★★.

**Removal difficulty:** Structural. Requires sentence merging and splitting, not just substitution.

### C.2 Parallel-structure overuse

LLMs love the rule of three. "X, Y, and Z" lists where X, Y, Z share the same grammatical structure. Examples:

- "It empowers users, supports developers, and accelerates innovation."
- "By analyzing the data, identifying patterns, and drawing conclusions."

Human prose uses parallel structure too, but at lower density and with more variation. Severity: ★★★★.

### C.3 The "It's not X, it's Y" / "Not just X but Y" construction

A near-perfect AI signature when used more than once or twice in a document:

- "This isn't just a tool; it's a paradigm shift."
- "It's not about replacing humans, it's about augmenting them."
- "More than a feature — it's a philosophy."

Severity: ★★★★.

### C.4 The "X is a Y of Z" copular template

LLM output frequently uses the pattern *[abstract noun] is a [metaphor noun] of [domain noun]*:

- "Privacy is a cornerstone of digital trust."
- "Innovation is the engine of progress."
- "Education is the foundation of opportunity."

Severity: ★★★.

### C.5 Paragraph length uniformity

LLM-generated essays tend toward paragraphs of 3-5 sentences each, very evenly. Human essays have wild paragraph-length variance. Severity: ★★★.

### C.6 Topic-sentence-then-elaboration structure

LLM paragraphs almost always begin with a topic sentence and then elaborate. Human prose often buries the lead, opens with a question, opens with a quote, opens with an anecdote, or never has an explicit topic sentence at all. Severity: ★★★.

### C.7 The five-paragraph essay shape

When asked for medium-length expository writing, LLMs produce: introduction with thesis → 2-3 body paragraphs → conclusion. This is the default essay structure of US K-12 instruction. Severity: ★★★.

### C.8 Conclusion that restates the introduction

LLM conclusions almost always recapitulate the introduction. Human conclusions often raise new implications, end with a question, end with a personal observation, or trail off. Severity: ★★★.

---

## D. Register tells

### D.1 Uniform formality

LLM output maintains a single register throughout. Human writing in the wild — especially in any genre except formal academic — fluctuates between casual and formal, professional and personal, dry and emotional. The lack of register fluctuation is itself a signal. Severity: ★★★★.

### D.2 Absence of profanity, slang, and idiom

LLM output is sanitized by RLHF. Mild profanity, generational slang, and current idioms are essentially absent unless explicitly prompted. Severity: ★★★ as a presence-absence signal.

### D.3 Excessive politeness markers

"I'd be happy to," "great question," "certainly," "of course." Severity: ★★★ in conversational AI; less applicable to long-form content.

### D.4 Refusal-style hedges

Even outside refusal contexts, LLM output retains traces of refusal-grammar: "It's important to consider both sides," "while X has merits, it's also worth noting that Y." Severity: ★★.

### D.5 Optimism/positivity bias

LLM output skews positive in sentiment. Sandler et al. 2024 [Sandler2024, arXiv:2401.16587] used LIWC analysis to show ChatGPT-generated dialogues exhibit elevated "social processes," "analytical style," "cognition," "attentional focus," and "positive emotional tone" relative to matched human dialogues. Severity: ★★★ (especially in opinion or essay genres).

---

## E. Content / pragmatic tells

### E.1 No personal anecdotes

Human writing — even in formal contexts — frequently grounds claims in specific experiences ("when I was working on X"). LLM writing rarely does. Severity: ★★★★.

### E.2 No specific named details

Humans name people, places, products, dates, and numbers. LLM output gravitates to generic placeholders ("a recent study," "many experts," "in some cases"). Severity: ★★★★.

### E.3 No typos, no autocorrect-flavored errors

Pristine spelling and grammar throughout. Real human writing in any informal register has at least occasional typos and autocorrect-style errors ("of course" / "ofcourse," "could of" / "could have," missing apostrophes). Severity: ★★★★ (presence of errors is a strong human signal).

### E.4 Hallucinated specificity

Where LLMs do produce specific details, they are sometimes wrong — fabricated citations, invented quotations, plausible-but-fake statistics. This is technically a content quality issue but operates as a tell because human writers usually *don't* invent details out of whole cloth.

### E.5 No cultural / temporal grounding

LLMs avoid current-events references, recent slang, and culturally specific allusions, partly because of training-data cutoff and partly because of safety training. Human writers in any era saturate their prose with the moment they are writing in. Severity: ★★★.

### E.6 No genuine disagreement, ambiguity, or unresolved tension

LLM essays resolve. They reach a conclusion, often a synthesizing both-sides one. Human writers leave threads unresolved, contradict themselves between paragraphs, change their minds mid-essay. Severity: ★★★.

### E.7 The "Conclusion: a balanced approach is needed" closing

A specific high-frequency LLM essay-conclusion pattern: "Ultimately, [topic] requires a balanced approach that considers both [X] and [Y]. As we navigate the complexities of [topic], it is essential to..." Severity: ★★★★.

---

## F. Conversational AI tells (specific to chat output that gets pasted)

### F.1 The opening compliment

"Great question!" / "What an interesting topic!" / "I'd be happy to help with that!" — these survive into pasted output and are an instant tell.

### F.2 Self-reference

"As an AI language model, I..." "I should mention that I..." "While I don't have personal experiences, I can..."

### F.3 The closing offer

"Let me know if you'd like me to elaborate on any of these points!" / "Feel free to ask if you have follow-up questions!"

### F.4 Numbered preamble

"Here are X reasons..." / "There are several factors to consider..." followed by a numbered list.

Severity for all of F: ★★★★★ when present. Easy to detect, easy to remove.

---

## G. Modern-LLM-specific tells

LLMs released in 2024-2026 (Claude 3.5/4, GPT-4o/5, Gemini 2, Llama 3.3+) have evolved past some classical tells but introduced new ones:

- **Hedge-stacking:** modern instruction-tuned LLMs hedge more, not less ("It depends," "While there's no single answer," "Different contexts may call for different approaches").
- **Structured Markdown bleed:** modern LLMs produce headers, bold text, and bulleted summaries even in plain-prose contexts. When this Markdown survives a copy-paste, it is a tell.
- **Bullet-summary-after-prose:** the pattern of writing a paragraph and then summarizing it in bullet form below.
- **The "Here's a more nuanced take" pattern** when asked to revise: rather than substantively changing, the model reorders and adds hedges.
- **Sycophantic agreement before pushback:** "That's a really thoughtful question. You're right that X. However, I'd gently push back on Y."

---

## H. Stylometric features used by detectors (computational signature)

For implementation reference — these are the features that supervised stylometric detectors (XGBoost-on-StyloMetrix, Ghostbuster's feature-search, Pangram's classifier) actually compute:

- Token-level perplexity under a reference LM and its variance (burstiness).
- Type-token ratio and MTLD (Measure of Textual Lexical Diversity).
- Mean sentence length and sentence-length variance.
- Mean dependency-tree depth.
- Distribution of POS tags.
- Function-word frequency vector (the classical Mosteller-Wallace authorship feature set).
- N-gram repetition rates (especially 4-grams and 5-grams).
- Punctuation-mark distribution (em dashes, semicolons, colons).
- Discourse-marker density.
- Distribution of specific top-1000 tokens (which is where the "delve" cluster gets caught).

A humanization pipeline that targets only the *human-readable* tells (em dashes, "delve," parallel structure) but ignores the *statistical* signature (burstiness, perplexity-curvature) will pass casual reader inspection but fail commercial detectors. Conversely, a pipeline that only targets the statistical signature will defeat detectors but produce text that still feels off to a human reader.

**Both layers must be addressed.**

---

## Summary table

| Category | Top-severity tells | Removal difficulty |
|----------|-------------------|--------------------|
| Lexical | "delve," "tapestry," "leverage," "intricate" | Cosmetic (substitution) |
| Punctuation | em dash overuse | Cosmetic |
| Structural | sentence-length uniformity (burstiness) | Structural (re-segmentation) |
| Structural | parallel-structure overuse, "It's not X, it's Y" | Structural |
| Register | uniform formality | Structural (style mixing) |
| Content | no personal anecdotes, no typos, no specifics | Compositional (hard) |
| Conversational | "Great question!" / "Let me know if..." | Cosmetic |
| Statistical | low perplexity, low burstiness, low MTLD | Structural / iterative |

The HumanizeMCP pipeline should address each of these in order of removal cost: cheap cosmetic substitutions first, structural rewriting second, compositional hard problems (anecdote injection, deliberate typo insertion, register mixing) third, statistical-signature optimization (iterative paraphrasing against detector signal) last and most expensive.
