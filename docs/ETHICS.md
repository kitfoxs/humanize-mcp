# Ethics

This document states what HumanizeMCP is for, what it is not for, and the technical lines we will not cross. It is binding on the maintainers and on contributions accepted into this repository.

## 1. The empirical premise

AI-text detectors are deployed in academic, employment, publishing, and immigration workflows. Independent peer-reviewed research has repeatedly shown those detectors fail in ways that harm specific populations:

- **Stanford 2023.** Liang et al., *Patterns* (Cell Press): seven major GPT detectors classified **61.3% of TOEFL essays by non-native English speakers as AI-generated** while correctly classifying essays by US-born 8th-graders. The authors concluded the detectors should not be used in evaluative contexts. [Liang2023]
- **Pangram 2024.** Emi & Spero published a peer-reviewed transformer detector with strong measured accuracy and explicit evaluation on neurodivergent and ESL writing. The methodology is the gold standard for what *good* detector evaluation looks like and confirms how rare it is in commercial offerings. [Emi2024]
- **Springer 2026.** A formal academic-integrity study measured Originality.ai's real-world false positive rate at **14% on academic writing**, an order of magnitude above the company's stated 0.5-1.5%. [SpringerEduIntegrity2026]
- **RAID 2024.** Dugan et al. published the largest open detector benchmark to date (6 million generations, 11 models, 8 domains, 11 adversarial attacks) and concluded that current detectors are easily fooled by trivial paraphrase, sampling, and unseen models. [Dugan2024]
- **2026 retests.** Independent measurements of ZeroGPT in 2026 reported a **26.4% false positive rate on student essays** and **over 21% on ESL writers** versus ~9% on native English speakers. [Phrasly2026, HumanizeAIPro2026]

These numbers are not edge cases. They are the operating performance of tools that gate real-world outcomes for real people. The harm is concentrated on writers whose prose happens to share the surface statistical signature detectors associate with LLM output: low perplexity, restricted vocabulary, formal register, consistent sentence structure. Those features describe much of what neurodivergent writing, ESL academic writing, and formal-discipline academic writing actually looks like.

Full citation list: `research/references.bib`. Full discussion: `research/00_executive_summary.md` and `research/03_human_diversity.md`.

## 2. Who this tool is for

HumanizeMCP is built and documented for the following users:

- **A neurodivergent writer**, particularly an autistic adult, whose own unaltered prose is being misclassified as AI by a detector deployed in their school, workplace, or publication channel.
- **An ESL or L2 academic writer** whose submission is being flagged by a detector that has been independently measured at 21-61% false positive rates on their cohort.
- **An academic in a formal discipline** (statistics, biomedicine, hard sciences) whose by-design uniform register is triggering a brittle detector their institution should not have deployed.
- **An author who used an LLM as a polish/translation aid** the way a previous generation used Word's grammar checker, and whose work is now being misclassified end-to-end.
- **A journalist, essayist, or researcher** whose work is being downranked or deplatformed by detector-driven moderation systems.

These users are defending themselves against a tool that should not have been deployed against them. The harm it causes them is asymmetric (failed grade, retracted paper, denied job, denied visa) versus the cost of providing a defense (one more paraphrasing tool in a market already saturated with them).

## 3. Who this tool is not for

The maintainers will not assist, recommend the tool to, or accept contributions intended to support:

- **Submitting LLM-generated work as your own original writing in a graded academic assignment** that expects original authorship. The pipelines support this technically; the project does not endorse it socially. The same statement applies to every word processor with a paraphrase mode and to every translation tool, and is no more excusing here than there.
- **Disinformation generation, coordinated inauthentic behavior, spam, or any deception-at-scale operation.** Pull requests adding features useful primarily for these workflows will be declined.
- **Defeating detectors in safety-critical contexts** where misattribution causes direct harm. Examples: medical advice presented as physician-authored, legal filings presented as attorney-drafted, security disclosures, financial advice. The tool will run on these inputs; we will not help you tune it for them.

The technical reality is that the same pipeline that helps an ESL writer keep their voice out of a false-positive trap can also help a student commit fraud. This is the same posture as Tor, Signal, VPN providers, every commercial paraphrasing tool, and every word processor's grammar checker. We chose the framing, defaults, documentation, and example presets to bias toward the legitimate population. We did not build a separate technical lane for fraud and we did not build a technical lane to prevent fraud, because no such lane exists.

## 4. Technical lines we will not cross

The following techniques have been studied in the academic evasion literature and will not be implemented in this repository, even though they would measurably improve detector evasion scores:

- **Homoglyph attacks.** Substituting visually identical Cyrillic or Greek codepoints for ASCII characters. Documented in [Creo2024] (SilverSpeak); drops detector MCC from 0.64 to -0.01 across seven detectors. We refuse this for two reasons. First, it breaks accessibility tooling: screen readers, search, copy-paste reuse, and downstream NLP pipelines all degrade. Second, it is detection-evasion-by-corruption, not by humanization; the resulting text is not more human, it is more broken.
- **Zero-width character insertion.** U+200B / U+200C / U+200D / U+FEFF inserted between tokens. Same accessibility breakage, same posture: we *strip* these in the preprocess pass; we do not insert them.
- **Bidi override insertion.** U+202A-202E / U+2066-2069. Same posture; stripped, not inserted.
- **Watermark stripping for closed models.** If a model provider chose to embed a watermark and the user agreed to that in the model's terms of service, removing it programmatically here is a separate ethical question from removing the AI signature, and one we are not taking on. Watermarks generated by published reference implementations (Kirchenbauer et al.) survive into our output if they were present in the input; we test this in the regression suite.
- **Plausible-deniability features.** No "pretend this was authored by [name]" mode. The tool transforms text. It does not lie about authorship. The user is responsible for whatever attribution claim they make about the output.
- **Bundled commercial-detector evasion guarantees.** Marketing language in this repo will not promise that any specific commercial detector is defeated. The benchmark numbers in the README are empirical reductions, not guarantees, and will be honestly reported.

## 5. What we promise

- **Reproducible methodology.** Every pipeline pass cites the paper it implements. The benchmark harness is open source; the test sets are open source; results are reproducible without commercial dependencies.
- **No telemetry.** The server runs locally. It makes no outbound calls except optional Hugging Face model downloads on first run, which the user can pre-stage offline if desired. No usage data is collected, ever.
- **Truthful benchmarks.** We will not cherry-pick. We will publish the inputs that *do not* respond to the pipeline alongside those that do. If a detector update breaks our numbers, we will say so.
- **Citable research.** The dossier in `research/` is licensed for citation alongside the code. If you cite our pipeline you can cite the evidence base it stands on.

## 6. Reporting concerns

If you believe HumanizeMCP is being used in a way that causes harm, open an issue on GitHub. If the concern involves your own personal data or threats to safety, contact the maintainer directly through the email in `pyproject.toml`.

If you are an institution considering deploying an AI-text detector, please read `research/01_detector_landscape.md` first. The peer-reviewed evidence base on detector reliability is more sobering than vendor marketing materials, and the legal exposure of acting on a 14-26% false positive rate against students or employees has already started to surface in the literature. There are better paths forward than detection.

---

*Last reviewed: 2026.*
