# 05 — Style Corpora

A catalog of text corpora usable as (a) training/few-shot reference material for HumanizeMCP's voice-preservation and style-transfer features, (b) calibration material for detectors and self-evaluation harnesses, and (c) ground-truth human writing for benchmarking.

Each entry includes: source, content type, scale, license, recommended use, access path. Licensing notes are crucial — much of the natural-prose web is not actually re-distributable.

---

## 1. AI-vs-human reference corpora

These corpora pair human writing with LLM-generated equivalents and are the standard calibration material for both detectors and humanizers.

### 1.1 HC3 (Human ChatGPT Comparison Corpus)

- **Citation:** Guo, B., et al. (2023). arXiv:2301.07597.
- **Scale:** ~24,000 question-answer pairs, each with a human expert response and a ChatGPT-3.5 response. Domains: open-domain QA, finance, medicine, law, psychology.
- **License:** Released for research; check repo for specifics.
- **Access:** HuggingFace `Hello-SimpleAI/HC3` and `Hello-SimpleAI/HC3-Chinese`.
- **Use:** Reference for detector training and evaluation; gold-standard human-vs-AI comparison.
- **Notes:** The most-cited single dataset in the AI-detection literature. Some 2023 vintage; LLM-side is GPT-3.5 era.

### 1.2 RAID (Robust AI-text Detection benchmark)

- **Citation:** Dugan, L., et al. (2024). arXiv:2405.07940.
- **Scale:** **6 million generations** across 11 LLMs, 8 domains, 11 adversarial attacks, 4 decoding strategies.
- **License:** Research; check repo.
- **Access:** https://github.com/liamdugan/raid
- **Use:** **Primary HumanizeMCP self-evaluation benchmark.** Gives both the AI text to humanize *and* the standard adversarial-attack reference points to compare against.

### 1.3 MAGE / M4 / MultiSocial / CEAID

Multilingual AI-text detection corpora:
- **M4** (Wang et al.): 6 LLMs, 10+ languages.
- **MultiSocial** (Macko et al. 2024, arXiv:2406.12549): 472,097 social media texts in 22 languages from 5 platforms. **Key for short-text humanization** (tweets, Mastodon, Bluesky, Reddit comments).
- **CEAID** (Macko et al. 2025, arXiv:2509.26051): Central European languages.
- **GenAI Content Detection Task 2** (Chowdhury et al. 2024, arXiv:2412.18274): English + Arabic academic essays.

### 1.4 PASTED

- **Citation:** Li, Y., et al. (2024). arXiv:2405.12689.
- **Scope:** Paraphrased text span detection. Sentence-level annotations of which spans of a document are AI-paraphrased vs. original.
- **Access:** https://github.com/Linzwcs/PASTED
- **Use:** Useful for training a model that detects which sentences in a document need humanization (vs. which already read as human).

### 1.5 CoAuthor

- **Citation:** Lee et al. 2022 (UIST).
- **Scope:** Real human-AI collaborative writing sessions; multi-turn interactions in which users selectively accept and edit AI suggestions. ~1,400 sessions.
- **Use:** Realistic hybrid-text training material; reflects actual user behavior rather than synthetic mixing.

### 1.6 PADBen

- **Citation:** Zha et al. 2025.
- **Scope:** Five-type text taxonomy from original to deeply paraphrased; benchmarks for both authorship-obfuscation and plagiarism-evasion attack scenarios.
- **Access:** https://github.com/JonathanZha47/PadBen-Paraphrase-Attack-Benchmark

---

## 2. Human writing reference corpora (by genre)

### 2.1 Casual / conversational

#### Reddit (Pushshift)

- **Source:** Pushshift Reddit dump.
- **Scale:** Billions of comments and submissions across all of Reddit.
- **License:** Reddit's terms of service and the now-changed Pushshift situation make this complicated. The 2023 Reddit API changes and the Pushshift takedown by Reddit Inc. mean that fresh Pushshift data is no longer freely accessible. Older snapshots (pre-2023) are still in circulation through academic mirrors but legal status is contested.
- **Recommended access:** HuggingFace `webis/reddit` and various academic snapshots; the Anthropic-curated `Anthropic/hh-rlhf` includes Reddit-derived material in a more clearly licensed form.
- **Use:** The premier source of casual conversational English. Subreddit selection lets you target register: r/AskHistorians for formal-but-personal, r/AmItheAsshole for narrative-personal, r/explainlikeimfive for casual-pedagogical.

#### Reddit TIFU

- **Source:** Subreddit r/TIFU posts.
- **Scale:** ~120,000 posts.
- **License:** Research-permissible.
- **Access:** HuggingFace `reddit_tifu`.
- **Use:** Narrative-confessional first-person writing. Useful style-transfer target for "personal blog" voice.

#### Blended Skill Talk

- **Citation:** Smith et al. 2020 (Facebook).
- **Scale:** Crowd-sourced multi-turn dialogues; ~7,000 conversations.
- **Access:** HuggingFace `blended_skill_talk`.
- **Use:** Polished but conversational dialogue. Useful for chatbot-output humanization targets.

#### Twitter / X / Bluesky / Mastodon corpora

- **Sources:** Various academic Twitter dumps (mostly pre-2023 due to API closure); newer Bluesky and Mastodon corpora are emerging via the open AT Protocol and ActivityPub firehoses.
- **License:** Variable. Twitter dumps are now legally fraught; Bluesky data is more clearly available under the public protocol.
- **Use:** Short-form, very informal English with slang, abbreviations, hashtags.

### 2.2 Personal / blog

#### Blog Authorship Corpus

- **Citation:** Schler, Koppel, Argamon, Pennebaker 2006.
- **Scale:** 681,288 blog posts from 19,320 distinct bloggers (Blogger.com, 2004 vintage).
- **License:** Released for research.
- **Access:** HuggingFace `blog_authorship_corpus`.
- **Use:** Personal-narrative writing with rich author metadata (gender, age, occupation). Useful for diverse-voice training and demographic-specific style transfer.
- **Caveat:** 20 years old. Reflects 2004 web register, not modern blog/Substack/Medium register.

#### Substack / Medium scrapes

- **Source:** Public Substack/Medium content via web scraping.
- **License:** Murky. Individual posts are copyrighted by authors.
- **Use:** Modern long-form personal essay style. Recommend manual curation rather than redistribution.

### 2.3 Literary

#### Project Gutenberg

- **Source:** https://www.gutenberg.org
- **Scale:** ~70,000 books, primarily pre-1928 US public domain.
- **License:** Public domain in the US (most works); each work has individual licensing notes; Gutenberg's text wrappers are public domain.
- **Access:** HuggingFace `manu/project_gutenberg`, `eminorhan/gutenberg_en`, `sedthh/gutenberg_english`; bulk download from Gutenberg directly; `nltk.corpus.gutenberg` for a small curated subset.
- **Use:** Literary prose style. Excellent for "novelistic" or "essayistic" preserve-voice targets. Strong stylistic diversity (Austen, Dickens, James, Conrad, Twain).
- **Caveat:** Pre-modern register. May not fit contemporary writing tasks.

#### BookCorpus / BookCorpus2 / Books3

- **Source:** Various; Books3 was the most controversial (taken down in 2023 due to copyright concerns; included pirated commercial books).
- **License:** **Do not use.** Books3 in particular is implicated in active copyright litigation. The HumanizeMCP project should not redistribute or rely on these corpora.

### 2.4 Professional email

#### Enron Email Corpus

- **Source:** Cohen, W., Carnegie Mellon University.
- **Scale:** ~500,000 emails from ~150 senior Enron employees, made public during the FERC investigation.
- **License:** Public; widely used in academic NLP research.
- **Access:** http://www.cs.cmu.edu/~enron/ ; also on HuggingFace as `snoop2head/enron_aeslc_emails` and similar mirrors.
- **Use:** Professional email writing style — short, transactional, hierarchical. The premier corpus for "business email" style targets.
- **Caveat:** 2001 vintage; predates modern email conventions, mobile email, emoji.

#### AESLC (Annotated Enron Subject Line Corpus)

- A subset of Enron with subject-line annotations.
- **Use:** Summarization tasks; useful for short professional-prose targets.

### 2.5 Academic

#### arXiv / S2ORC / OpenAlex

- **Sources:** arXiv full-text API; Allen Institute's S2ORC; OpenAlex.
- **License:** arXiv content licensed individually by paper (most CC-BY, some all-rights-reserved); S2ORC and OpenAlex are research aggregations.
- **Use:** Academic prose by discipline. Useful for "academic preset" style targets.

#### PubMed Central Open Access Subset

- **Source:** NCBI.
- **License:** PMC-OA includes only papers with open licenses (CC-BY, CC-BY-NC, etc.); confirm per-article.
- **Use:** Biomedical academic prose. Notably this is the corpus where the "delve" lexical shift was measured (Kobak et al. 2024).

#### EFCAMDAT (EF Cambridge Open Language Database)

- **Source:** EF Education First / University of Cambridge.
- **Scale:** ~1.2 million essays from ~175,000 ESL learners across 16 proficiency levels.
- **License:** Restricted academic use (registration required).
- **Use:** **Critical for ESL preset.** Real ESL student writing across proficiency levels. The ground truth for "what does ESL writing actually look like at scale."

#### ICLE (International Corpus of Learner English)

- **Source:** Université Catholique de Louvain.
- **Scale:** ~6,000 essays from ESL university students across 16 L1 backgrounds.
- **License:** Commercial; academic licenses available.
- **Use:** Smaller but better-controlled ESL corpus. The de facto standard for L2 English research.

#### Cambridge Learner Corpus

- **Source:** Cambridge University Press.
- **License:** Commercial; restricted access.
- **Use:** Largest ESL corpus available; expensive.

### 2.6 News and journalism

#### CNN/DailyMail

- **Source:** Hermann et al. 2015.
- **Scale:** ~300,000 news articles with summaries.
- **License:** Research use.
- **Use:** Journalistic prose targets.

#### XSum

- **Source:** Narayan et al. 2018.
- **Use:** BBC News articles + extreme summaries.

### 2.7 Web scrape (broad mixed prose)

#### OpenWebText / OpenWebText2

- **Source:** Reproduction of OpenAI's WebText, scraped from URLs in Reddit posts with karma > 3.
- **License:** Unclear and risky. Includes content from arbitrary copyrighted websites.
- **Use:** As a *background* / negative-control corpus for stylometric analysis, not as redistributable training data.

#### C4 (Colossal Clean Crawled Corpus)

- **Source:** Raffel et al. 2020 (T5 paper). Filtered Common Crawl.
- **License:** Use at your own risk; copyright burden on user.
- **Access:** HuggingFace `c4`, `allenai/c4`.
- **Use:** As above — analytical reference, not redistributable training material.

#### RedPajama / SlimPajama / FineWeb / FineWeb-Edu

- **Sources:** Together Computer; HuggingFace.
- **License:** Same caveats as C4. FineWeb-Edu is filtered for educational content and is the most defensible of the lot for academic use.
- **Access:** HuggingFace `HuggingFaceFW/fineweb-edu`.
- **Use:** General prose reference; FineWeb-Edu is recommended over alternatives.

### 2.8 Public domain heavy

#### Wikipedia

- **License:** CC-BY-SA. The cleanest large corpus for redistributable use.
- **Access:** `huggingface/wikipedia` or any mirror.
- **Use:** Encyclopedic prose target. Note this is *itself* a register that detectors flag, because it is uniform, formal, low-burstiness — Wikipedia prose has the same statistical signature as AI prose in many respects. (Stylometry recognizes Wikipedia + GPT-4 distinguishable at .98 accuracy [Przystalski2025].)

#### Project Gutenberg

(Listed above.) The other premier public-domain corpus.

#### US Government documents

- **License:** Public domain (US federal works).
- **Use:** Bureaucratic / formal-administrative prose target.

---

## 3. Style-feature corpora and tools

### 3.1 LIWC (Linguistic Inquiry and Word Count)

- **Source:** Pennebaker, Boyd et al.
- **License:** Commercial (LIWC-22).
- **Use:** 90+ psycholinguistic categories computed from word lists. Used in Sandler et al. 2024 to characterize ChatGPT vs. human dialogues. Useful as a feature signature for stylometric scoring.
- **Open alternative:** `empath` (Fast et al. 2016) and `liwc-text` Python ports of older LIWC dictionaries.

### 3.2 StyloMetrix

- **Citation:** Okulska & Stetsenko 2023.
- **License:** Open source (Apache 2.0).
- **Access:** https://github.com/ZILiAT-NASK/StyloMetrix
- **Use:** **Recommended primary stylometric library.** Computes 250+ stylometric features (lexical, syntactic, grammatical, punctuation) for English and several other languages.

### 3.3 Coh-Metrix

- **Source:** McNamara & Graesser et al.
- **License:** Free for academic use; the original tool is Java-based and dated.
- **Use:** Cohesion and coherence metrics. The Python port `lftk` (Lee et al. 2023) is more practical.

### 3.4 textstat (Python)

- **Access:** `pip install textstat`.
- **Use:** Readability metrics (Flesch-Kincaid, SMOG, ARI, Coleman-Liau, etc.). Cheap and fast; useful as a basic feature pass.

### 3.5 Sentence-Transformers / SBERT

- **Use:** Semantic similarity for paraphrase quality verification (does the rewrite preserve meaning?).
- **Access:** `sentence-transformers` PyPI package, model `all-MiniLM-L6-v2` for speed or `all-mpnet-base-v2` for quality.

---

## 4. Recommended corpus selection for HumanizeMCP

For the initial HumanizeMCP build, the following minimal corpus stack is sufficient:

### Calibration / self-evaluation
- **HC3** (HuggingFace `Hello-SimpleAI/HC3`) — gold-standard AI-vs-human pairs.
- **RAID subset** (HuggingFace `liamdugan/raid` if available, or download via repo) — multi-attack benchmark.
- **MultiSocial** for short-form / social-media calibration.

### Style-transfer presets
- **Preset: casual** — `reddit_tifu` + selected r/AskHistorians and r/explainlikeimfive subsets.
- **Preset: blog** — `blog_authorship_corpus` (with awareness of vintage).
- **Preset: literary** — Project Gutenberg, curated subset (Austen, Twain, Wharton, Hemingway as canonical varied-voice references).
- **Preset: academic** — selected arXiv abstracts from cs.CL by humans pre-2022 (cleaner human baseline) + PMC-OA biomedical abstracts pre-2022.
- **Preset: professional_email** — Enron + AESLC.
- **Preset: ESL** — ICLE if licensed; else manually curated open ESL essay collections.

### Stylometric scoring
- **StyloMetrix** as primary feature library.
- **textstat** for readability features.
- **sentence-transformers** for semantic-preservation checking.

### What to *not* ship
- Books3 or any pirated-book corpus.
- Twitter dumps acquired post-2023.
- Pushshift Reddit dumps newer than the Reddit API closure date.
- C4/RedPajama as bundled redistributable data (link to them, don't host).

### Licensing summary
- **Safe to redistribute:** Wikipedia, Project Gutenberg (US), US government documents, HC3, RAID (per repo terms), MultiSocial (per repo terms), Enron, blog_authorship_corpus.
- **Use but don't redistribute:** OpenWebText, C4, Common Crawl-derivatives.
- **Academic-licensed:** ICLE, EFCAMDAT, Cambridge Learner Corpus, LIWC.
- **Do not use:** Books3, post-2023 Twitter, post-API-change Pushshift, any pirated book corpus.

---

## 5. Suggested corpus storage layout

For implementation in HumanizeMCP, recommended directory layout under `humanize-mcp/styles/`:

```
styles/
├── corpora/                # downloaded corpora, not committed
│   ├── hc3/
│   ├── raid/
│   ├── enron/
│   ├── gutenberg/
│   └── reddit_tifu/
├── presets/                # curated few-shot examples per preset
│   ├── casual.jsonl
│   ├── blog.jsonl
│   ├── literary.jsonl
│   ├── academic.jsonl
│   ├── email.jsonl
│   ├── esl.jsonl
│   ├── neurodivergent.jsonl
│   └── preserve.jsonl
├── stylometric_targets/    # per-preset stylometric feature vectors
│   └── *.json
└── README.md               # licensing, sources, refresh instructions
```

The `presets/*.jsonl` files should each contain 20-50 curated short passages exemplifying the preset, drawn from the corpora above and licensed for redistribution. Each preset should be small enough to ship in the repo (under 1MB).
