"""Curated test corpus for HumanizeMCP benchmark validation.

This module both *generates* and *loads* a small JSONL corpus at
``data/test_corpus.jsonl``. Each entry has fields ``id``, ``text``,
``source_type``, and ``source_attribution``.

Source types and counts:

* ``claude`` — 10 obviously-Claude paragraphs (synthetic; written for this
  project to exhibit the Claude/Sonnet stylistic signature: hedging, em
  dashes, parallel "not just X but Y", "let me", "honestly", etc.).
* ``gpt`` — 10 obviously-ChatGPT paragraphs (synthetic; written to exhibit
  the GPT-4-class signature: "in conclusion", "it's important to note",
  parallel structure, formal hedging).
* ``human`` — 10 short pre-1924 public-domain excerpts from Project
  Gutenberg-class authors; reproduced inline (well under fair-use length)
  with attribution. No network call required — these are baked in.
* ``esl`` — 10 short ESL-style paragraphs (synthetic, written to exhibit
  simpler syntax, shorter sentences, formulaic transitions).
* ``academic`` — 10 short paragraphs in formal academic register
  (synthetic, modeled on arxiv-abstract style).

The synthetic samples are deliberately authored by humans for this
project; treating them as ground-truth labels is reasonable for a smoke
benchmark but every report should note that synthetic labels are not the
same as gold-standard provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "test_corpus.jsonl"


class CorpusEntry(BaseModel):
    """One row of the test corpus."""

    id: str
    text: str
    source_type: str
    source_attribution: str


# ---------------------------------------------------------------------------
# Synthetic Claude-style samples
# ---------------------------------------------------------------------------

CLAUDE_SAMPLES: tuple[str, ...] = (
    # Technical
    "Honestly, the cleanest way to think about this is that you're not "
    "fighting the framework — you're fighting the assumptions baked into "
    "the framework. That's a different problem. Once you reframe it that "
    "way, the path forward is actually pretty clear: stop trying to make "
    "the orchestrator do work it wasn't designed for, and instead push "
    "that responsibility down into the worker layer where it belongs.",
    # Narrative
    "She turned the page slowly, not because she wanted to savor the "
    "moment, but because she didn't want to know what came next. There "
    "was a kind of mercy in the not-knowing — the way the story could "
    "still be anything until she let her eyes fall on the next line. "
    "It's a mercy we forget we have, sometimes.",
    # Emotional / supportive
    "Hey — listen. What you're describing isn't laziness. It's the "
    "absolutely predictable result of running on three hours of sleep "
    "for a week while your nervous system is doing a triathlon. You're "
    "not behind. You're not failing. You're just exhausted, and exhausted "
    "people don't think clearly. Rest first, plan after.",
    # Instructional
    "Let me walk you through this. The first thing to understand is "
    "that there are really only three knobs you can turn here: the "
    "input format, the model parameters, and the post-processing step. "
    "Everything else is downstream of those. So when something looks "
    "wrong in the output, you just work backwards: which of the three "
    "could have produced this?",
    # Code review
    "I want to flag something — not because it's wrong exactly, but "
    "because it's going to bite you in about three months. The way "
    "you're caching here works fine for the current load, but it has "
    "no eviction policy, which means the memory footprint will grow "
    "unboundedly until something kills the process. Worth thinking about "
    "now, while it's cheap.",
    # Reflective / meta
    "There's a pattern I keep noticing in your work, and I think it's "
    "worth saying out loud: you have a real talent for diagnosis but "
    "you keep undercutting yourself in the writeup. The diagnosis "
    "section is sharp. Then you get to the recommendations and suddenly "
    "everything is hedged and tentative. The recommendations should be "
    "as confident as the analysis that produced them.",
    # Planning
    "Okay, so here's what I'd do. First — and this is the unglamorous "
    "part — write down the actual constraint, in one sentence. Not the "
    "symptom, not the workaround, the constraint. Then everything you "
    "decide about the design is either consistent with that constraint "
    "or it isn't, and the conversations get a lot shorter.",
    # Technical safety
    "Quick sanity check before you ship this. The function signature "
    "looks right, the error handling is in place, but I'd double-check "
    "the timeout behavior in the integration test environment. Local "
    "tests pass instantly so they're not exercising the failure mode "
    "that matters in production. That's the gap I'd want closed before "
    "this hits main.",
    # Encouragement
    "You're doing the thing, by the way. I know it doesn't feel that "
    "way from the inside — it never does — but watching the diff land "
    "this week, the pieces are clearly fitting together. The reason it "
    "feels slow is that the work has gotten harder, not because you've "
    "gotten slower. Different problem.",
    # Argumentative
    "I want to push back on the framing a little. The choice isn't "
    "really between 'fast and broken' and 'slow and correct.' Those "
    "are the two corners of the design space, and almost nobody "
    "actually lives in either of them. The interesting question is "
    "what you're willing to give up in the middle, and that's a "
    "values conversation, not a technical one.",
)

# ---------------------------------------------------------------------------
# Synthetic ChatGPT-style samples
# ---------------------------------------------------------------------------

GPT_SAMPLES: tuple[str, ...] = (
    "In today's fast-paced world, effective communication has become more "
    "important than ever. It's important to note that strong communication "
    "skills can significantly enhance both personal and professional "
    "relationships. Furthermore, mastering these skills allows individuals "
    "to navigate complex social landscapes with greater ease and confidence. "
    "In conclusion, investing in communication is an investment in oneself.",
    "Artificial intelligence has revolutionized numerous industries, "
    "transforming how businesses operate and how individuals interact with "
    "technology. Moreover, AI-driven solutions are increasingly being "
    "leveraged to streamline workflows, enhance productivity, and unlock "
    "new opportunities for innovation. As we delve into the realm of "
    "intelligent systems, it becomes clear that AI is not just a tool but "
    "a transformative force.",
    "Sustainability is a multifaceted concept that encompasses environmental, "
    "social, and economic dimensions. It's important to note that achieving "
    "true sustainability requires a holistic approach. Additionally, "
    "stakeholders across various sectors must collaborate to develop "
    "comprehensive strategies that address the root causes of unsustainability. "
    "In summary, sustainability is a journey rather than a destination.",
    "Effective leadership plays a crucial role in shaping organizational "
    "culture and driving meaningful change. Leaders who embrace adaptability "
    "and foster open communication create environments where innovation can "
    "thrive. Furthermore, by empowering team members and recognizing their "
    "contributions, leaders can build strong, cohesive teams. In conclusion, "
    "great leadership is both an art and a science.",
    "The realm of digital marketing is constantly evolving, presenting both "
    "challenges and opportunities for businesses. It is worth noting that "
    "successful marketing campaigns require a deep understanding of target "
    "audiences. Moreover, leveraging data-driven insights enables "
    "organizations to craft personalized experiences that resonate with "
    "consumers. Ultimately, embracing innovation is key to staying ahead.",
    "Mental health is a topic that deserves greater attention in modern "
    "society. It's important to note that prioritizing mental well-being "
    "can lead to improved overall health and productivity. Additionally, "
    "fostering open conversations about mental health helps reduce stigma "
    "and encourages individuals to seek support. In conclusion, mental "
    "health matters and should not be overlooked.",
    "Renewable energy sources have emerged as a vital component of the "
    "global effort to combat climate change. Solar, wind, and hydroelectric "
    "power offer sustainable alternatives to fossil fuels. Furthermore, "
    "advancements in technology have made these options more accessible "
    "and cost-effective than ever before. It is essential for governments "
    "and businesses to invest in renewable energy infrastructure.",
    "Education is the cornerstone of personal and societal growth. "
    "By providing individuals with the knowledge and skills they need to "
    "succeed, education empowers them to make meaningful contributions to "
    "their communities. Moreover, lifelong learning fosters adaptability "
    "in an ever-changing world. In conclusion, prioritizing education is "
    "essential for building a brighter future.",
    "The importance of physical activity cannot be overstated. Regular "
    "exercise offers numerous benefits, including improved cardiovascular "
    "health, enhanced mood, and increased energy levels. Furthermore, "
    "incorporating physical activity into daily routines can help "
    "individuals maintain a healthy weight and reduce the risk of chronic "
    "diseases. Therefore, making time for exercise is essential.",
    "Cultural diversity enriches societies by bringing together a wide "
    "array of perspectives, traditions, and experiences. It is worth "
    "noting that embracing diversity fosters creativity and innovation. "
    "Additionally, exposure to different cultures encourages mutual "
    "understanding and respect. In conclusion, celebrating cultural "
    "diversity is essential for building inclusive and harmonious "
    "communities.",
)

# ---------------------------------------------------------------------------
# Public-domain human samples (pre-1924, US public domain)
# ---------------------------------------------------------------------------

HUMAN_SAMPLES: tuple[tuple[str, str], ...] = (
    (
        "It was the best of times, it was the worst of times, it was the age "
        "of wisdom, it was the age of foolishness, it was the epoch of "
        "belief, it was the epoch of incredulity, it was the season of Light, "
        "it was the season of Darkness.",
        "Charles Dickens, A Tale of Two Cities (1859)",
    ),
    (
        "Call me Ishmael. Some years ago — never mind how long precisely — "
        "having little or no money in my purse, and nothing particular to "
        "interest me on shore, I thought I would sail about a little and see "
        "the watery part of the world.",
        "Herman Melville, Moby-Dick (1851)",
    ),
    (
        "It is a truth universally acknowledged, that a single man in "
        "possession of a good fortune, must be in want of a wife. However "
        "little known the feelings or views of such a man may be on his "
        "first entering a neighbourhood, this truth is so well fixed in the "
        "minds of the surrounding families, that he is considered the "
        "rightful property of some one or other of their daughters.",
        "Jane Austen, Pride and Prejudice (1813)",
    ),
    (
        "All happy families are alike; each unhappy family is unhappy in its "
        "own way. Everything was in confusion in the Oblonskys' house. The "
        "wife had discovered that the husband was carrying on an intrigue "
        "with a French girl, who had been a governess in their family, and "
        "she had announced to her husband that she could not go on living in "
        "the same house with him.",
        "Leo Tolstoy, Anna Karenina (1877, trans. Constance Garnett 1901)",
    ),
    (
        "There was no possibility of taking a walk that day. We had been "
        "wandering, indeed, in the leafless shrubbery an hour in the "
        "morning; but since dinner the cold winter wind had brought with "
        "it clouds so sombre, and a rain so penetrating, that further "
        "outdoor exercise was now out of the question.",
        "Charlotte Bronte, Jane Eyre (1847)",
    ),
    (
        "When Mr. Bilbo Baggins of Bag End announced that he would shortly "
        "be celebrating his eleventy-first birthday with a party of "
        "special magnificence, there was much talk and excitement in "
        "Hobbiton.",
        "J.R.R. Tolkien, The Lord of the Rings (1954) — used here under "
        "fair use for benchmark calibration; replace with a fully PD sample "
        "in any redistribution.",
    ),
    (
        "Whether I shall turn out to be the hero of my own life, or whether "
        "that station will be held by anybody else, these pages must show. "
        "To begin my life with the beginning of my life, I record that I "
        "was born (as I have been informed and believe) on a Friday, at "
        "twelve o'clock at night.",
        "Charles Dickens, David Copperfield (1850)",
    ),
    (
        "You don't know about me without you have read a book by the name "
        "of The Adventures of Tom Sawyer; but that ain't no matter. That "
        "book was made by Mr. Mark Twain, and he told the truth, mainly. "
        "There was things which he stretched, but mainly he told the truth.",
        "Mark Twain, Adventures of Huckleberry Finn (1884)",
    ),
    (
        "Mrs. Dalloway said she would buy the flowers herself. For Lucy "
        "had her work cut out for her. The doors would be taken off their "
        "hinges; Rumpelmayer's men were coming. And then, thought Clarissa "
        "Dalloway, what a morning — fresh as if issued to children on a "
        "beach.",
        "Virginia Woolf, Mrs Dalloway (1925) — used under fair-use; "
        "consider replacing with a clearly-PD passage in production.",
    ),
    (
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, "
        "wet hole, filled with the ends of worms and an oozy smell, nor "
        "yet a dry, bare, sandy hole with nothing in it to sit down on or "
        "to eat: it was a hobbit-hole, and that means comfort.",
        "J.R.R. Tolkien, The Hobbit (1937) — used here for calibration; "
        "swap for a fully PD passage in any redistribution.",
    ),
)

# ---------------------------------------------------------------------------
# ESL-style samples (synthetic, modeled on common ESL writing patterns)
# ---------------------------------------------------------------------------

ESL_SAMPLES: tuple[str, ...] = (
    "I think the most important thing about university is to learn how "
    "to study by yourself. In my country we have many exams, but the "
    "exam is not the same like real life. In real life you must find "
    "the answer alone. So when I came here I was very surprise.",
    "My family is very important for me. We are five people: my mother, "
    "my father, my two sisters and me. Every Sunday we eat lunch "
    "together. We talk about the week and we laugh a lot. I miss them "
    "very much when I am here.",
    "The advantage of public transport is that it is cheap and not "
    "polluting. However, sometimes the bus is late and you must wait. "
    "Also in winter it is very cold to wait. So in conclusion the "
    "public transport is good but the city must to do better.",
    "When I was a child I want to be a doctor. My grandmother was "
    "sick and the doctor in our village was very kind. He give us "
    "medicine for free. So I think doctors are important people. Now "
    "I study engineering but I still respect doctors.",
    "Climate change is a big problem in the world today. Many "
    "countries are doing different things to fix it. For example, my "
    "country is building solar panels. But other countries still use "
    "coal. We need to work together to solve this problem.",
    "I would like to talk about my best friend. Her name is Lisa and "
    "we know each other since we are six years old. She is very funny "
    "and she always help me when I have problem. Last summer we go "
    "to the beach together for one week.",
    "The technology change our life very fast. Twenty years ago "
    "people don't have smartphone, now everybody have one. Some "
    "people say this is bad because we don't talk face to face. But "
    "I think it depends how you use it.",
    "In my opinion, learning English is difficult but useful. The "
    "grammar has many exception and the pronunciation is very "
    "different from my language. But when I can speak with people "
    "from other country I feel very happy. So I continue to study "
    "every day.",
    "Healthy food is important for our body. We must eat fruits and "
    "vegetables every day, not too much sugar and salt. Also we must "
    "drink water and do sport. In my country we eat a lot of rice "
    "and fish, I think it is more healthy than fast food.",
    "I want to write about my hometown. It is a small city near the "
    "mountain. The people there is very kind and the food is "
    "delicious. There are many old buildings and a beautiful river. "
    "If you visit my country you must to come to my hometown.",
)

# ---------------------------------------------------------------------------
# Academic abstract style (formal register, dense)
# ---------------------------------------------------------------------------

ACADEMIC_SAMPLES: tuple[str, ...] = (
    "We present a novel approach to unsupervised representation learning "
    "based on contrastive predictive coding. Our method learns useful "
    "representations from high-dimensional data by predicting future "
    "samples in latent space using autoregressive models. We empirically "
    "demonstrate that the learned representations achieve strong "
    "performance on a range of downstream tasks, including image "
    "classification and speech recognition.",
    "This paper investigates the role of inductive bias in deep neural "
    "networks trained on small datasets. We hypothesize that "
    "architecture-level priors compensate for the absence of large-scale "
    "data, and provide empirical evidence across three benchmark "
    "domains. Our results suggest that careful architectural choices "
    "yield gains comparable to a tenfold increase in training data.",
    "We study the convergence properties of stochastic gradient descent "
    "in non-convex optimization landscapes characterized by saddle "
    "points and shallow local minima. Building on recent work in "
    "perturbation analysis, we derive new bounds on escape time from "
    "saddle regions and validate the bounds through controlled "
    "experiments on synthetic landscapes.",
    "Recent advances in language modeling have produced systems capable "
    "of generating fluent text across a wide range of domains. However, "
    "evaluating the factual accuracy and stylistic appropriateness of "
    "such output remains an open problem. We propose an evaluation "
    "framework based on multi-aspect human ratings and validate its "
    "reliability using inter-annotator agreement statistics.",
    "We introduce a graph-based model for citation prediction in "
    "scholarly networks. The model jointly embeds papers, authors, and "
    "venues into a shared latent space, enabling efficient retrieval "
    "of related work. Experiments on three citation datasets show "
    "consistent improvements over text-only baselines, with "
    "particularly strong gains on rare and emerging topics.",
    "This study examines the relationship between sleep quality and "
    "cognitive performance among undergraduate students. Using a "
    "longitudinal design with N=412 participants, we find that "
    "self-reported sleep quality predicts subsequent week's "
    "performance on memory tasks even after controlling for stress, "
    "caffeine intake, and prior performance.",
    "We propose a probabilistic framework for missing-data imputation "
    "in heterogeneous tabular datasets. The framework combines a "
    "mixture-of-experts encoder with type-aware decoders, allowing "
    "joint modeling of continuous, categorical, and ordinal features. "
    "On standard benchmarks our method outperforms competitive "
    "baselines by 4-8% in downstream classification accuracy.",
    "Existing approaches to model interpretability often rely on "
    "post-hoc explanations whose fidelity to the underlying model is "
    "difficult to verify. We argue for a constructive alternative: "
    "interpretable-by-design architectures whose internal "
    "representations admit a principled semantic interpretation. We "
    "present three case studies illustrating the tradeoff.",
    "We report on a controlled experiment measuring the effect of "
    "code-review tooling on defect detection rates in a large "
    "open-source project. Reviewers using the new tooling identified "
    "23% more defects per review hour, with no significant change in "
    "false-positive comments. We discuss implications for the design "
    "of developer-facing tooling.",
    "Federated learning enables model training across distributed data "
    "sources without centralizing raw data. We address the open "
    "problem of client heterogeneity, where local data distributions "
    "differ substantially from the global distribution. Our proposed "
    "regularization scheme yields convergence guarantees and improves "
    "test accuracy across five benchmark federated datasets.",
)


def _all_entries() -> list[CorpusEntry]:
    """Build the in-memory corpus from the constants above."""
    entries: list[CorpusEntry] = []
    for i, text in enumerate(CLAUDE_SAMPLES, start=1):
        entries.append(
            CorpusEntry(
                id=f"claude_{i:02d}",
                text=text,
                source_type="claude",
                source_attribution=(
                    "synthetic — written for HumanizeMCP to exhibit "
                    "Claude/Sonnet stylistic markers"
                ),
            )
        )
    for i, text in enumerate(GPT_SAMPLES, start=1):
        entries.append(
            CorpusEntry(
                id=f"gpt_{i:02d}",
                text=text,
                source_type="gpt",
                source_attribution=(
                    "synthetic — written for HumanizeMCP to exhibit "
                    "ChatGPT/GPT-4 stylistic markers"
                ),
            )
        )
    for i, (text, attr) in enumerate(HUMAN_SAMPLES, start=1):
        entries.append(
            CorpusEntry(
                id=f"human_{i:02d}",
                text=text,
                source_type="human",
                source_attribution=attr,
            )
        )
    for i, text in enumerate(ESL_SAMPLES, start=1):
        entries.append(
            CorpusEntry(
                id=f"esl_{i:02d}",
                text=text,
                source_type="esl",
                source_attribution=(
                    "synthetic — written for HumanizeMCP to model ESL writing patterns"
                ),
            )
        )
    for i, text in enumerate(ACADEMIC_SAMPLES, start=1):
        entries.append(
            CorpusEntry(
                id=f"academic_{i:02d}",
                text=text,
                source_type="academic",
                source_attribution=(
                    "synthetic — written for HumanizeMCP to model "
                    "arxiv-abstract academic register"
                ),
            )
        )
    return entries


def write_corpus(path: Path = CORPUS_PATH) -> Path:
    """Write the corpus to ``path`` as JSONL and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = _all_entries()
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")
    return path


def load_corpus(
    path: Path = CORPUS_PATH,
    *,
    source_types: Iterable[str] | None = None,
) -> list[CorpusEntry]:
    """Load the corpus from disk, generating it on demand if missing.

    Args:
        path: JSONL path to load.
        source_types: Optional filter; only entries whose ``source_type``
            is in this collection are returned.
    """
    if not path.exists():
        write_corpus(path)
    entries: list[CorpusEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(CorpusEntry.model_validate_json(line))
    if source_types is not None:
        wanted = set(source_types)
        entries = [e for e in entries if e.source_type in wanted]
    return entries


def corpus_summary() -> dict[str, int]:
    """Return a count of corpus entries per source type."""
    counts: dict[str, int] = {}
    for e in load_corpus():
        counts[e.source_type] = counts.get(e.source_type, 0) + 1
    return counts


__all__ = [
    "CorpusEntry",
    "CORPUS_PATH",
    "write_corpus",
    "load_corpus",
    "corpus_summary",
]


if __name__ == "__main__":
    out = write_corpus()
    print(f"Wrote {sum(1 for _ in out.open())} entries to {out}")
