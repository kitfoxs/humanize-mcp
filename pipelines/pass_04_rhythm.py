"""Pass 4: rhythm / burstiness injection (v0.2.0).

Re-implementation of Bet 2 from docs/REVIEW_v0.1.0.md section 3.5.

Detects sentence-length uniformity and injects variance until the
coefficient of variation (CV) of sentence word counts hits a configurable
target. Burstiness is the single most-used statistical signal in
commercial detectors (research/02 section C.1) and the heuristic
detector in benchmark/detectors.py applies a ``+0.8`` z-score penalty
when CV < 0.25 with 4+ sentences. A run-of-the-mill LLM paragraph sits
at CV ~0.20-0.35; pushing it above 0.7 drops the heuristic AI-prob
from ~22% to ~5-9% on its own.

Strategies, applied in order inside each paragraph:

1. **Split long sentences** (>= 16 words) at the conjunction nearest
   the middle. This adds short-to-medium sentences and lifts the
   low end of the length distribution variance.
2. **Merge short adjacent sentences** (current < 8 words AND next
   < 12 words) with comma + lowercase. Reverses uniform short cadence.
3. **At balanced and aggressive: merge two adjacent medium sentences**
   (each in 12-20 words) with one of ``and`` / ``, but`` / ``; though``
   to create a deliberately long sentence. Bumps the high tail of the
   length distribution. Only fires when no sentence in the paragraph
   already exceeds ``LONG_TAIL_THRESHOLD`` words.
4. **At aggressive: insert deliberate fragments** between two long
   sentences. Style-specific fragment pools live in
   :data:`FRAGMENT_POOLS`.

Strategies 1-4 iterate up to ``MAX_ITERATIONS`` times. The pass exits
early if (a) CV reaches a hard ``HARD_CV_CAP`` (1.2) to avoid
pathological output, or (b) CV reaches ``target_cv`` AND one polish
round has already run, or (c) a strategy round is a no-op.

The v0.1.0 "early exit if cv >= target" was removed: we always run
at least one polish round so the pass can lift CV from a barely-passing
0.4 toward the more comfortable 0.7+ range that drops detector
penalties.

Style-specific fragment pools (FRAGMENT_POOLS):

- ``reddit`` / ``casual_dm`` / ``twitter`` / ``linkedin``:
  casual fragments (``Right.``, ``Yeah.``, ``True.``, ``Same.``,
  ``Honestly.``, ``Lol.``, ``Fair.``)
- ``blog`` / ``esl_friendly`` / ``autistic_friendly``:
  balanced fragments (``Right.``, ``True.``, ``Or not.``,
  ``Fair point.``, ``Maybe.``)
- ``academic_human``:
  subdued discourse markers (``Indeed.``, ``True.``, ``Granted.``,
  ``Notably.``, ``Of course.``)
- ``book_chapter`` / ``creative_fiction``:
  literary one-words (``Quiet.``, ``Or so it seemed.``, ``Maybe.``,
  ``Or did it?``)

Per-intensity default targets:

- ``minimal`` / ``light``: 0.45
- ``balanced``: 0.65
- ``aggressive``: 0.85

Override per style via ``pass_configs.pass_04_rhythm.target_cv`` in
``styles/*.json``. Reddit may want CV 0.95 (very chunky), academic_human
may want 0.55 (subtler).

Determinism: a seeded :class:`random.Random` is constructed from
``config["pass_configs"]["pass_04_rhythm"]["seed"]`` if present, else
the global ``config["seed"]``. The same input + same seed produces
identical output.

Paragraph preservation (introduced in v0.1.1) is preserved: the public
:meth:`RhythmPass.apply` splits on blank lines and processes each
paragraph independently inside :meth:`_apply_one_paragraph`.
"""

from __future__ import annotations

import random
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

from .base import PipelinePass

try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize

    try:
        _nltk_sent_tokenize("Sanity check.")
        _SENT_TOKENIZE = _nltk_sent_tokenize
    except LookupError:
        _SENT_TOKENIZE = None
except ImportError:
    _SENT_TOKENIZE = None


_FALLBACK_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'(])")


def sentence_split(text: str) -> List[str]:
    """Split ``text`` into sentences using NLTK if available, regex fallback otherwise."""
    if _SENT_TOKENIZE is not None:
        return _SENT_TOKENIZE(text)
    return [s.strip() for s in _FALLBACK_SENT_RE.split(text) if s.strip()]


SPLIT_CONJUNCTIONS: Tuple[str, ...] = (
    ", and ",
    ", but ",
    ", so ",
    ", because ",
    ", which ",
    "; ",
)

# Lowered from v0.1.x: real LLM output clusters at 17-20 words/sentence,
# so a >= 22 trigger almost never fired. >= 16 triggers reliably.
LONG_SENTENCE_TRIGGER: int = 16
SHORT_MERGE_LEFT: int = 8       # was 6
SHORT_MERGE_RIGHT: int = 12     # was 9

# Window for "medium" sentences eligible for combine-into-long.
MEDIUM_LO: int = 12
MEDIUM_HI: int = 20

# Don't synthesise more longs if the paragraph already has a long tail.
LONG_TAIL_THRESHOLD: int = 25
# Stop adding longs once this fraction of sentences are already long.
MAX_LONG_RATIO: float = 0.45

# Stop iterating once we hit this; further pumping produces unreadable prose.
HARD_CV_CAP: float = 1.2

MAX_ITERATIONS: int = 4

# Connectors used at balanced/aggressive when synthesising a long sentence.
LONG_CONNECTORS: Tuple[str, ...] = (" and ", ", but ", "; though ")


# Style -> fragment pool. Lookups go through :func:`_fragments_for_style`
# which falls back to the BLOG pool for unknown styles.
FRAGMENT_POOLS: Dict[str, Tuple[str, ...]] = {
    "reddit": ("Right.", "Yeah.", "True.", "Same.", "Honestly.", "Lol.", "Fair."),
    "casual_dm": ("Right.", "Yeah.", "True.", "Same.", "Honestly.", "Lol.", "Fair."),
    "twitter": ("Right.", "Yeah.", "True.", "Same.", "Honestly.", "Lol.", "Fair."),
    "linkedin": ("Right.", "Yeah.", "True.", "Same.", "Honestly.", "Fair."),
    "blog": ("Right.", "True.", "Or not.", "Fair point.", "Maybe."),
    "esl_friendly": ("Right.", "True.", "Or not.", "Fair point.", "Maybe."),
    "autistic_friendly": ("Right.", "True.", "Or not.", "Fair point.", "Maybe."),
    "academic_human": ("Indeed.", "True.", "Granted.", "Notably.", "Of course."),
    "book_chapter": ("Quiet.", "Or so it seemed.", "Maybe.", "Or did it?"),
    "creative_fiction": ("Quiet.", "Or so it seemed.", "Maybe.", "Or did it?"),
}

# Default target CV by intensity. Style overrides apply on top.
INTENSITY_TARGETS: Dict[str, float] = {
    "minimal": 0.45,
    "light": 0.45,
    "balanced": 0.65,
    "aggressive": 0.85,
}

# How often a fragment is inserted (every Nth eligible long-pair).
# Balanced uses a very light hand (every 6th pair) so already-bursty
# input is barely touched. Aggressive uses every 3rd pair to push
# uniform input firmly above CV 0.7. The per-iteration CV-improvement
# guard in :meth:`RhythmPass._apply_one_paragraph` prevents either
# stride from harming input that is already past target.
FRAGMENT_STRIDE_BY_INTENSITY: Dict[str, int] = {
    "minimal": 0,
    "light": 0,
    "balanced": 4,
    "aggressive": 2,
}


class RhythmPass(PipelinePass):
    pass_id = 4
    pass_name = "rhythm"

    def apply(self, text: str, config: Dict[str, Any]) -> str:
        # Paragraph preservation (v0.1.1): process each paragraph separately
        # so we never collapse a double-newline boundary. All real work
        # happens inside :meth:`_apply_one_paragraph`.
        if "\n" in text:
            parts = re.split(r"(\n\s*\n)", text)
            out_parts: List[str] = []
            for part in parts:
                if part.strip() == "" or "\n" in part:
                    out_parts.append(part)
                else:
                    out_parts.append(self._apply_one_paragraph(part, config))
            return "".join(out_parts)
        return self._apply_one_paragraph(text, config)

    def _apply_one_paragraph(self, text: str, config: Dict[str, Any]) -> str:
        self.reset_changes()
        intensity = str(config.get("intensity", "balanced"))
        target_cv = self._resolve_target_cv(config, intensity)
        rng = self._rng_from_config(config)
        style_name = self._style_name(config)
        fragments = _fragments_for_style(style_name)
        fragment_stride = FRAGMENT_STRIDE_BY_INTENSITY.get(intensity, 0)

        sentences = sentence_split(text)
        if len(sentences) < 3:
            return text

        polish_done = False
        for _iteration in range(MAX_ITERATIONS):
            cv = self._coefficient_of_variation(sentences)
            if cv >= HARD_CV_CAP:
                break
            if cv >= target_cv:
                if polish_done:
                    break
                polish_done = True

            cv_at_iter_start = cv
            before_iter = list(sentences)

            # Structural cleanups: always safe to apply. They fix obvious
            # uniformity problems (too-long sentences, run-on shorts).
            sentences = self._split_long_sentences(sentences)
            sentences = self._merge_short_adjacent(sentences)

            # Variance pumps: only commit if they actually raise CV.
            # Protects already-bursty input from the polish round.
            if intensity in ("balanced", "aggressive"):
                candidate = self._merge_mediums_into_long(
                    sentences, rng, target_cv
                )
                if self._coefficient_of_variation(candidate) > self._coefficient_of_variation(sentences):
                    sentences = candidate
            if intensity == "aggressive":
                # Stretch the high tail: pick the longest sentence and
                # combine it with an adjacent medium to create a mega-long
                # (40+ words). This widens the spread enough to clear
                # CV >= 0.8 reliably across seeds.
                candidate = self._extend_longest(sentences, rng)
                if self._coefficient_of_variation(candidate) > self._coefficient_of_variation(sentences):
                    sentences = candidate
            if intensity == "aggressive" and fragment_stride > 0:
                candidate = self._insert_fragments(
                    sentences, target_cv, fragment_stride, fragments, rng
                )
                if self._coefficient_of_variation(candidate) > self._coefficient_of_variation(sentences):
                    sentences = candidate
            elif intensity == "balanced" and fragment_stride > 0:
                # Light fragment touch at balanced. Same commit guard.
                candidate = self._insert_fragments(
                    sentences, target_cv, fragment_stride, fragments, rng
                )
                if self._coefficient_of_variation(candidate) > self._coefficient_of_variation(sentences):
                    sentences = candidate

            cv_after = self._coefficient_of_variation(sentences)
            # Hill-climbing guards. Never let an iteration regress CV
            # below where it started; protects already-bursty input
            # AND prevents the loop from over-iterating on a high target.
            if cv_after < cv_at_iter_start:
                sentences = before_iter
                break
            if sentences == before_iter:
                break

        return self._reassemble(text, sentences)

    @staticmethod
    def _resolve_target_cv(config: Dict[str, Any], intensity: str) -> float:
        # Per-style override lives in
        # config["style"]["pass_configs"]["pass_04_rhythm"]["target_cv"]
        # but the orchestrator also flattens that into the top-level
        # pass config, so check both.
        if "target_cv" in config:
            return float(config["target_cv"])
        style = config.get("style", {})
        if isinstance(style, dict):
            override = (
                style.get("pass_configs", {})
                .get("pass_04_rhythm", {})
                .get("target_cv")
            )
            if override is not None:
                return float(override)
        return INTENSITY_TARGETS.get(intensity, 0.65)

    @staticmethod
    def _rng_from_config(config: Dict[str, Any]) -> random.Random:
        seed = config.get("seed", 42)
        # Accept either a top-level seed override on this pass or the
        # global seed. Cast to int defensively.
        if "pass_seed" in config:
            seed = config["pass_seed"]
        try:
            seed_int = int(seed)
        except (TypeError, ValueError):
            seed_int = 42
        return random.Random(seed_int)

    @staticmethod
    def _style_name(config: Dict[str, Any]) -> str:
        style = config.get("style", {})
        if isinstance(style, dict):
            return str(style.get("name", "blog"))
        if isinstance(style, str):
            return style
        return "blog"

    @staticmethod
    def _coefficient_of_variation(sents: List[str]) -> float:
        lengths = [len(s.split()) for s in sents]
        if len(lengths) < 2:
            return 0.0
        mean = statistics.mean(lengths)
        if mean == 0:
            return 0.0
        return statistics.pstdev(lengths) / mean

    def _split_long_sentences(self, sents: List[str]) -> List[str]:
        out: List[str] = []
        for s in sents:
            if len(s.split()) < LONG_SENTENCE_TRIGGER:
                out.append(s)
                continue
            split_at = self._find_split_point(s)
            if split_at is None:
                out.append(s)
                continue
            left, right = s[:split_at].rstrip(", ;"), s[split_at:].lstrip()
            if not left.endswith((".", "!", "?")):
                left += "."
            right = right[:1].upper() + right[1:] if right else right
            self.log_change("rhythm_split", s, f"{left} {right}")
            out.append(left)
            out.append(right)
        return out

    @staticmethod
    def _find_split_point(s: str) -> Optional[int]:
        # Pick the conjunction occurrence nearest the middle.
        mid = len(s) // 2
        best_idx: Optional[int] = None
        best_dist = 10**9
        for conj in SPLIT_CONJUNCTIONS:
            idx = 0
            while True:
                idx = s.find(conj, idx)
                if idx == -1:
                    break
                dist = abs(idx - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx + len(conj)
                idx += 1
        if best_idx is not None:
            return best_idx
        # Fallback: split at the comma nearest the middle (no conjunction
        # available). We only use this for genuinely long sentences so
        # the resulting halves are still grammatical-ish.
        comma_indices = [i for i, ch in enumerate(s) if ch == ","]
        if comma_indices and len(s.split()) >= LONG_SENTENCE_TRIGGER + 4:
            comma_pick = min(comma_indices, key=lambda i: abs(i - mid))
            # Skip past the comma + any whitespace.
            cut = comma_pick + 1
            while cut < len(s) and s[cut] == " ":
                cut += 1
            if 0 < cut < len(s):
                return cut
        return None

    def _merge_short_adjacent(self, sents: List[str]) -> List[str]:
        out: List[str] = []
        i = 0
        while i < len(sents):
            current = sents[i]
            if (
                i + 1 < len(sents)
                and len(current.split()) < SHORT_MERGE_LEFT
                and len(sents[i + 1].split()) < SHORT_MERGE_RIGHT
            ):
                nxt = sents[i + 1]
                merged = (
                    current.rstrip(".!?") + ", " + nxt[:1].lower() + nxt[1:]
                )
                self.log_change("rhythm_merge", f"{current} {nxt}", merged)
                out.append(merged)
                i += 2
                continue
            out.append(current)
            i += 1
        return out

    def _merge_mediums_into_long(
        self,
        sents: List[str],
        rng: random.Random,
        target_cv: float,
    ) -> List[str]:
        """Combine adjacent medium sentences into deliberately long ones.

        Skipped entirely for short paragraphs (< 5 sentences) where
        merging tends to collapse variance instead of injecting it.

        Loops until one of:
          * CV reaches ``target_cv``,
          * the proportion of "long" sentences (>= LONG_TAIL_THRESHOLD)
            exceeds ``MAX_LONG_RATIO``,
          * no eligible adjacent medium pairs remain, or
          * we've performed ``max_merges`` joins (capped to avoid
            collapsing the whole paragraph into a wall of long).

        Each merge picks an eligible pair via ``rng`` so the choice is
        deterministic per seed but spread across the paragraph.
        """
        if len(sents) < 5:
            return sents
        max_merges = max(1, len(sents) // 3)
        merges_done = 0
        sents = list(sents)
        while merges_done < max_merges:
            lengths = [len(s.split()) for s in sents]
            if not lengths:
                break
            long_count = sum(1 for L in lengths if L >= LONG_TAIL_THRESHOLD)
            if long_count / len(lengths) > MAX_LONG_RATIO:
                break
            if self._coefficient_of_variation(sents) >= target_cv:
                break
            candidates: List[int] = []
            for i in range(len(sents) - 1):
                if (
                    MEDIUM_LO <= lengths[i] <= MEDIUM_HI
                    and MEDIUM_LO <= lengths[i + 1] <= MEDIUM_HI
                ):
                    candidates.append(i)
            if not candidates:
                break
            pick = rng.choice(candidates)
            connector = rng.choice(LONG_CONNECTORS)
            left = sents[pick].rstrip(".!?")
            right = sents[pick + 1]
            # Lowercase the start of the right half so the join reads
            # naturally regardless of which connector we picked.
            right_text = right[:1].lower() + right[1:]
            merged = left + connector + right_text
            self.log_change(
                "rhythm_long_synth",
                f"{sents[pick]} {sents[pick + 1]}",
                merged,
            )
            sents = sents[:pick] + [merged] + sents[pick + 2 :]
            merges_done += 1
        return sents

    def _extend_longest(
        self,
        sents: List[str],
        rng: random.Random,
    ) -> List[str]:
        """Stretch the high tail by combining the longest sentence with a neighbor.

        Aggressive-only. Picks the longest sentence; if it has an
        adjacent sentence in [``MEDIUM_LO``, ``MEDIUM_HI``] words, joins
        them with a connector to produce a deliberately mega-long
        sentence (40+ words). Skips if no eligible neighbor exists or
        if the longest is already past 45 words.
        """
        if len(sents) < 3:
            return sents
        lengths = [len(s.split()) for s in sents]
        longest_idx = max(range(len(sents)), key=lambda i: lengths[i])
        if lengths[longest_idx] >= 45:
            return sents
        candidates: List[int] = []
        if longest_idx > 0 and MEDIUM_LO <= lengths[longest_idx - 1] <= MEDIUM_HI:
            candidates.append(longest_idx - 1)
        if (
            longest_idx + 1 < len(sents)
            and MEDIUM_LO <= lengths[longest_idx + 1] <= MEDIUM_HI
        ):
            candidates.append(longest_idx + 1)
        if not candidates:
            return sents
        neighbor = rng.choice(candidates)
        connector = rng.choice(LONG_CONNECTORS)
        if neighbor < longest_idx:
            left_sent = sents[neighbor].rstrip(".!?")
            right_sent = sents[longest_idx]
            right_text = right_sent[:1].lower() + right_sent[1:]
            merged = left_sent + connector + right_text
            new_sents = (
                sents[:neighbor] + [merged] + sents[longest_idx + 1 :]
            )
        else:
            left_sent = sents[longest_idx].rstrip(".!?")
            right_sent = sents[neighbor]
            right_text = right_sent[:1].lower() + right_sent[1:]
            merged = left_sent + connector + right_text
            new_sents = (
                sents[:longest_idx] + [merged] + sents[neighbor + 1 :]
            )
        self.log_change("rhythm_long_extend", sents[longest_idx], merged)
        return new_sents

    def _insert_fragments(
        self,
        sents: List[str],
        target_cv: float,
        stride: int,
        fragments: Tuple[str, ...],
        rng: random.Random,
    ) -> List[str]:
        """Insert fragments between long-sentence pairs.

        Walks the sentence list and inserts a randomly-chosen fragment
        between every ``stride``-th eligible adjacent pair where both
        sides are reasonably long (>= 10 words). Skips early once the
        running CV passes ``target_cv * 1.05`` so we don't over-fragment.
        """
        if not fragments or stride <= 0 or len(sents) < 2:
            return sents
        # Adaptive stride: on small paragraphs the configured stride
        # (e.g. 4 for balanced) would never fire because there aren't
        # enough eligible pairs. Cap stride so at least one fragment
        # gets a chance per ~3 sentences of input.
        effective_stride = min(stride, max(1, len(sents) // 3))

        out: List[str] = []
        eligible_seen = 0
        inserted_any = False
        for i, s in enumerate(sents):
            out.append(s)
            if i >= len(sents) - 1:
                continue
            here_len = len(s.split())
            next_len = len(sents[i + 1].split())
            # Both sides need some weight for a fragment between them
            # to actually create a contrast (and not a pile of three
            # short sentences in a row). 8+ words covers most LLM prose.
            if here_len < 8 or next_len < 8:
                continue
            eligible_seen += 1
            if eligible_seen % effective_stride != 0:
                continue
            # Stop adding fragments once CV is comfortably past target.
            current_cv = self._coefficient_of_variation(out + sents[i + 1 :])
            if current_cv >= target_cv * 1.05:
                continue
            frag = rng.choice(fragments)
            out.append(frag)
            self.log_change("rhythm_fragment_insert", "", frag)
            inserted_any = True

        if not inserted_any:
            return sents
        return out

    @staticmethod
    def _reassemble(original: str, sentences: List[str]) -> str:
        out = " ".join(s.strip() for s in sentences if s.strip())
        if original.endswith("\n"):
            out += "\n"
        return out


def _fragments_for_style(style_name: str) -> Tuple[str, ...]:
    """Return the fragment pool for a style, defaulting to the blog pool."""
    if style_name in FRAGMENT_POOLS:
        return FRAGMENT_POOLS[style_name]
    return FRAGMENT_POOLS["blog"]
