"""AI-text detector wrappers used by the HumanizeMCP benchmark suite.

Each detector is a small class with a uniform ``score(text)`` method that
returns a :class:`DetectorScore` value. Heavy resources (models, tokenizers)
are loaded lazily on first use so importing this module is cheap.

The wrapped detectors fall into three tiers:

1. **Lightweight, always-available**:
   :class:`HeuristicDetector` is a pure-Python feature-based scorer that
   needs no model and is the transparent baseline.

2. **HuggingFace classifier wrappers** (download on first use):
   * :class:`RobertaOpenAIDetector` (``roberta-base-openai-detector``)
   * :class:`ChatGPTRobertaDetector` (``Hello-SimpleAI/chatgpt-detector-roberta``)
   * :class:`DesklibAIDetector` (``desklib/ai-text-detector-v1.01`` if available)

3. **Zero-shot, reference-LM based** (download on first use; can be heavy):
   * :class:`FastDetectGPT` — Bao et al. 2023, conditional probability
     curvature. Implemented here against a single reference LM (defaults to
     ``gpt2``) as a pragmatic, CPU-feasible variant.
   * :class:`BinocularsDetector` — Hans et al. 2024, ratio of cross-entropy
     to perplexity between an "observer" and "performer" LM. Falls back to a
     lighter model pair if the canonical Falcon-7B pair is not available.

Calibration warnings — many of these detectors are documented to have high
false-positive rates on certain registers (academic writing, ESL writers,
short text). The :class:`Detector.bias_notes` field surfaces this metadata
to downstream report generators so it is never silently averaged into a
"verdict".
"""

from __future__ import annotations

import logging
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import ClassVar, Optional

from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

# Detector registry tags — used by reports to distinguish detector quality.
TIER_HEURISTIC = "heuristic"
TIER_CLASSIFIER = "classifier"
TIER_ZERO_SHOT = "zero_shot"


class DetectorScore(BaseModel):
    """Uniform output of a single detector run.

    Attributes:
        detector: Name of the detector that produced this score.
        ai_probability: Probability the text is AI-generated, in [0, 1].
        human_probability: Probability the text is human-written, in [0, 1].
        confidence: How sure the detector is in its top label, in [0, 1].
        verdict: One of ``"ai"``, ``"human"``, or ``"uncertain"``.
        latency_seconds: Wall-clock time the score took to compute.
        tier: One of ``heuristic``, ``classifier``, ``zero_shot``.
        bias_notes: Optional list of known bias / calibration caveats that
            downstream code should surface in any aggregate report.
        details: Free-form structured detail (per-detector specifics).
        error: If non-empty, the detector failed and downstream code should
            treat the score as missing rather than as evidence.
    """

    detector: str
    ai_probability: float = Field(ge=0.0, le=1.0)
    human_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    verdict: str
    latency_seconds: float = 0.0
    tier: str = TIER_CLASSIFIER
    bias_notes: list[str] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)
    error: str = ""

    @classmethod
    def from_ai_prob(
        cls,
        detector: str,
        ai_prob: float,
        *,
        latency_seconds: float = 0.0,
        tier: str = TIER_CLASSIFIER,
        bias_notes: Optional[list[str]] = None,
        details: Optional[dict] = None,
        uncertain_band: tuple[float, float] = (0.4, 0.6),
    ) -> "DetectorScore":
        """Build a :class:`DetectorScore` from a single AI probability.

        Args:
            detector: Detector name.
            ai_prob: Probability the text is AI-generated.
            latency_seconds: How long the score took.
            tier: Detector tier label.
            bias_notes: Calibration caveats to attach.
            details: Per-detector free-form details.
            uncertain_band: Probabilities inside this open interval are
                reported as ``"uncertain"`` rather than a hard verdict.
        """
        ai_prob = max(0.0, min(1.0, float(ai_prob)))
        human_prob = 1.0 - ai_prob
        if uncertain_band[0] < ai_prob < uncertain_band[1]:
            verdict = "uncertain"
        else:
            verdict = "ai" if ai_prob >= 0.5 else "human"
        confidence = max(ai_prob, human_prob)
        return cls(
            detector=detector,
            ai_probability=ai_prob,
            human_probability=human_prob,
            confidence=confidence,
            verdict=verdict,
            latency_seconds=latency_seconds,
            tier=tier,
            bias_notes=bias_notes or [],
            details=details or {},
        )

    @classmethod
    def errored(
        cls,
        detector: str,
        error: str,
        *,
        tier: str = TIER_CLASSIFIER,
    ) -> "DetectorScore":
        """Construct a placeholder score representing a detector failure."""
        return cls(
            detector=detector,
            ai_probability=0.5,
            human_probability=0.5,
            confidence=0.0,
            verdict="error",
            tier=tier,
            error=error,
        )


@dataclass
class _LoadedModel:
    """Container for a lazy-loaded HuggingFace model + tokenizer pair."""

    tokenizer: object
    model: object
    device: str = "cpu"


class Detector:
    """Abstract base class for all detectors.

    Subclasses must set :attr:`name`, :attr:`tier`, and override
    :meth:`_score_impl`. They should also populate :attr:`bias_notes` with
    any documented calibration caveats (e.g. high FPR on ESL text) so the
    benchmark report can surface them.
    """

    name: str = "abstract"
    tier: str = TIER_CLASSIFIER
    bias_notes: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self._loaded = False

    def score(self, text: str) -> DetectorScore:
        """Public entry point. Times the underlying scoring call and wraps errors."""
        if not text or not text.strip():
            return DetectorScore.errored(self.name, "empty input text", tier=self.tier)
        t0 = time.time()
        try:
            score = self._score_impl(text)
        except Exception as exc:  # noqa: BLE001 - we want to surface any failure
            LOGGER.exception("Detector %s failed", self.name)
            return DetectorScore.errored(self.name, repr(exc), tier=self.tier)
        score.latency_seconds = time.time() - t0
        return score

    # Subclasses override:
    def _score_impl(self, text: str) -> DetectorScore:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tier 1: heuristic, always available
# ---------------------------------------------------------------------------


# Surface tells we surveyed in research/02_ai_tells_catalog.md. Kept
# deliberately small and conservative — this is a baseline, not a winner.
_TELL_PHRASES = (
    "delve into",
    "in conclusion",
    "in summary",
    "it is important to note",
    "it's important to note",
    "it is worth noting",
    "moreover",
    "furthermore",
    "additionally",
    "in today's world",
    "in today's fast-paced",
    "navigating the complexities",
    "navigating the landscape",
    "the realm of",
    "tapestry",
    "at its core",
    "at the heart of",
    "underscores the importance",
    "plays a crucial role",
    "plays a vital role",
    "leverage",
    "harness the power",
    "embark on",
    "let's dive",
    "robust",
    "seamless",
    "comprehensive",
    "multifaceted",
    "ever-evolving",
    "stands as a testament",
    "not just",  # over-used "not just X but Y" template
    "but rather",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter; good enough for stylometric features."""
    text = text.strip()
    if not text:
        return []
    pieces = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in pieces if p.strip()]


def _word_tokens(text: str) -> list[str]:
    """Word tokenizer that strips punctuation."""
    return re.findall(r"[A-Za-z']+", text.lower())


def burstiness(text: str) -> float:
    """Compute a simple burstiness score on sentence-length variance.

    Burstiness here is the coefficient of variation of sentence word counts:
    ``stdev / mean``. Higher = more human-like (more variety). This is a
    proxy for the GPTZero-style burstiness metric, which uses per-token
    perplexity variance — we substitute sentence length because we do not
    require a reference LM.

    Returns:
        Coefficient of variation, or ``0.0`` for fewer than two sentences.
    """
    sents = _split_sentences(text)
    if len(sents) < 2:
        return 0.0
    lengths = [len(_word_tokens(s)) for s in sents if _word_tokens(s)]
    if len(lengths) < 2:
        return 0.0
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0
    sd = statistics.pstdev(lengths)
    return sd / mean


def lexical_diversity(text: str) -> float:
    """Type-token ratio as a coarse lexical diversity proxy."""
    tokens = _word_tokens(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def count_tells(text: str) -> tuple[int, list[str]]:
    """Count surface AI tells. Returns (count, list_of_matches)."""
    lower = text.lower()
    hits = [phrase for phrase in _TELL_PHRASES if phrase in lower]
    # Em-dash count: include both U+2014 and the ASCII " -- " convention.
    # Do not double-count spaces around U+2014 since text.count("\u2014")
    # already covers that case.
    em_dashes = text.count("\u2014") + text.count(" -- ")
    if em_dashes >= 2:
        hits.append(f"em_dash_x{em_dashes}")
    # Smart quotes are common in LLM output that has been auto-formatted.
    smart_quotes = sum(text.count(c) for c in "\u2018\u2019\u201c\u201d")
    if smart_quotes >= 4:
        hits.append(f"smart_quotes_x{smart_quotes}")
    return len(hits), hits


class HeuristicDetector(Detector):
    """Pure-Python feature-based detector. No model needed.

    Combines three signals:

    * **Burstiness** (sentence-length CV). Higher is more human-like.
    * **Lexical diversity** (type-token ratio). Mid-range expected.
    * **Surface AI tells** (catalog from research/02). More hits => more AI.

    The combination is a calibrated logistic, with weights chosen by
    hand to roughly center on 0.5 for "neutral" prose and push toward 1.0
    when many tells are present and burstiness is low.
    """

    name = "heuristic"
    tier = TIER_HEURISTIC
    bias_notes: ClassVar[list[str]] = [
        "Heuristic baseline; no peer-reviewed validation.",
        "Surface tells list is hand-curated and English-only.",
    ]

    def _score_impl(self, text: str) -> DetectorScore:
        b = burstiness(text)
        td = lexical_diversity(text)
        n_tells, tell_hits = count_tells(text)
        sents = _split_sentences(text)
        n_sents = len(sents)

        # Normalize features. Burstiness > 0.5 is "human-bursty"; tells_density
        # is per 100 words. Diversity is type-token ratio.
        word_count = max(1, len(_word_tokens(text)))
        tells_per_100 = (n_tells / word_count) * 100

        # Linear combination then logistic. Weights tuned by sanity checks
        # rather than fitted on a real dataset — this is a transparent
        # baseline, not a trained classifier.
        z = 0.0
        z += -2.5 * b  # high burstiness pushes toward human
        z += +1.6 * tells_per_100  # tells push toward AI
        z += -1.2 * (td - 0.45)  # very low diversity also pushes toward AI
        if n_sents >= 4 and b < 0.25:
            # Strong signal: 4+ sentences, all roughly the same length.
            z += 0.8

        # Logistic squash, then center mid-band wide for short text where
        # we have low confidence either way.
        ai_prob = 1.0 / (1.0 + math.exp(-z))
        if word_count < 30:
            # Pull toward 0.5 (uncertainty) when we barely have any text.
            ai_prob = 0.5 + (ai_prob - 0.5) * 0.4

        return DetectorScore.from_ai_prob(
            self.name,
            ai_prob,
            tier=self.tier,
            bias_notes=self.bias_notes,
            details={
                "burstiness": round(b, 4),
                "lexical_diversity": round(td, 4),
                "tells_count": n_tells,
                "tells_hits": tell_hits[:10],
                "tells_per_100_words": round(tells_per_100, 4),
                "word_count": word_count,
                "sentence_count": n_sents,
            },
        )


# ---------------------------------------------------------------------------
# Tier 2: HuggingFace classifier wrappers
# ---------------------------------------------------------------------------


class _HFClassifierDetector(Detector):
    """Shared loader for two-class HF sequence-classification detectors."""

    model_id: str = ""
    # Index in the model's logits that corresponds to the "AI / fake" class.
    ai_label_index: int = 1

    def __init__(self) -> None:
        super().__init__()
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                f"transformers/torch not installed; cannot load {self.model_id}"
            ) from exc
        LOGGER.info("Loading detector model %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.eval()
        self._torch = torch
        self._loaded = True

    def _score_impl(self, text: str) -> DetectorScore:
        self._ensure_loaded()
        torch = self._torch
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            out = self._model(**inputs)
        probs = torch.softmax(out.logits, dim=-1)[0].tolist()
        if len(probs) < 2:
            raise RuntimeError(
                f"{self.model_id} returned {len(probs)} classes; expected at least 2"
            )
        ai_prob = float(probs[self.ai_label_index])
        return DetectorScore.from_ai_prob(
            self.name,
            ai_prob,
            tier=self.tier,
            bias_notes=self.bias_notes,
            details={"raw_probs": probs, "model_id": self.model_id},
        )


class RobertaOpenAIDetector(_HFClassifierDetector):
    """Wraps the original ``roberta-base-openai-detector``.

    This is the canonical academic baseline. It was trained against GPT-2
    1.5B output and is documented to be poorly calibrated on modern (2024+)
    LLM output. See research/01 for details. We include it for historical
    comparability, not as a primary signal.
    """

    name = "roberta_openai"
    model_id = "roberta-base-openai-detector"
    ai_label_index = 1  # 0=real (human), 1=fake (machine)
    bias_notes: ClassVar[list[str]] = [
        "Trained on GPT-2 outputs; biased on modern LLM output.",
        "Documented to flag much real human text as 99%+ AI on out-of-domain inputs.",
        "Use as one signal among many; do not treat as ground truth.",
    ]


class ChatGPTRobertaDetector(_HFClassifierDetector):
    """Wraps ``Hello-SimpleAI/chatgpt-detector-roberta`` (HC3 dataset)."""

    name = "chatgpt_roberta"
    model_id = "Hello-SimpleAI/chatgpt-detector-roberta"
    # Per the model card: label 0 = ChatGPT, label 1 = Human.
    ai_label_index = 0
    bias_notes: ClassVar[list[str]] = [
        "Trained on the HC3 dataset; ChatGPT-era distribution.",
        "Reported to have ESL false-positive bias.",
    ]


class DesklibAIDetector(_HFClassifierDetector):
    """Wraps ``desklib/ai-text-detector-v1.01`` if reachable.

    This model uses a custom architecture (see its README); we attempt the
    AutoModelForSequenceClassification path first and fall back to logging a
    helpful error if it requires custom code.
    """

    name = "desklib"
    model_id = "desklib/ai-text-detector-v1.01"
    ai_label_index = 1
    bias_notes: ClassVar[list[str]] = [
        "Vendor-released checkpoint; methodology not peer-reviewed.",
        "Custom architecture may require trust_remote_code at load time.",
    ]

    def _ensure_loaded(self) -> None:  # noqa: D401 - same shape as parent
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers/torch not installed") from exc
        LOGGER.info("Loading detector model %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        try:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Desklib detector ({exc!r}); install separately."
            ) from exc
        self._model.eval()
        self._torch = torch
        self._loaded = True


# ---------------------------------------------------------------------------
# Tier 3: zero-shot detectors (Fast-DetectGPT, Binoculars)
# ---------------------------------------------------------------------------


class _ReferenceLM:
    """Lazy holder for a causal-LM used as a reference for perplexity / curvature."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        LOGGER.info("Loading reference LM %s", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._model.eval()
        self._torch = torch
        self._loaded = True

    def token_logprobs(self, text: str, max_length: int = 512) -> tuple:
        """Return per-position next-token log-probabilities and the input ids."""
        self.load()
        torch = self._torch
        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            raise RuntimeError("Reference LM requires at least 2 tokens of input")
        with torch.no_grad():
            out = self._model(**enc)
        logits = out.logits  # [1, T, V]
        # For each position t in [0, T-2], we want the log-prob of the actual
        # next token (input_ids[t+1]) under the model's distribution at t.
        log_probs = torch.log_softmax(logits, dim=-1)
        next_tokens = input_ids[:, 1:].unsqueeze(-1)  # [1, T-1, 1]
        gathered = log_probs[:, :-1, :].gather(-1, next_tokens).squeeze(-1)  # [1, T-1]
        return gathered, input_ids, log_probs

    def cross_entropy(self, text: str, max_length: int = 512) -> float:
        """Mean cross-entropy of ``text`` under this model (in nats)."""
        gathered, _, _ = self.token_logprobs(text, max_length=max_length)
        return float(-gathered.mean().item())


class FastDetectGPT(Detector):
    """Conditional probability curvature detector, Bao et al. 2023.

    The original method needs a sampling step using a stronger model.
    For a CPU-feasible variant we approximate the curvature using a
    per-token comparison between the actual log-probability of the
    observed token and the expected log-probability under the model's
    own distribution at that position. Specifically:

        score = mean_t [ logp(x_t) - mean_y~p(.|x<t)[ logp(y) ] ] / sigma

    where ``mean_y~p(.|x<t)[ logp(y) ]`` is the entropy-style quantity
    ``sum_y p(y) * log p(y)`` and ``sigma`` is the std of the same
    quantity across positions. Higher score => more AI-like (the actual
    token is consistently high-probability under the LM's own view).
    This is a lightweight variant of the published method and is
    explicitly labeled as such in the report.

    Args:
        reference_model: HF model id used as the reference LM. Defaults
            to ``"gpt2"`` for CPU feasibility. Use ``"EleutherAI/gpt-neo-2.7B"``
            for closer parity with the published numbers.
    """

    name = "fast_detect_gpt"
    tier = TIER_ZERO_SHOT
    bias_notes: ClassVar[list[str]] = [
        "CPU-friendly variant of Bao et al. 2023; not the canonical implementation.",
        "Calibration depends on choice of reference LM.",
    ]

    def __init__(self, reference_model: str = "gpt2") -> None:
        super().__init__()
        self._lm = _ReferenceLM(reference_model)
        self._reference_model = reference_model

    def _score_impl(self, text: str) -> DetectorScore:
        import math as _math

        gathered, _, log_probs = self._lm.token_logprobs(text)
        # gathered: [1, T-1] log-probs of observed next tokens
        observed = gathered.squeeze(0).cpu().tolist()
        # Per-position entropy of the model's distribution.
        torch = self._lm._torch
        probs = torch.exp(log_probs[:, :-1, :])
        entropies = -(probs * log_probs[:, :-1, :]).sum(dim=-1).squeeze(0).cpu().tolist()

        # The Fast-DetectGPT statistic in the paper is the conditional
        # probability curvature; here we use its single-model proxy:
        # how much higher the observed token's log-prob is than the
        # negative entropy at that step (i.e. is the observed token
        # consistently above the typical sample?), normalized by the std.
        deltas = [obs - (-ent) for obs, ent in zip(observed, entropies)]
        if len(deltas) < 2:
            raise RuntimeError("Not enough tokens for Fast-DetectGPT scoring")
        mean_delta = statistics.mean(deltas)
        sd_delta = statistics.pstdev(deltas) or 1e-6
        z = mean_delta / sd_delta

        # Squash to [0, 1]. Empirically z is in roughly [-3, 3]; we map
        # via logistic with a gentle scale.
        ai_prob = 1.0 / (1.0 + _math.exp(-z))

        return DetectorScore.from_ai_prob(
            self.name,
            ai_prob,
            tier=self.tier,
            bias_notes=self.bias_notes,
            details={
                "reference_model": self._reference_model,
                "z_statistic": round(z, 4),
                "mean_delta_nats": round(mean_delta, 4),
                "std_delta_nats": round(sd_delta, 4),
                "n_tokens_scored": len(deltas),
                "note": "single-LM proxy; see docstring",
            },
        )


class BinocularsDetector(Detector):
    """Binoculars detector (Hans et al. 2024).

    The published method uses the ratio of cross-entropy to perplexity
    between an observer LM (e.g. ``falcon-7b``) and a performer LM
    (e.g. ``falcon-7b-instruct``). To stay CPU-feasible we default to
    a much smaller pair: ``gpt2`` vs ``distilgpt2``. The score is the
    Binoculars statistic ``B = log_ppl_observer / cross_entropy_performer``;
    higher B means the text looks more "machine-typical" relative to the
    pair, which we map to a higher AI probability via a calibrated logistic.

    The class falls back to an error score if either model can't be loaded
    rather than crashing the benchmark suite.
    """

    name = "binoculars"
    tier = TIER_ZERO_SHOT
    bias_notes: ClassVar[list[str]] = [
        "CPU-friendly variant of Hans et al. 2024; default model pair is gpt2/distilgpt2.",
        "Calibration is approximate; absolute scores are not directly comparable to the paper.",
    ]

    def __init__(
        self,
        observer_model: str = "gpt2",
        performer_model: str = "distilgpt2",
    ) -> None:
        super().__init__()
        self._observer = _ReferenceLM(observer_model)
        self._performer = _ReferenceLM(performer_model)
        self._observer_id = observer_model
        self._performer_id = performer_model

    def _score_impl(self, text: str) -> DetectorScore:
        import math as _math

        ce_observer = self._observer.cross_entropy(text)
        ce_performer = self._performer.cross_entropy(text)
        if ce_performer <= 1e-6:
            raise RuntimeError("Performer cross-entropy was zero; degenerate input")
        # Binoculars statistic: ratio of the two cross-entropies.
        # Lower B => more AI-like; we invert the sign in the logistic.
        b = ce_observer / ce_performer
        # Calibration: AI text empirically gives B near 1.0; human text gives
        # B noticeably above 1.0. We center the logistic at B=1.05 so values
        # at-or-below 1.0 land in the AI region.
        z = -8.0 * (b - 1.05)
        ai_prob = 1.0 / (1.0 + _math.exp(-z))

        return DetectorScore.from_ai_prob(
            self.name,
            ai_prob,
            tier=self.tier,
            bias_notes=self.bias_notes,
            details={
                "observer_model": self._observer_id,
                "performer_model": self._performer_id,
                "binoculars_statistic": round(b, 4),
                "ce_observer_nats": round(ce_observer, 4),
                "ce_performer_nats": round(ce_performer, 4),
            },
        )


# ---------------------------------------------------------------------------
# Convenience: the default suite the benchmark uses if no list is provided.
# ---------------------------------------------------------------------------


def default_detectors(*, include_optional: bool = False) -> list[Detector]:
    """Return the default detector list used by :class:`BenchmarkSuite`.

    The "always-on" tier is the heuristic detector plus the two small
    HuggingFace classifiers. The "optional" tier (zero-shot scorers and
    Desklib) only runs when ``include_optional`` is True because they
    download multi-hundred-MB models on first use.

    Args:
        include_optional: If True, append Fast-DetectGPT, Binoculars,
            and Desklib to the returned list.
    """
    detectors: list[Detector] = [
        HeuristicDetector(),
        RobertaOpenAIDetector(),
        ChatGPTRobertaDetector(),
    ]
    if include_optional:
        detectors.append(DesklibAIDetector())
        detectors.append(FastDetectGPT())
        detectors.append(BinocularsDetector())
    return detectors


__all__ = [
    "DetectorScore",
    "Detector",
    "HeuristicDetector",
    "RobertaOpenAIDetector",
    "ChatGPTRobertaDetector",
    "DesklibAIDetector",
    "FastDetectGPT",
    "BinocularsDetector",
    "default_detectors",
    "burstiness",
    "lexical_diversity",
    "count_tells",
    "TIER_HEURISTIC",
    "TIER_CLASSIFIER",
    "TIER_ZERO_SHOT",
]
