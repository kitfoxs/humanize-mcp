"""Local-model paraphrase backends for pass 9.

This module isolates the heavy ML dependencies (transformers, torch,
sentence-transformers) so the pure-Python light paraphrase path stays
zero-cost on import. Each class lazy-loads its underlying model on first
use and degrades gracefully if the load fails (no network, no disk space,
optional dependency missing).

Two backends:

* :class:`T5Paraphraser` wraps ``humarin/chatgpt_paraphraser_on_T5_base``
  for batched deterministic paraphrase and stochastic candidate sampling.
* :class:`SemanticGate` wraps ``sentence-transformers/all-MiniLM-L6-v2``
  for the anti-drift similarity check. If sentence-transformers is not
  installed (it ships in ``requirements-heavy.txt`` only), the gate
  becomes a no-op that accepts every candidate. The pipeline still works,
  it just loses the drift safety net.

Reference: docs/REVIEW_v0.1.0.md section 3.2 (bet 1) and
research/04_humanization_techniques.md sections 1.2-1.3.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


# Tokens that, if echoed into the T5 prompt, can derail generation or
# leak special-token semantics into the output. We strip them before the
# tokenizer ever sees the text. ``<extra_id_N>`` covers the T5 sentinel
# range (0-99 is the standard span); we use a regex rather than enumerate.
_INJECTION_PATTERNS = (
    re.compile(r"</?s>"),
    re.compile(r"<\|im_start\|>"),
    re.compile(r"<\|im_end\|>"),
    re.compile(r"<extra_id_\d{1,2}>"),
)


def sanitize_for_t5(text: str) -> str:
    """Remove control / injection tokens before sending text to a T5 model."""

    out = text
    for pat in _INJECTION_PATTERNS:
        out = pat.sub("", out)
    return out


class T5Paraphraser:
    """Lazy wrapper around a HuggingFace seq2seq paraphrase model.

    The default model is ``humarin/chatgpt_paraphraser_on_T5_base`` (~250MB,
    CPU-runnable). On first ``load()`` we check the local HF cache so the
    log line tells the user whether they are downloading or warm-loading.
    """

    DEFAULT_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name: str = model_name or self.DEFAULT_MODEL
        self._model = None
        self._tokenizer = None
        self._loaded: bool = False
        self._load_failed: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        """Load model + tokenizer. Returns True on success.

        Idempotent. Once a load fails we do not retry within the same
        process (avoids hammering a broken network on every paragraph).
        """

        if self._loaded:
            return True
        if self._load_failed:
            return False

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:
            logger.warning(
                "transformers not importable (%s); heavy paraphrase disabled", exc
            )
            self._load_failed = True
            return False

        cache_hit = self._is_cached()
        if cache_hit:
            logger.info("loading paraphraser from HF cache: %s", self.model_name)
        else:
            logger.info(
                "downloading paraphraser %s (~250MB, first run only); "
                "subsequent runs use HF cache",
                self.model_name,
            )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.eval()
        except Exception as exc:
            logger.warning(
                "could not load paraphraser %s (%s); falling back to light mode",
                self.model_name,
                exc,
            )
            self._tokenizer = None
            self._model = None
            self._load_failed = True
            return False

        self._loaded = True
        logger.info("heavy paraphrase model ready: %s", self.model_name)
        return True

    def _is_cached(self) -> bool:
        """Best-effort cache probe so we can log download vs warm-load."""

        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(
                repo_id=self.model_name, filename="config.json"
            )
            return bool(cached)
        except Exception:
            return False

    def paraphrase_batch(
        self,
        texts: Sequence[str],
        num_beams: int = 4,
        max_length: int = 512,
    ) -> List[str]:
        """Deterministic batched paraphrase.

        Returns a list parallel to ``texts``. If the model is not loaded
        we return the inputs unchanged so callers do not have to special-
        case the failure path.
        """

        if not texts:
            return []
        if not self.load():
            return list(texts)

        import torch

        prompts = [f"paraphrase: {sanitize_for_t5(t).strip()}" for t in texts]
        inputs = self._tokenizer(  # type: ignore[union-attr]
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False,
                length_penalty=1.0,
                early_stopping=True,
            )
        return [
            self._tokenizer.decode(o, skip_special_tokens=True)  # type: ignore[union-attr]
            for o in outputs
        ]

    def paraphrase_candidates(
        self,
        text: str,
        n: int = 3,
        top_p: float = 0.9,
        max_length: int = 512,
    ) -> List[str]:
        """Stochastic candidate generation for detector-guided iteration.

        ``n=1`` falls back to deterministic beam search (matches the
        existing behavior). For ``n>1`` we use nucleus sampling with
        ``num_return_sequences=n``. Duplicates are de-duplicated while
        preserving order; if dedup leaves fewer than ``n`` strings we
        return what we have rather than fabricating filler.
        """

        if n <= 0:
            return []
        if not self.load():
            return [text]
        if n == 1:
            return self.paraphrase_batch([text], num_beams=4, max_length=max_length)

        import torch

        prompt = f"paraphrase: {sanitize_for_t5(text).strip()}"
        inputs = self._tokenizer(  # type: ignore[union-attr]
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            outputs = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=n,
                num_beams=1,
            )
        decoded = [
            self._tokenizer.decode(o, skip_special_tokens=True)  # type: ignore[union-attr]
            for o in outputs
        ]
        seen: set[str] = set()
        deduped: List[str] = []
        for r in decoded:
            if r not in seen:
                seen.add(r)
                deduped.append(r)
        return deduped


class SemanticGate:
    """Cosine-similarity drift gate for paraphrase outputs.

    Backed by ``sentence-transformers/all-MiniLM-L6-v2`` (~80MB). If the
    sentence-transformers package is not installed (it lives in
    ``requirements-heavy.txt``) the gate becomes a no-op that accepts
    every candidate; we log an info line once so users notice.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        threshold: float = 0.85,
    ) -> None:
        self.model_name: str = model_name or self.DEFAULT_MODEL
        self.threshold: float = float(threshold)
        self._model = None
        self._loaded: bool = False
        self._load_failed: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            logger.info(
                "sentence-transformers not installed (%s); semantic drift gate "
                "is disabled (install requirements-heavy.txt to enable)",
                exc,
            )
            self._load_failed = True
            return False

        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            logger.warning(
                "could not load semantic gate model %s (%s); "
                "all paraphrases will be accepted",
                self.model_name,
                exc,
            )
            self._load_failed = True
            return False

        self._loaded = True
        logger.info("semantic drift gate ready: %s (threshold=%.2f)", self.model_name, self.threshold)
        return True

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity in [-1, 1]. Returns 1.0 if the gate is unavailable."""

        if not self.load():
            return 1.0

        import numpy as np

        embs = self._model.encode(  # type: ignore[union-attr]
            [a, b],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        norm_a = float(np.linalg.norm(embs[0]))
        norm_b = float(np.linalg.norm(embs[1]))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(embs[0], embs[1]) / (norm_a * norm_b))

    def passes(self, original: str, candidate: str) -> bool:
        """True if the candidate is semantically close enough to the original."""

        return self.similarity(original, candidate) >= self.threshold
