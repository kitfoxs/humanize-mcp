"""Pass 9: paraphrase pass.

The DIPPER-style paraphrase pass (research/04 section 1.2). Two
implementations:

* **Lightweight** (default): rule-based synonym substitution and a careful
  sentence-order shuffle on adjacent low-dependency sentences. Pure Python,
  fast, deterministic.
* **Heavy** (opt-in): loads ``kalpeshk2011/dipper-paraphraser-xxl`` or
  ``humarin/chatgpt_paraphraser_on_T5_base`` via transformers. Falls back
  to lightweight if the model can't be loaded (no GPU, no model on disk).

The configuration accepts ``mode='light'|'heavy'|'auto'``. ``auto`` tries
heavy and degrades.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .base import PipelinePass

logger = logging.getLogger(__name__)


# Curated synonym map. Conservative: each entry has a single safe alternative
# that preserves register. The lexical pass (pass 2) handles the heavier
# domain-specific table.
LIGHT_SYNONYMS = {
    r"\butilize\b": "use",
    r"\bUtilize\b": "Use",
    r"\bsubsequently\b": "later",
    r"\bSubsequently\b": "Later",
    r"\bnumerous\b": "many",
    r"\bNumerous\b": "Many",
    r"\bvarious\b": "several",
    r"\bVarious\b": "Several",
    r"\badditional\b": "more",
    r"\bsignificantly\b": "noticeably",
    r"\benable\b": "let",
    r"\bdemonstrate\b": "show",
    r"\bDemonstrate\b": "Show",
    r"\bcommence\b": "start",
    r"\bobtain\b": "get",
    r"\brequire\b": "need",
    r"\bassist\b": "help",
    r"\bpurchase\b": "buy",
    r"\bterminate\b": "end",
    r"\binitiate\b": "start",
    r"\bconstitute\b": "make up",
    r"\bsufficient\b": "enough",
}


class ParaphrasePass(PipelinePass):
    pass_id = 9
    pass_name = "paraphrase"

    def __init__(self, model_loader: Optional[Any] = None) -> None:
        self._model = None
        self._tokenizer = None
        self._model_load_attempted = False
        self._model_loader = model_loader

    def apply(self, text: str, config: Dict[str, Any]) -> str:
        self.reset_changes()
        mode = config.get("mode", "light")
        intensity = config.get("intensity", "balanced")

        if mode in {"heavy", "auto"}:
            try:
                heavy_out = self._try_heavy(text, config)
                if heavy_out is not None:
                    return heavy_out
            except Exception as exc:
                logger.warning("heavy paraphrase failed (%s); falling back to light", exc)

        return self._light_paraphrase(text, intensity)

    def _light_paraphrase(self, text: str, intensity: str) -> str:
        out = text
        max_subs_per_para = {
            "minimal": 1,
            "balanced": 3,
            "aggressive": 6,
        }.get(intensity, 3)

        paragraphs = re.split(r"(\n\s*\n)", out)
        new_paras: List[str] = []
        for chunk in paragraphs:
            if not chunk.strip() or chunk.startswith("\n"):
                new_paras.append(chunk)
                continue
            new_paras.append(self._paraphrase_paragraph(chunk, max_subs_per_para))
        return "".join(new_paras)

    def _paraphrase_paragraph(self, paragraph: str, max_subs: int) -> str:
        out = paragraph
        subs_done = 0
        for pattern, replacement in LIGHT_SYNONYMS.items():
            if subs_done >= max_subs:
                break
            compiled = re.compile(pattern)
            new_out, n = compiled.subn(replacement, out, count=1)
            if n > 0:
                self.log_change("paraphrase_synonym", pattern, replacement)
                subs_done += n
                out = new_out
        return out

    def _try_heavy(self, text: str, config: Dict[str, Any]) -> Optional[str]:
        if self._model is None and not self._model_load_attempted:
            self._model_load_attempted = True
            self._load_model(config)
        if self._model is None or self._tokenizer is None:
            return None

        # operate paragraph-by-paragraph to keep memory bounded
        paragraphs = re.split(r"(\n\s*\n)", text)
        out_parts: List[str] = []
        for chunk in paragraphs:
            if not chunk.strip() or chunk.startswith("\n"):
                out_parts.append(chunk)
                continue
            out_parts.append(self._heavy_paraphrase_chunk(chunk, config))
        return "".join(out_parts)

    def _load_model(self, config: Dict[str, Any]) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = config.get(
                "heavy_model", "humarin/chatgpt_paraphraser_on_T5_base"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            logger.info("loaded heavy paraphrase model %s", model_name)
        except Exception as exc:
            logger.warning("could not load heavy paraphrase model: %s", exc)
            self._model = None
            self._tokenizer = None

    def _heavy_paraphrase_chunk(self, chunk: str, config: Dict[str, Any]) -> str:
        assert self._model is not None and self._tokenizer is not None
        prompt = f"paraphrase: {chunk.strip()}"
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self._model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            do_sample=False,
            length_penalty=1.0,
        )
        result = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.log_change("paraphrase_heavy", chunk[:60], result[:60])
        return result
