"""Pass 8: register shift.

Final pass before paraphrase. Ensures the output sits in the right register
band for the active style:

* casual targets: heavy contractions, colloquialisms, fragments OK.
* neutral targets: balanced.
* academic targets: keep formality but break parallel structures and add
  measured hedges ("the data appear to suggest", "broadly speaking").
* formal targets: minimal contractions, no slang.

The register shift is mostly handled by the per-style configs that earlier
passes consumed; this pass handles a few targeted adjustments that need to
happen at the end.

Reference: research/02 D.1, research/03 section 3.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .base import PipelinePass

ACADEMIC_HEDGE_INSERTIONS = [
    ("This shows", "The data appear to show"),
    ("This proves", "The evidence suggests"),
    ("This demonstrates", "This appears to demonstrate"),
    ("clearly indicates", "indicates"),
    ("obviously", ""),
    ("undoubtedly", "arguably"),
    ("definitely", "in this case"),
]

CASUAL_BOOSTERS = [
    (r"\bvery\b", "really"),
    (r"\bvery\b", "super"),
    (r"\bsignificant\b", "big"),
    (r"\butilize\b", "use"),
    (r"\bUtilize\b", "Use"),
    (r"\bindividuals\b", "people"),
    (r"\bIndividuals\b", "People"),
]

FORMAL_DECONTRACTIONS = [
    (r"\bdon't\b", "do not"),
    (r"\bDon't\b", "Do not"),
    (r"\bcan't\b", "cannot"),
    (r"\bCan't\b", "Cannot"),
    (r"\bwon't\b", "will not"),
    (r"\bWon't\b", "Will not"),
]


class RegisterShiftPass(PipelinePass):
    pass_id = 8
    pass_name = "register_shift"

    def apply(self, text: str, config: Dict[str, Any]) -> str:
        self.reset_changes()
        target_register = config.get("target_register", "neutral")

        if target_register == "academic":
            return self._shift_academic(text)
        if target_register == "formal":
            return self._shift_formal(text)
        if target_register == "casual":
            return self._shift_casual(text)
        return text

    def _shift_academic(self, text: str) -> str:
        out = text
        for original, replacement in ACADEMIC_HEDGE_INSERTIONS:
            if original in out:
                if replacement:
                    new = out.replace(original, replacement, 1)
                else:
                    # remove the certainty word + following space
                    new = re.sub(rf"\b{re.escape(original)}\s*", "", out, count=1)
                if new != out:
                    self.log_change("academic_hedge", original, replacement)
                out = new
        return out

    def _shift_casual(self, text: str) -> str:
        out = text
        # first decontraction set is irrelevant for casual; instead use the boosters
        digest = hash(text) & 0xFF
        toggle = {"i": digest}
        for pattern, replacement in CASUAL_BOOSTERS:
            compiled = re.compile(pattern)
            count = 0

            def _sub(m: re.Match[str]) -> str:
                nonlocal count
                if count > 0:
                    return m.group(0)
                count += 1
                self.log_change("casual_booster", m.group(0), replacement)
                return replacement

            out = compiled.sub(_sub, out)
        return out

    def _shift_formal(self, text: str) -> str:
        out = text
        for pattern, replacement in FORMAL_DECONTRACTIONS:
            compiled = re.compile(pattern)

            def _sub(m: re.Match[str]) -> str:
                self.log_change("formal_decontraction", m.group(0), replacement)
                return replacement

            out = compiled.sub(_sub, out)
        return out
