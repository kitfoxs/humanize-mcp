"""Style preset loader.

Loads JSON style presets from this directory and merges them with the
default ``base.json`` schema. Presets are referenced by name (for example
``"reddit"`` loads ``reddit.json``).

The merge is shallow on top-level keys, with one exception: ``pass_configs``
is merged one level deep so a preset can override a single pass config
without restating the whole block.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, List

_STYLES_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_json(filename: str) -> Dict[str, Any]:
    path = os.path.join(_STYLES_DIR, filename)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_styles() -> List[str]:
    """Return the names of every available style preset (excluding base)."""
    out: List[str] = []
    for name in os.listdir(_STYLES_DIR):
        if not name.endswith(".json"):
            continue
        if name in {"base.json", "lexical_substitutions.json"}:
            continue
        out.append(os.path.splitext(name)[0])
    return sorted(out)


def load_style(name: str) -> Dict[str, Any]:
    """Load a style by name, merged on top of base.json."""
    base = _load_json("base.json")
    if name in {"base", "default"} or name is None:
        return base

    filename = f"{name}.json"
    path = os.path.join(_STYLES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"unknown style preset '{name}'. available: {list_styles()}"
        )
    overlay = _load_json(filename)
    return _merge(base, overlay)


def _merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if (
            key == "pass_configs"
            and isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged_pass_configs = dict(merged[key])
            for pkey, pval in value.items():
                if pkey in merged_pass_configs and isinstance(merged_pass_configs[pkey], dict) and isinstance(pval, dict):
                    merged_pass_configs[pkey] = {**merged_pass_configs[pkey], **pval}
                else:
                    merged_pass_configs[pkey] = pval
            merged[key] = merged_pass_configs
        elif (
            key == "lexical_substitutions"
            and isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged_subs = dict(merged.get(key, {}))
            merged_subs.update(value)
            merged[key] = merged_subs
        else:
            merged[key] = value
    return merged


__all__ = ["load_style", "list_styles"]
