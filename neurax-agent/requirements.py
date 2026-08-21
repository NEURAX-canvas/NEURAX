"""Extract the hard constraints a client states in their request.

A prompt like *"a model for the phone, images and text, under 1 MB"* carries a
budget the design must actually meet. Parsing is deliberately deterministic
rather than delegated to the model: a constraint that decides pass/fail should
not depend on sampling, and a missed constraint is worse than no constraint at
all because the design then silently ignores it.

Anything not recognised here is left to the planner as free-form intent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Optional

# Size units expressed in bytes.
_SIZE_UNITS = {
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kb": 1024,
    "kib": 1024,
    "ko": 1024,
    "mb": 1024 ** 2,
    "mib": 1024 ** 2,
    "mo": 1024 ** 2,
    # Spoken forms: people write "1 mega" and "2 giga" as often as the symbols.
    "mega": 1024 ** 2,
    "megabyte": 1024 ** 2,
    "megabytes": 1024 ** 2,
    "megaoctet": 1024 ** 2,
    "megaoctets": 1024 ** 2,
    "gb": 1024 ** 3,
    "gib": 1024 ** 3,
    "go": 1024 ** 3,
    "giga": 1024 ** 3,
    "gigabyte": 1024 ** 3,
    "gigabytes": 1024 ** 3,
    "kilo": 1024,
    "kilobyte": 1024,
    "kilobytes": 1024,
    "kilooctet": 1024,
    "kilooctets": 1024,
}

_TIME_UNITS_MS = {"ms": 1.0, "millisecond": 1.0, "milliseconds": 1.0, "s": 1000.0,
                  "sec": 1000.0, "second": 1000.0, "seconds": 1000.0}

# Words that mark an upper bound, in English and French.
#
# "runs on 3GB of RAM" states a ceiling exactly as much as "under 3GB" does —
# stating what the model must fit *inside* is a normal, common way to phrase a
# resource limit, not a synonym gap to leave unhandled. Verified against a real
# miss: "qui va tourner sur 3giga de ram" (a real client request, not a
# constructed test case) matched nothing before these were added, and the
# constraint was silently dropped rather than reaching the planner at all.
_UNDER = r"(?:under|below|less\s+than|at\s+most|max(?:imum)?|no\s+more\s+than|within|" \
         r"(?:runs?|running|able\s+to\s+run|capable\s+of\s+running|fit(?:s|ting)?)\s+(?:on|in|within)|" \
         r"moins\s+de|au\s+plus|inf[ée]rieur\s+[àa]|sous|maximum|" \
         r"(?:tourner|fonctionner|capable\s+de\s+(?:tourner|fonctionner)|tenir)\s+(?:sur|dans|avec)|" \
         r"tient\s+dans)"

_NUMBER = r"(\d+(?:[.,]\d+)?)"


def _to_float(raw: str) -> float:
    return float(raw.replace(",", "."))


@dataclass
class DeploymentBudget:
    """Hard limits the finished design has to satisfy."""

    max_size_bytes: Optional[float] = None
    max_latency_ms: Optional[float] = None
    max_vram_bytes: Optional[float] = None
    max_parameters: Optional[float] = None
    #: Free-form deployment target ("phone", "browser", ...), when stated.
    target_device: Optional[str] = None

    def is_empty(self) -> bool:
        return all(
            value is None
            for value in (
                self.max_size_bytes,
                self.max_latency_ms,
                self.max_vram_bytes,
                self.max_parameters,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def describe(self) -> str:
        """One line naming every constraint, for the run log and the UI."""
        parts = []
        if self.max_size_bytes is not None:
            parts.append(f"size ≤ {self.max_size_bytes / 1024 ** 2:.3g} MB")
        if self.max_parameters is not None:
            parts.append(f"parameters ≤ {self.max_parameters:,.0f}")
        if self.max_latency_ms is not None:
            parts.append(f"latency ≤ {self.max_latency_ms:g} ms")
        if self.max_vram_bytes is not None:
            parts.append(f"VRAM ≤ {self.max_vram_bytes / 1024 ** 3:.3g} GB")
        if self.target_device:
            parts.append(f"target: {self.target_device}")
        return ", ".join(parts) if parts else "no explicit budget"


def _find_size(text: str) -> Optional[float]:
    units = "|".join(sorted(_SIZE_UNITS, key=len, reverse=True))
    # "under 1 MB", "moins de 500 ko", "< 2mb"
    for pattern in (
        rf"{_UNDER}\s*(?:de\s*)?{_NUMBER}\s*({units})\b",
        rf"[<≤]\s*{_NUMBER}\s*({units})\b",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _to_float(match.group(1)) * _SIZE_UNITS[match.group(2).lower()]
    return None


def _find_latency(text: str) -> Optional[float]:
    units = "|".join(sorted(_TIME_UNITS_MS, key=len, reverse=True))
    for pattern in (
        rf"{_UNDER}\s*(?:de\s*)?{_NUMBER}\s*({units})\b",
        rf"[<≤]\s*{_NUMBER}\s*({units})\b",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _to_float(match.group(1)) * _TIME_UNITS_MS[match.group(2).lower()]
    return None


def _find_parameters(text: str) -> Optional[float]:
    scale = {"k": 1e3, "m": 1e6, "b": 1e9, "bn": 1e9}
    pattern = rf"{_UNDER}\s*(?:de\s*)?{_NUMBER}\s*([kmb]|bn)?\s*(?:param\w*|param[èe]tres?)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        factor = scale.get((match.group(2) or "").lower(), 1.0)
        return _to_float(match.group(1)) * factor
    return None


#: Devices whose mention implies a deployment context worth recording.
_DEVICE_HINTS = {
    "phone": ("phone", "mobile", "smartphone", "t[ée]l[ée]phone", "android", "ios"),
    "browser": ("browser", "navigateur", "webassembly", "wasm", "in-browser"),
    "edge": ("edge device", "embedded", "microcontroller", "raspberry", "embarqu[ée]"),
    "laptop": ("laptop", "portable", "cpu-only", "cpu only"),
}


def _find_device(text: str) -> Optional[str]:
    for device, hints in _DEVICE_HINTS.items():
        for hint in hints:
            if re.search(rf"\b{hint}\b", text, re.IGNORECASE):
                return device
    return None


def extract_budget(user_message: str, hw_config: dict[str, Any] | None = None) -> DeploymentBudget:
    """Read the budget out of the request, and out of the panel when it is set.

    Values stated in the request win: the panel holds defaults, the sentence
    holds the requirement for this particular design.
    """
    text = user_message or ""
    budget = DeploymentBudget(
        max_size_bytes=_find_size(text),
        max_latency_ms=_find_latency(text),
        max_parameters=_find_parameters(text),
        target_device=_find_device(text),
    )

    if hw_config:
        # A configured GPU bounds VRAM even when the sentence says nothing.
        gpu_memory_gb = hw_config.get("gpuMemoryGb") or hw_config.get("gpu_memory_gb")
        if budget.max_vram_bytes is None and gpu_memory_gb:
            try:
                budget.max_vram_bytes = float(gpu_memory_gb) * 1024 ** 3
            except (TypeError, ValueError):
                pass

    return budget
