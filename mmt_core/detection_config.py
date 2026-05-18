"""Detection engine configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_MANGA_MODEL_ID = "ogkalu/comic-text-and-bubble-detector"
DETECTION_ENGINE_CLASSIC = "classic"
DETECTION_ENGINE_MANGA_RTDETR = "manga_rtdetr"
VALID_DETECTION_ENGINES = {
    DETECTION_ENGINE_CLASSIC,
    DETECTION_ENGINE_MANGA_RTDETR,
}


def _normalize_engine(value: Any) -> str:
    normalized = str(value or DETECTION_ENGINE_MANGA_RTDETR).strip().lower()
    if normalized not in VALID_DETECTION_ENGINES:
        return DETECTION_ENGINE_MANGA_RTDETR
    return normalized


def _normalize_label_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = []

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        label = str(candidate or "").strip()
        if not label:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(label)
    return tuple(normalized)


def _normalize_device(value: Any) -> str:
    normalized = str(value or "auto").strip()
    return normalized or "auto"


def _normalize_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except Exception:
        threshold = 0.35
    return max(0.01, min(0.99, threshold))


def _normalized_signature_from_value(value: Any) -> tuple[str, str, str, float, tuple[str, ...], tuple[str, ...]]:
    return DetectionConfig.from_value(value).signature()


@dataclass(slots=True)
class DetectionConfig:
    engine: str = DETECTION_ENGINE_MANGA_RTDETR
    manga_model_id: str = DEFAULT_MANGA_MODEL_ID
    manga_device: str = "auto"
    manga_confidence_threshold: float = 0.35
    manga_text_labels: tuple[str, ...] = ()
    manga_bubble_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.engine = _normalize_engine(self.engine)
        self.manga_model_id = str(self.manga_model_id or DEFAULT_MANGA_MODEL_ID).strip() or DEFAULT_MANGA_MODEL_ID
        self.manga_device = _normalize_device(self.manga_device)
        self.manga_confidence_threshold = _normalize_threshold(self.manga_confidence_threshold)
        self.manga_text_labels = _normalize_label_tuple(self.manga_text_labels)
        self.manga_bubble_labels = _normalize_label_tuple(self.manga_bubble_labels)

    @classmethod
    def from_value(cls, value: Any) -> "DetectionConfig":
        if isinstance(value, cls):
            return cls(
                engine=value.engine,
                manga_model_id=value.manga_model_id,
                manga_device=value.manga_device,
                manga_confidence_threshold=value.manga_confidence_threshold,
                manga_text_labels=value.manga_text_labels,
                manga_bubble_labels=value.manga_bubble_labels,
            )
        if isinstance(value, dict):
            return cls(
                engine=value.get("engine", DETECTION_ENGINE_MANGA_RTDETR),
                manga_model_id=value.get("manga_model_id", DEFAULT_MANGA_MODEL_ID),
                manga_device=value.get("manga_device", "auto"),
                manga_confidence_threshold=value.get("manga_confidence_threshold", 0.35),
                manga_text_labels=value.get("manga_text_labels", ()),
                manga_bubble_labels=value.get("manga_bubble_labels", ()),
            )
        return cls()

    def to_settings_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "manga_model_id": self.manga_model_id,
            "manga_device": self.manga_device,
            "manga_confidence_threshold": self.manga_confidence_threshold,
            "manga_text_labels": list(self.manga_text_labels),
            "manga_bubble_labels": list(self.manga_bubble_labels),
        }

    def signature(self) -> tuple[str, str, str, float, tuple[str, ...], tuple[str, ...]]:
        return (
            self.engine,
            self.manga_model_id,
            self.manga_device,
            round(float(self.manga_confidence_threshold), 6),
            tuple(label.casefold() for label in self.manga_text_labels),
            tuple(label.casefold() for label in self.manga_bubble_labels),
        )


def detection_configs_match(current: Any, cached: Any) -> bool:
    if not isinstance(cached, dict):
        return False
    return _normalized_signature_from_value(current) == _normalized_signature_from_value(cached)


__all__ = [
    "DEFAULT_MANGA_MODEL_ID",
    "DETECTION_ENGINE_CLASSIC",
    "DETECTION_ENGINE_MANGA_RTDETR",
    "DetectionConfig",
    "detection_configs_match",
    "VALID_DETECTION_ENGINES",
]
