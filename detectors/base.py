from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


BBox = tuple[int, int, int, int]
LegacyBubbleDetection = list[int | float | None]


def _coerce_bbox_to_int(bbox: Sequence[object]) -> BBox:
    if len(bbox) != 4:
        raise ValueError("Region bbox must contain exactly 4 values")

    x1, y1, x2, y2 = bbox
    return (int(x1), int(y1), int(x2), int(y2))


@dataclass(slots=True)
class Region:
    kind: str
    bbox: BBox
    score: float = 1.0
    class_id: int | None = None
    mask: Any | None = None
    detector: str = ""

    def __post_init__(self) -> None:
        self.bbox = _coerce_bbox_to_int(self.bbox)
        self.score = float(self.score)
        self.detector = str(self.detector or "")
        if self.class_id is not None:
            self.class_id = int(self.class_id)


@dataclass(slots=True)
class BubbleRegion(Region):
    kind: str = field(default="bubble", init=False)
    is_dark: bool = False
    fill_color: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        Region.__post_init__(self)
        self.is_dark = bool(self.is_dark)


@dataclass(slots=True)
class LayoutRegion(Region):
    kind: str = field(default="layout", init=False)
    label: str = ""
    label_id: int | None = None
    reading_order: int | None = None
    polygon_points: Any | None = None

    def __post_init__(self) -> None:
        Region.__post_init__(self)
        self.label = str(self.label)
        if self.label_id is not None:
            self.label_id = int(self.label_id)
        if self.reading_order is not None:
            self.reading_order = int(self.reading_order)


@dataclass(slots=True)
class PageDetectionResult:
    bubbles: list[BubbleRegion] = field(default_factory=list)
    layout_regions: list[LayoutRegion] = field(default_factory=list)
    method: str = "legacy_yolo"
    stats: dict[str, Any] = field(default_factory=dict)

    def to_legacy_detections(self) -> list[LegacyBubbleDetection]:
        legacy_detections: list[LegacyBubbleDetection] = []
        for bubble in self.bubbles:
            x1, y1, x2, y2 = bubble.bbox
            legacy_detections.append(
                [x1, y1, x2, y2, bubble.score, bubble.class_id, int(bubble.is_dark)]
            )
        return legacy_detections


def bubble_region_from_legacy_detection(
    detection: Sequence[object],
) -> BubbleRegion:
    if len(detection) < 6:
        raise ValueError("Legacy bubble detection must contain at least 6 values")

    class_id = detection[5]
    is_dark = bool(int(detection[6])) if len(detection) >= 7 else False

    return BubbleRegion(
        bbox=_coerce_bbox_to_int(detection[:4]),
        score=float(detection[4]),
        class_id=None if class_id is None else int(class_id),
        is_dark=is_dark,
    )


__all__ = [
    "BubbleRegion",
    "LayoutRegion",
    "LegacyBubbleDetection",
    "PageDetectionResult",
    "Region",
    "bubble_region_from_legacy_detection",
]
