"""Helpers for building OCR preparation items from cached detection results."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

TEXT_LIKE_LAYOUT_LABEL_PARTS = ("text", "title", "caption", "content")


def build_ocr_items_from_detection(
    detection_data: dict[str, Any],
    image_shape: Sequence[int],
    *,
    logger: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Build conservative OCR items from cached detection JSON.

    This intentionally keeps the logic simple for the milestone:
    - one bubble OCR item per detected bubble using text-like layout regions as
      the bubble OCR source
    - unmatched text regions become outside-text OCR items
    - text-like layout regions act as conservative fallbacks when they do not
      strongly overlap existing OCR candidates

    Future milestones can reuse more of the old render-item heuristics here
    without changing the file-backed OCR cache contract.
    """

    bubbles = [bubble for bubble in detection_data.get("bubbles", []) if not bool(bubble.get("excluded", False))]
    text_regions = [
        text_region
        for text_region in detection_data.get("text_regions", [])
        if not bool(text_region.get("excluded", False))
    ]
    layout_regions = [
        layout_region
        for layout_region in detection_data.get("layout_regions", [])
        if not bool(layout_region.get("excluded", False))
    ]

    bubble_items: list[dict[str, Any]] = []
    outside_items: list[dict[str, Any]] = []
    existing_non_bubble_boxes: list[tuple[int, int, int, int]] = []
    page_name = Path(str(detection_data.get("source_image", "") or "page")).name or "page"

    for bubble_index, bubble in enumerate(bubbles):
        bubble_id = _coerce_int(bubble.get("id"), bubble_index)
        bubble_bbox = clamp_bbox_to_image(bubble.get("bbox"), image_shape)
        if bubble_bbox is None:
            if logger is not None:
                logger(
                    "[ocr_items] Skipping invalid bubble candidate "
                    f"on {page_name}: bbox={bubble.get('bbox')!r}, "
                    f"source={bubble.get('detector') or bubble.get('source') or 'detector'}"
                )
            continue

        matched_text_regions = [
            text_region
            for text_region in text_regions
            if text_region.get("bubble_id") == bubble_id
        ]
        filtered_text_regions: list[dict[str, Any]] = []
        for text_region in matched_text_regions:
            text_bbox = clamp_bbox_to_image(text_region.get("bbox"), image_shape)
            if text_bbox is None:
                continue
            if text_region_belongs_to_bubble(text_bbox, bubble_bbox):
                filtered_text_regions.append(text_region)
                continue
            if logger is not None:
                logger(
                    f"[ocr_items] Rejected text region {_coerce_int(text_region.get('id'), -1)} "
                    f"for bubble {bubble_id}: outside bubble bbox."
                )
        matched_text_regions = filtered_text_regions
        matched_layout_regions = _layout_regions_for_bubble(layout_regions, bubble_bbox, image_shape)

        refined_bbox = union_bboxes(
            (
                clamp_bbox_to_image(layout_region.get("bbox"), image_shape)
                for layout_region in matched_layout_regions
            ),
            image_shape,
        )
        if refined_bbox is not None:
            ocr_bbox = intersect_bboxes(refined_bbox, bubble_bbox, image_shape)
            if ocr_bbox is None:
                ocr_bbox = bubble_bbox
            elif ocr_bbox != refined_bbox and logger is not None:
                logger(f"[ocr_items] Clipped bubble OCR bbox to bubble bbox for bubble {bubble_id}.")
        else:
            ocr_bbox = bubble_bbox
        reading_order = _min_reading_order(matched_layout_regions) or _min_reading_order(matched_text_regions)
        source_direction = _first_non_empty(
            [layout_region.get("source_direction") for layout_region in matched_layout_regions]
        ) or _first_non_empty(
            [text_region.get("source_direction") for text_region in matched_text_regions]
        ) or infer_source_direction(ocr_bbox)

        detector_sources = unique_strings(
            [
                bubble.get("detector"),
                *[layout_region.get("detector") for layout_region in matched_layout_regions],
                *[text_region.get("detector") for text_region in matched_text_regions],
            ]
        )

        bubble_items.append(
            {
                "id": bubble_index,
                "kind": "bubble",
                "bbox": bbox_to_list(bubble_bbox),
                "ocr_bbox": bbox_to_list(ocr_bbox),
                "crop_path": None,
                "bubble_id": bubble_id,
                "reading_order": reading_order,
                "detector_sources": detector_sources,
                "source_direction": source_direction,
                "layout_region_ids": _region_ids(matched_layout_regions),
                "text_region_ids": _region_ids(matched_text_regions),
                "ocr_bbox_source": "layout_region",
                "ocr_bbox_source_color": "purple",
                "text": "",
                "status": "prepared",
            }
        )

    ordered_unmatched_text_regions = sorted(
        [
            text_region
            for text_region in text_regions
            if text_region.get("bubble_id") is None
        ],
        key=lambda entry: sort_key_for_region(entry.get("bbox"), entry.get("reading_order")),
    )

    for region_index, text_region in enumerate(ordered_unmatched_text_regions):
        region_bbox = clamp_bbox_to_image(text_region.get("bbox"), image_shape)
        if region_bbox is None:
            if logger is not None:
                logger(
                    "[ocr_items] Skipping invalid text-region candidate "
                    f"on {page_name}: bbox={text_region.get('bbox')!r}, "
                    f"source={text_region.get('detector') or text_region.get('source') or 'detector'}"
                )
            continue

        ocr_bbox = expand_bbox(region_bbox, image_shape, 16)
        existing_non_bubble_boxes.append(ocr_bbox)
        outside_items.append(
            {
                "id": region_index,
                "kind": "outside_text",
                "bbox": bbox_to_list(region_bbox),
                "ocr_bbox": bbox_to_list(ocr_bbox),
                "crop_path": None,
                "bubble_id": None,
                "reading_order": _coerce_optional_int(text_region.get("reading_order")),
                "detector_sources": unique_strings([text_region.get("detector")]),
                "source_direction": text_region.get("source_direction") or infer_source_direction(ocr_bbox),
                "text": "",
                "status": "prepared",
            }
        )

    fallback_layout_items: list[dict[str, Any]] = []
    for layout_region in sorted(
        layout_regions,
        key=lambda entry: sort_key_for_region(entry.get("bbox"), entry.get("reading_order")),
    ):
        label = str(layout_region.get("label") or "").strip().lower()
        if not is_text_like_layout_label(label):
            continue

        layout_bbox = clamp_bbox_to_image(layout_region.get("bbox"), image_shape)
        if layout_bbox is None:
            if logger is not None:
                logger(
                    "[ocr_items] Skipping invalid layout candidate "
                    f"on {page_name}: bbox={layout_region.get('bbox')!r}, "
                    f"source={layout_region.get('detector') or layout_region.get('source') or 'detector'}"
                )
            continue

        if is_huge_bbox(layout_bbox, image_shape, max_region_ratio=0.35):
            continue

        if overlap_ratio_against_many(layout_bbox, (bubble.get("bbox") for bubble in bubbles), image_shape) >= 0.35:
            continue

        if overlap_ratio_against_many(layout_bbox, existing_non_bubble_boxes, image_shape) >= 0.60:
            continue

        ocr_bbox = expand_bbox(layout_bbox, image_shape, 16)
        existing_non_bubble_boxes.append(ocr_bbox)
        fallback_layout_items.append(
            {
                "id": len(fallback_layout_items),
                "kind": "layout_text",
                "bbox": bbox_to_list(layout_bbox),
                "ocr_bbox": bbox_to_list(ocr_bbox),
                "crop_path": None,
                "bubble_id": None,
                "reading_order": _coerce_optional_int(layout_region.get("reading_order")),
                "detector_sources": unique_strings([layout_region.get("detector"), "pp_doclayout_v3"]),
                "source_direction": infer_source_direction(ocr_bbox),
                "text": "",
                "status": "prepared",
            }
        )

    all_items = bubble_items + outside_items + fallback_layout_items
    all_items = sorted(
        all_items,
        key=lambda entry: sort_key_for_region(entry.get("ocr_bbox") or entry.get("bbox"), entry.get("reading_order")),
    )

    for item_id, item in enumerate(all_items):
        item["id"] = item_id

    if logger is not None:
        logger(
            "Prepared OCR items from detection cache: "
            f"{len(bubble_items)} bubble, "
            f"{len(outside_items)} outside_text, "
            f"{len(fallback_layout_items)} layout_text"
        )

    return all_items


def build_ocr_items_from_canon_state(
    detection_json_or_state: dict[str, Any],
    image_shape: Sequence[int],
    *,
    logger: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Build OCR items from active canon_state workflow items."""

    from .canon_state import get_active_canon_items

    canon_items = get_active_canon_items(detection_json_or_state)
    ordered_items = sorted(
        canon_items,
        key=lambda entry: sort_key_for_region(
            entry.get("ocr_bbox") or entry.get("bbox"),
            entry.get("reading_order"),
        ),
    )

    ocr_items: list[dict[str, Any]] = []
    counts_by_kind: dict[str, int] = {}
    for item_id, canon_item in enumerate(ordered_items):
        bbox = clamp_bbox_to_image(canon_item.get("bbox"), image_shape)
        ocr_bbox = clamp_bbox_to_image(canon_item.get("ocr_bbox") or canon_item.get("bbox"), image_shape)
        if bbox is None and ocr_bbox is None:
            continue
        if bbox is None:
            bbox = ocr_bbox
        if ocr_bbox is None:
            ocr_bbox = bbox
        if bbox is None or ocr_bbox is None:
            continue

        kind = str(canon_item.get("kind", "") or "")
        counts_by_kind[kind] = counts_by_kind.get(kind, 0) + 1
        detector_refs = canon_item.get("detector_refs", {})
        metadata = canon_item.get("metadata", {})
        detector_sources = metadata.get("detector_sources", []) if isinstance(metadata, dict) else []
        if not isinstance(detector_sources, list):
            detector_sources = []

        ocr_items.append(
            {
                "id": item_id,
                "canon_id": str(canon_item.get("canon_id", "") or ""),
                "kind": kind,
                "bbox": bbox_to_list(bbox),
                "ocr_bbox": bbox_to_list(ocr_bbox),
                "crop_path": None,
                "bubble_id": detector_refs.get("bubble_id") if isinstance(detector_refs, dict) else None,
                "reading_order": canon_item.get("reading_order"),
                "detector_sources": [str(value) for value in detector_sources if str(value).strip()],
                "source_direction": str(
                    canon_item.get("source_direction", "") or infer_source_direction(ocr_bbox)
                ),
                "text": "",
                "status": "prepared",
                # Snapshot only. canon_state remains the source of truth.
                "excluded": False,
            }
        )

    if logger is not None:
        counts_text = ", ".join(
            f"{count} {kind}" for kind, count in sorted(counts_by_kind.items())
        ) or "0 items"
        logger(f"Prepared OCR items from canon_state: {counts_text}")

    return ocr_items


def clamp_bbox_to_image(
    bbox: Any,
    image_shape: Sequence[int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    height = int(image_shape[0])
    width = int(image_shape[1])
    x1, y1, x2, y2 = [int(value) for value in bbox[:4]]

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def expand_bbox(
    bbox: tuple[int, int, int, int] | None,
    image_shape: Sequence[int],
    padding: int,
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None

    expanded = (
        int(bbox[0]) - int(padding),
        int(bbox[1]) - int(padding),
        int(bbox[2]) + int(padding),
        int(bbox[3]) + int(padding),
    )
    return clamp_bbox_to_image(expanded, image_shape)


def union_bboxes(
    boxes: Iterable[tuple[int, int, int, int] | None],
    image_shape: Sequence[int],
) -> tuple[int, int, int, int] | None:
    normalized_boxes = [box for box in boxes if box is not None]
    if not normalized_boxes:
        return None

    merged_bbox = (
        min(box[0] for box in normalized_boxes),
        min(box[1] for box in normalized_boxes),
        max(box[2] for box in normalized_boxes),
        max(box[3] for box in normalized_boxes),
    )
    return clamp_bbox_to_image(merged_bbox, image_shape)


def intersect_bboxes(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
    image_shape: Sequence[int],
) -> tuple[int, int, int, int] | None:
    if a is None or b is None:
        return None

    intersection = (
        max(int(a[0]), int(b[0])),
        max(int(a[1]), int(b[1])),
        min(int(a[2]), int(b[2])),
        min(int(a[3]), int(b[3])),
    )
    return clamp_bbox_to_image(intersection, image_shape)


def bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def bbox_intersection_area(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
) -> int:
    if a is None or b is None:
        return 0

    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def overlap_ratio_against_many(
    bbox: tuple[int, int, int, int],
    candidates: Iterable[Any],
    image_shape: Sequence[int],
) -> float:
    target_area = max(bbox_area(bbox), 1)
    best_ratio = 0.0

    for candidate in candidates:
        candidate_bbox = clamp_bbox_to_image(candidate, image_shape)
        if candidate_bbox is None:
            continue
        best_ratio = max(
            best_ratio,
            bbox_intersection_area(bbox, candidate_bbox) / target_area,
        )

    return best_ratio


def is_text_like_layout_label(label: str) -> bool:
    normalized = str(label or "").strip().lower()
    return any(part in normalized for part in TEXT_LIKE_LAYOUT_LABEL_PARTS)


def is_huge_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: Sequence[int],
    *,
    max_region_ratio: float,
) -> bool:
    page_area = max(1, int(image_shape[0]) * int(image_shape[1]))
    return bbox_area(bbox) > int(page_area * float(max_region_ratio))


def infer_source_direction(bbox: tuple[int, int, int, int]) -> str:
    width = max(1, int(bbox[2]) - int(bbox[0]))
    height = max(1, int(bbox[3]) - int(bbox[1]))
    return "vertical" if height >= (width * 1.15) else "horizontal"


def text_region_belongs_to_bubble(
    text_bbox: tuple[int, int, int, int],
    bubble_bbox: tuple[int, int, int, int],
) -> bool:
    return region_belongs_to_bubble(text_bbox, bubble_bbox)


def region_belongs_to_bubble(
    region_bbox: tuple[int, int, int, int],
    bubble_bbox: tuple[int, int, int, int],
) -> bool:
    center_x = (float(region_bbox[0]) + float(region_bbox[2])) / 2.0
    center_y = (float(region_bbox[1]) + float(region_bbox[3])) / 2.0
    if (
        float(bubble_bbox[0]) <= center_x <= float(bubble_bbox[2])
        and float(bubble_bbox[1]) <= center_y <= float(bubble_bbox[3])
    ):
        return True

    region_area = max(1, bbox_area(region_bbox))
    return (bbox_intersection_area(region_bbox, bubble_bbox) / float(region_area)) >= 0.50


def _layout_regions_for_bubble(
    layout_regions: Sequence[dict[str, Any]],
    bubble_bbox: tuple[int, int, int, int],
    image_shape: Sequence[int],
) -> list[dict[str, Any]]:
    matched_regions: list[dict[str, Any]] = []
    for layout_region in layout_regions:
        label = str(layout_region.get("label") or "").strip().lower()
        if not is_text_like_layout_label(label):
            continue
        layout_bbox = clamp_bbox_to_image(layout_region.get("bbox"), image_shape)
        if layout_bbox is None:
            continue
        if not region_belongs_to_bubble(layout_bbox, bubble_bbox):
            continue
        matched_regions.append(layout_region)
    return matched_regions


def _region_ids(regions: Sequence[dict[str, Any]]) -> list[int]:
    region_ids: list[int] = []
    for region in regions:
        region_id = _coerce_optional_int(region.get("id"))
        if region_id is None:
            continue
        region_ids.append(region_id)
    return region_ids


def crop_image_to_bbox(image: Any, bbox: Sequence[int | float]) -> Any:
    x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
    return image[y1:y2, x1:x2]


def bbox_to_list(bbox: Sequence[int | float] | None) -> list[int] | None:
    if bbox is None:
        return None
    return [int(value) for value in bbox[:4]]


def sort_key_for_region(
    bbox: Any,
    reading_order: Any,
) -> tuple[int, int, int]:
    normalized_bbox = bbox if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else (0, 0, 0, 0)
    order = _coerce_optional_int(reading_order)
    return (
        order if order is not None else 10**9,
        int(normalized_bbox[1]),
        int(normalized_bbox[0]),
    )


def unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)

    return ordered


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _min_reading_order(regions: Iterable[dict[str, Any]]) -> int | None:
    orders = [_coerce_optional_int(region.get("reading_order")) for region in regions]
    valid_orders = [order for order in orders if order is not None]
    return min(valid_orders) if valid_orders else None


def _first_non_empty(values: Iterable[Any]) -> str | None:
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


__all__ = [
    "bbox_area",
    "bbox_intersection_area",
    "bbox_to_list",
    "build_ocr_items_from_canon_state",
    "build_ocr_items_from_detection",
    "clamp_bbox_to_image",
    "crop_image_to_bbox",
    "expand_bbox",
    "infer_source_direction",
    "intersect_bboxes",
    "is_text_like_layout_label",
    "overlap_ratio_against_many",
    "region_belongs_to_bubble",
    "sort_key_for_region",
    "text_region_belongs_to_bubble",
    "union_bboxes",
    "unique_strings",
]
