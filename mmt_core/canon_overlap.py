"""Resolve duplicate detection-region workflow candidates before canon activation.

This resolver operates on detection-region geometry only. It intentionally uses
``bbox`` and never ``ocr_bbox`` because OCR crop boxes may include padding,
unioned text coverage, or other downstream context that must not drive
duplicate-region suppression.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

from .ocr_items import bbox_area, clamp_bbox_to_image

OVERLAP_COMPONENT_CONTAINMENT_THRESHOLD = 0.10


def resolve_largest_overlap_components(
    items: list[dict[str, Any]],
    image_shape: Sequence[int],
    *,
    logger: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Keep the largest detection-region bbox in each same-group component."""

    resolved_items: list[dict[str, Any]] = [deepcopy(item) for item in items if isinstance(item, dict)]
    grouped_candidate_indices: dict[str, list[int]] = {}
    indexed_boxes: dict[int, tuple[int, int, int, int]] = {}

    for workflow_index, item in enumerate(resolved_items):
        item["workflow_index"] = workflow_index
        compare_box = _compare_box(item, image_shape)
        if compare_box is None or _is_manual_item(item):
            continue
        overlap_group_kind = _overlap_group_kind(item)
        if overlap_group_kind is None:
            continue
        indexed_boxes[workflow_index] = compare_box
        grouped_candidate_indices.setdefault(overlap_group_kind, []).append(workflow_index)

    grouped_components: list[list[int]] = []
    candidate_count = 0
    for candidate_indices in grouped_candidate_indices.values():
        candidate_count += len(candidate_indices)
        grouped_components.extend(
            component
            for component in _connected_components(candidate_indices, indexed_boxes, resolved_items)
            if len(component) > 1
        )

    suppressed_count = 0

    for component_index, component in enumerate(grouped_components, start=1):
        overlap_group_id = f"overlap_{component_index:04d}"
        winner_index = _winner_for_component(component, indexed_boxes)
        winner_item = resolved_items[winner_index]
        winner_metadata = _ensure_metadata_dict(winner_item)
        winner_item["suppressed"] = False
        winner_item["excluded"] = False
        winner_item["overlap_group_id"] = overlap_group_id
        winner_metadata["suppressed"] = False
        winner_metadata["overlap_group_id"] = overlap_group_id

        for workflow_index in component:
            item = resolved_items[workflow_index]
            metadata = _ensure_metadata_dict(item)
            item["overlap_group_id"] = overlap_group_id
            metadata["overlap_group_id"] = overlap_group_id
            if workflow_index == winner_index:
                metadata.pop("suppression_reason", None)
                metadata.pop("suppressed_by_workflow_index", None)
                metadata.pop("suppressed_by", None)
                item.pop("suppression_reason", None)
                item.pop("suppressed_by_workflow_index", None)
                item.pop("suppressed_by", None)
                continue

            item["suppressed"] = True
            item["enabled"] = False
            item["excluded"] = True
            item["suppression_reason"] = "overlap_component_smaller_rect"
            item["suppressed_by_workflow_index"] = winner_index
            metadata["suppressed"] = True
            metadata["suppression_reason"] = "overlap_component_smaller_rect"
            metadata["suppressed_by_workflow_index"] = winner_index
            suppressed_count += 1
            if logger is not None:
                logger(
                    "[canon_overlap] Suppressed workflow item "
                    f"{workflow_index} in {overlap_group_id} because winner {winner_index} has larger area."
                )

    if logger is not None:
        logger(
            "[canon_overlap] Resolved "
            f"{len(grouped_components)} overlap components across {candidate_count} workflow items; "
            f"suppressed {suppressed_count} smaller items."
        )

    return resolved_items


def _compare_box(item: dict[str, Any], image_shape: Sequence[int]) -> tuple[int, int, int, int] | None:
    return clamp_bbox_to_image(item.get("bbox"), image_shape)


def _bbox_area(box: tuple[int, int, int, int] | None) -> int:
    return max(1, bbox_area(box))


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    overlap_x = min(int(a[2]), int(b[2])) - max(int(a[0]), int(b[0]))
    overlap_y = min(int(a[3]), int(b[3])) - max(int(a[1]), int(b[1]))
    if overlap_x <= 0 or overlap_y <= 0:
        return 0
    return overlap_x * overlap_y


def _center_inside(
    smaller: tuple[int, int, int, int],
    larger: tuple[int, int, int, int],
) -> bool:
    center_x = (float(smaller[0]) + float(smaller[2])) / 2.0
    center_y = (float(smaller[1]) + float(smaller[3])) / 2.0
    return (
        float(larger[0]) <= center_x <= float(larger[2])
        and float(larger[1]) <= center_y <= float(larger[3])
    )


def _should_connect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    intersection_area = _intersection_area(a, b)
    if intersection_area > 0:
        containment_ratio = intersection_area / float(min(_bbox_area(a), _bbox_area(b)))
        if containment_ratio >= OVERLAP_COMPONENT_CONTAINMENT_THRESHOLD:
            return True

    if _bbox_area(a) <= _bbox_area(b):
        return _center_inside(a, b)
    return _center_inside(b, a)


def _items_can_overlap_merge(
    *,
    left_item: dict[str, Any],
    right_item: dict[str, Any],
) -> bool:
    left_ids = _layout_region_id_set(left_item)
    right_ids = _layout_region_id_set(right_item)
    if left_ids and right_ids:
        return not left_ids.isdisjoint(right_ids)
    return True


def _layout_region_id_set(item: dict[str, Any]) -> set[int]:
    detector_refs = item.get("detector_refs", {})
    if not isinstance(detector_refs, dict):
        return set()
    layout_region_ids = detector_refs.get("layout_region_ids", [])
    if not isinstance(layout_region_ids, list):
        return set()
    resolved: set[int] = set()
    for value in layout_region_ids:
        try:
            resolved.add(int(value))
        except Exception:
            continue
    return resolved


def _connected_components(
    candidate_indices: Sequence[int],
    indexed_boxes: dict[int, tuple[int, int, int, int]],
    resolved_items: Sequence[dict[str, Any]],
) -> list[list[int]]:
    adjacency: dict[int, set[int]] = {index: set() for index in candidate_indices}
    for left_offset, left_index in enumerate(candidate_indices):
        left_box = indexed_boxes[left_index]
        for right_index in candidate_indices[left_offset + 1 :]:
            if not _items_can_overlap_merge(
                left_item=resolved_items[left_index],
                right_item=resolved_items[right_index],
            ):
                continue
            right_box = indexed_boxes[right_index]
            if not _should_connect(left_box, right_box):
                continue
            adjacency[left_index].add(right_index)
            adjacency[right_index].add(left_index)

    components: list[list[int]] = []
    seen: set[int] = set()
    for start_index in candidate_indices:
        if start_index in seen:
            continue
        stack = [start_index]
        component: list[int] = []
        seen.add(start_index)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency.get(current, ()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def _winner_for_component(
    component: Sequence[int],
    indexed_boxes: dict[int, tuple[int, int, int, int]],
) -> int:
    return sorted(
        component,
        key=lambda workflow_index: (
            -_bbox_area(indexed_boxes[workflow_index]),
            workflow_index,
        ),
    )[0]


def _overlap_group_kind(item: dict[str, Any]) -> str | None:
    kind = str(item.get("kind", "") or "").strip().lower()
    if kind == "bubble":
        return "bubble"
    if kind == "layout_text":
        return "layout_text"
    return None


def _is_manual_item(item: dict[str, Any]) -> bool:
    if bool(item.get("manual", False)):
        return True
    if bool(item.get("bbox_user_edited", False)) or bool(item.get("ocr_bbox_user_edited", False)):
        return True
    if bool(item.get("render_bbox_user_edited", False)):
        return True
    if str(item.get("source", "") or "").strip().lower() == "manual":
        return True
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        if bool(metadata.get("manual", False)):
            return True
        if bool(metadata.get("user_edited", False)) or bool(metadata.get("user_edited_bbox", False)):
            return True
    return False


def _ensure_metadata_dict(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    item["metadata"] = dict(metadata)
    return item["metadata"]


__all__ = [
    "OVERLAP_COMPONENT_CONTAINMENT_THRESHOLD",
    "resolve_largest_overlap_components",
]
