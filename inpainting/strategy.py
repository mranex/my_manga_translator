from __future__ import annotations

import importlib
from typing import Callable, Iterable, Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


BBox = tuple[int, int, int, int]


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("numpy is required for inpainting strategy helpers")
    return np


def _get_cv2():
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError:
        return None


def _to_mask_array(mask) -> "np.ndarray":
    np_module = _require_numpy()
    current = mask
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    array = np_module.asarray(current)
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("Mask must be 2D")
    return np_module.where(array > 0, 255, 0).astype(np_module.uint8)


def _dilate_binary_mask(mask, dilation: int):
    np_module = _require_numpy()
    binary_mask = _to_mask_array(mask)
    if dilation <= 0:
        return binary_mask

    cv2 = _get_cv2()
    if cv2 is not None:
        kernel = np_module.ones((dilation * 2 + 1, dilation * 2 + 1), dtype=np_module.uint8)
        return cv2.dilate(binary_mask, kernel, iterations=1)

    padded = np_module.pad(binary_mask, dilation, mode="constant", constant_values=0)
    out = np_module.zeros_like(binary_mask)
    for offset_y in range(dilation * 2 + 1):
        for offset_x in range(dilation * 2 + 1):
            out = np_module.maximum(
                out,
                padded[offset_y:offset_y + binary_mask.shape[0], offset_x:offset_x + binary_mask.shape[1]],
            )
    return out.astype(np_module.uint8)


def _erode_binary_mask(mask, erosion: int):
    np_module = _require_numpy()
    binary_mask = _to_mask_array(mask)
    if erosion <= 0:
        return binary_mask

    cv2 = _get_cv2()
    if cv2 is not None:
        kernel = np_module.ones((erosion * 2 + 1, erosion * 2 + 1), dtype=np_module.uint8)
        return cv2.erode(binary_mask, kernel, iterations=1)

    padded = np_module.pad(binary_mask, erosion, mode="constant", constant_values=0)
    out = np_module.full_like(binary_mask, 255)
    for offset_y in range(erosion * 2 + 1):
        for offset_x in range(erosion * 2 + 1):
            out = np_module.minimum(
                out,
                padded[offset_y:offset_y + binary_mask.shape[0], offset_x:offset_x + binary_mask.shape[1]],
            )
    return out.astype(np_module.uint8)


def _clamp_box(box: Sequence[int], image_shape: Sequence[int]) -> BBox:
    height = int(image_shape[0])
    width = int(image_shape[1])
    x1, y1, x2, y2 = [int(value) for value in box]
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def boxes_from_mask(mask) -> list[BBox]:
    np_module = _require_numpy()
    binary_mask = _to_mask_array(mask)
    if binary_mask.size == 0 or not np_module.any(binary_mask):
        return []

    cv2 = _get_cv2()
    if cv2 is not None:
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
        return sorted(boxes, key=lambda box: (box[1], box[0], box[3], box[2]))

    visited = np_module.zeros(binary_mask.shape, dtype=bool)
    height, width = binary_mask.shape
    boxes: list[BBox] = []

    for y in range(height):
        for x in range(width):
            if visited[y, x] or binary_mask[y, x] == 0:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            while stack:
                current_x, current_y = stack.pop()
                min_x = min(min_x, current_x)
                min_y = min(min_y, current_y)
                max_x = max(max_x, current_x)
                max_y = max(max_y, current_y)
                for offset_x, offset_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_x = current_x + offset_x
                    next_y = current_y + offset_y
                    if next_x < 0 or next_y < 0 or next_x >= width or next_y >= height:
                        continue
                    if visited[next_y, next_x] or binary_mask[next_y, next_x] == 0:
                        continue
                    visited[next_y, next_x] = True
                    stack.append((next_x, next_y))
            boxes.append((min_x, min_y, max_x + 1, max_y + 1))

    return sorted(boxes, key=lambda box: (box[1], box[0], box[3], box[2]))


def crop_box(image, mask, box: Sequence[int], margin: int) -> tuple[object, object, BBox]:
    np_module = _require_numpy()
    binary_mask = _to_mask_array(mask)
    crop = _clamp_box(
        (
            int(box[0]) - int(margin),
            int(box[1]) - int(margin),
            int(box[2]) + int(margin),
            int(box[3]) + int(margin),
        ),
        getattr(image, "shape", binary_mask.shape),
    )
    x1, y1, x2, y2 = crop
    return image[y1:y2, x1:x2].copy(), binary_mask[y1:y2, x1:x2].copy(), crop


def pad_to_modulo(image, mask, mod: int, bubble_mask=None):
    np_module = _require_numpy()
    if mod <= 1:
        return image.copy(), _to_mask_array(mask), None if bubble_mask is None else _to_mask_array(bubble_mask), image.shape[:2]

    height, width = image.shape[:2]
    pad_bottom = (mod - (height % mod)) % mod
    pad_right = (mod - (width % mod)) % mod
    if pad_bottom == 0 and pad_right == 0:
        return image.copy(), _to_mask_array(mask), None if bubble_mask is None else _to_mask_array(bubble_mask), (height, width)

    image_mode = "reflect" if height > 1 and width > 1 else "edge"
    padded_image = np_module.pad(
        image,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode=image_mode,
    )
    padded_mask = np_module.pad(
        _to_mask_array(mask),
        ((0, pad_bottom), (0, pad_right)),
        mode="constant",
        constant_values=0,
    )
    padded_bubble = None
    if bubble_mask is not None:
        padded_bubble = np_module.pad(
            _to_mask_array(bubble_mask),
            ((0, pad_bottom), (0, pad_right)),
            mode="constant",
            constant_values=0,
        )
    return padded_image, padded_mask, padded_bubble, (height, width)


def resize_max_side(image, mask, max_side: int, bubble_mask=None):
    np_module = _require_numpy()
    if max_side <= 0:
        raise ValueError("max_side must be positive")

    height, width = image.shape[:2]
    current_max = max(height, width)
    if current_max <= max_side:
        return image.copy(), _to_mask_array(mask), None if bubble_mask is None else _to_mask_array(bubble_mask), 1.0

    scale = float(max_side) / float(current_max)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    cv2 = _get_cv2()
    if cv2 is not None:
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(_to_mask_array(mask), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        resized_bubble = None
        if bubble_mask is not None:
            resized_bubble = cv2.resize(_to_mask_array(bubble_mask), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    else:
        x_indices = np_module.clip(
            np_module.floor(np_module.arange(new_width) / scale).astype(int),
            0,
            width - 1,
        )
        y_indices = np_module.clip(
            np_module.floor(np_module.arange(new_height) / scale).astype(int),
            0,
            height - 1,
        )
        resized_image = image[y_indices][:, x_indices].copy()
        resized_mask = _to_mask_array(mask)[y_indices][:, x_indices].copy()
        resized_bubble = None if bubble_mask is None else _to_mask_array(bubble_mask)[y_indices][:, x_indices].copy()

    resized_mask = np_module.where(resized_mask > 0, 255, 0).astype(np_module.uint8)
    if resized_bubble is not None:
        resized_bubble = np_module.where(resized_bubble > 0, 255, 0).astype(np_module.uint8)
    return resized_image, resized_mask, resized_bubble, scale


def composite_masked(base_image, patch_image, mask_patch, x1: int, y1: int) -> None:
    np_module = _require_numpy()
    binary_mask = _to_mask_array(mask_patch)
    height, width = binary_mask.shape
    roi = base_image[y1:y1 + height, x1:x1 + width]
    mask_pixels = binary_mask > 0
    if roi.ndim == 3:
        roi[mask_pixels] = patch_image[mask_pixels]
    else:
        roi[mask_pixels] = patch_image[mask_pixels]


def clear_masked_region(working_mask, mask_patch, x1: int, y1: int) -> None:
    binary_mask = _to_mask_array(mask_patch)
    height, width = binary_mask.shape
    roi = working_mask[y1:y1 + height, x1:x1 + width]
    roi[binary_mask > 0] = 0


def _boxes_intersect_or_touch(a: BBox, b: BBox) -> bool:
    return not (
        a[2] < b[0]
        or b[2] < a[0]
        or a[3] < b[1]
        or b[3] < a[1]
    )


def _merge_boxes(boxes: Iterable[BBox]) -> list[BBox]:
    merged: list[BBox] = []
    for box in boxes:
        current = tuple(int(value) for value in box)
        merged_any = True
        while merged_any:
            merged_any = False
            next_merged: list[BBox] = []
            for existing in merged:
                if _boxes_intersect_or_touch(existing, current):
                    current = (
                        min(existing[0], current[0]),
                        min(existing[1], current[1]),
                        max(existing[2], current[2]),
                        max(existing[3], current[3]),
                    )
                    merged_any = True
                else:
                    next_merged.append(existing)
            merged = next_merged
        merged.append(current)
    return sorted(merged, key=lambda box: (box[1], box[0], box[3], box[2]))


def crop_windows_from_bboxes(
    boxes: Sequence[Sequence[int]],
    image_shape: Sequence[int],
    ratio: float = 1.7,
    aspect_ratio: float = 1.0,
) -> list[BBox]:
    windows: list[BBox] = []
    for box in boxes or []:
        x1, y1, x2, y2 = [int(value) for value in box]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        center_x = x1 + (width / 2.0)
        center_y = y1 + (height / 2.0)
        base_size = max(width, int(round(height * aspect_ratio)))
        expanded_width = max(width, int(round(base_size * ratio)))
        expanded_height = max(height, int(round((base_size / max(aspect_ratio, 1e-6)) * ratio)))
        window = _clamp_box(
            (
                int(round(center_x - (expanded_width / 2.0))),
                int(round(center_y - (expanded_height / 2.0))),
                int(round(center_x + (expanded_width / 2.0))),
                int(round(center_y + (expanded_height / 2.0))),
            ),
            image_shape,
        )
        if window[2] <= window[0] or window[3] <= window[1]:
            continue
        windows.append(window)

    if not windows:
        return []

    return _merge_boxes(windows)


def apply_bubble_fill_fast_path(
    image_bgr,
    text_mask,
    bubble_mask,
    *,
    min_overlap_ratio: float = 0.65,
    color_sample_erode: int = 3,
    fill_dilate: int = 1,
):
    np_module = _require_numpy()
    binary_text_mask = _to_mask_array(text_mask)
    working_image = image_bgr.copy()
    if bubble_mask is None:
        return working_image, binary_text_mask, {
            "components": 0,
            "filled_components": 0,
            "filled_pixels": 0,
            "remaining_pixels": int(np_module.count_nonzero(binary_text_mask)),
        }

    binary_bubble_mask = _to_mask_array(bubble_mask)
    if not np_module.any(binary_text_mask) or not np_module.any(binary_bubble_mask):
        return working_image, binary_text_mask, {
            "components": 0,
            "filled_components": 0,
            "filled_pixels": 0,
            "remaining_pixels": int(np_module.count_nonzero(binary_text_mask)),
        }

    remaining_mask = binary_text_mask.copy()
    component_boxes = boxes_from_mask(binary_text_mask)
    filled_components = 0
    filled_pixels = 0

    for x1, y1, x2, y2 in component_boxes:
        component_mask = binary_text_mask[y1:y2, x1:x2]
        component_pixels = int(np_module.count_nonzero(component_mask))
        if component_pixels <= 0:
            continue

        component_bubble_mask = binary_bubble_mask[y1:y2, x1:x2]
        overlap_pixels = int(np_module.count_nonzero(
            np_module.logical_and(component_mask > 0, component_bubble_mask > 0)
        ))
        overlap_ratio = overlap_pixels / float(component_pixels)
        if overlap_ratio < float(min_overlap_ratio):
            continue

        sample_margin = max(int(color_sample_erode) * 3, int(fill_dilate) * 3, 6)
        sample_x1 = max(0, x1 - sample_margin)
        sample_y1 = max(0, y1 - sample_margin)
        sample_x2 = min(binary_text_mask.shape[1], x2 + sample_margin)
        sample_y2 = min(binary_text_mask.shape[0], y2 + sample_margin)

        text_window = binary_text_mask[sample_y1:sample_y2, sample_x1:sample_x2]
        bubble_window = binary_bubble_mask[sample_y1:sample_y2, sample_x1:sample_x2]
        image_window = working_image[sample_y1:sample_y2, sample_x1:sample_x2]

        component_window = np_module.zeros_like(text_window, dtype=np_module.uint8)
        component_window[
            y1 - sample_y1:y2 - sample_y1,
            x1 - sample_x1:x2 - sample_x1,
        ] = component_mask

        bubble_sampling_mask = _erode_binary_mask(bubble_window, color_sample_erode)
        if not np_module.any(bubble_sampling_mask):
            bubble_sampling_mask = bubble_window

        sample_ring = _dilate_binary_mask(component_window, max(2, color_sample_erode + fill_dilate))
        sample_mask = np_module.logical_and(
            bubble_sampling_mask > 0,
            text_window == 0,
        )
        sample_mask = np_module.logical_and(sample_mask, sample_ring > 0)
        if not np_module.any(sample_mask):
            sample_mask = np_module.logical_and(
                bubble_sampling_mask > 0,
                text_window == 0,
            )
        if not np_module.any(sample_mask):
            continue

        sample_pixels = image_window[sample_mask]
        if sample_pixels.size == 0:
            continue
        sampled_color = np_module.median(sample_pixels, axis=0)
        fill_mask = _dilate_binary_mask(component_mask, fill_dilate)
        fill_mask = np_module.where(
            np_module.logical_and(fill_mask > 0, component_bubble_mask > 0),
            255,
            0,
        ).astype(np_module.uint8)
        if not np_module.any(fill_mask):
            continue

        fill_pixel_mask = fill_mask > 0
        working_image[y1:y2, x1:x2][fill_pixel_mask] = np_module.clip(sampled_color, 0, 255).astype(np_module.uint8)
        remaining_mask[y1:y2, x1:x2][fill_pixel_mask] = 0
        filled_components += 1
        filled_pixels += int(np_module.count_nonzero(fill_mask))

    return working_image, remaining_mask, {
        "components": len(component_boxes),
        "filled_components": filled_components,
        "filled_pixels": filled_pixels,
        "remaining_pixels": int(np_module.count_nonzero(remaining_mask)),
    }


def _run_forward_padded(
    inpaint_forward: Callable,
    image,
    mask,
    *,
    bubble_mask=None,
    pad_mod: int,
):
    padded_image, padded_mask, padded_bubble, (orig_h, orig_w) = pad_to_modulo(
        image,
        mask,
        pad_mod,
        bubble_mask=bubble_mask,
    )
    result = inpaint_forward(
        padded_image,
        padded_mask,
        padded_bubble,
    )
    return result[:orig_h, :orig_w].copy()


def run_inpaint_resize(
    inpaint_forward: Callable,
    image,
    mask,
    *,
    bubble_mask=None,
    resize_limit: int,
    pad_mod: int,
):
    np_module = _require_numpy()
    resized_image, resized_mask, resized_bubble, _ = resize_max_side(
        image,
        mask,
        resize_limit,
        bubble_mask=bubble_mask,
    )
    resized_result = _run_forward_padded(
        inpaint_forward,
        resized_image,
        resized_mask,
        bubble_mask=resized_bubble,
        pad_mod=pad_mod,
    )

    cv2 = _get_cv2()
    original_height, original_width = image.shape[:2]
    if cv2 is not None:
        restored = cv2.resize(
            resized_result,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        x_indices = np_module.clip(
            np_module.floor(np_module.arange(original_width) * (resized_result.shape[1] / original_width)).astype(int),
            0,
            resized_result.shape[1] - 1,
        )
        y_indices = np_module.clip(
            np_module.floor(np_module.arange(original_height) * (resized_result.shape[0] / original_height)).astype(int),
            0,
            resized_result.shape[0] - 1,
        )
        restored = resized_result[y_indices][:, x_indices].copy()

    output = image.copy()
    binary_mask = _to_mask_array(mask)
    output[binary_mask > 0] = restored[binary_mask > 0]
    return output


def run_inpaint_crop(
    inpaint_forward: Callable,
    image,
    mask,
    *,
    bubble_mask=None,
    crop_trigger_size: int,
    crop_margin: int,
    resize_limit: int,
    pad_mod: int,
    crop_windows=None,
):
    np_module = _require_numpy()
    binary_mask = _to_mask_array(mask)
    if not np_module.any(binary_mask):
        return image.copy()

    output = image.copy()
    working_mask = binary_mask.copy()
    working_bubble_mask = None if bubble_mask is None else _to_mask_array(bubble_mask).copy()

    windows = list(crop_windows or [])
    if not windows:
        windows = boxes_from_mask(binary_mask)

    for window in windows:
        if crop_windows:
            crop_image, crop_mask, crop_bounds = crop_box(output, working_mask, window, max(16, int(crop_margin)))
        else:
            crop_image, crop_mask, crop_bounds = crop_box(output, working_mask, window, crop_margin)
        if crop_mask.size == 0 or not np_module.any(crop_mask):
            continue

        crop_bubble = None
        if working_bubble_mask is not None:
            x1, y1, x2, y2 = crop_bounds
            crop_bubble = working_bubble_mask[y1:y2, x1:x2].copy()

        if max(crop_image.shape[:2]) > resize_limit:
            crop_result = run_inpaint_resize(
                inpaint_forward,
                crop_image,
                crop_mask,
                bubble_mask=crop_bubble,
                resize_limit=resize_limit,
                pad_mod=pad_mod,
            )
        else:
            crop_result = _run_forward_padded(
                inpaint_forward,
                crop_image,
                crop_mask,
                bubble_mask=crop_bubble,
                pad_mod=pad_mod,
            )

        x1, y1, _, _ = crop_bounds
        composite_masked(output, crop_result, crop_mask, x1, y1)
        clear_masked_region(working_mask, crop_mask, x1, y1)

    residual_boxes = boxes_from_mask(working_mask)
    for box in residual_boxes:
        crop_image, crop_mask, crop_bounds = crop_box(output, working_mask, box, crop_margin)
        if crop_mask.size == 0 or not np_module.any(crop_mask):
            continue
        crop_bubble = None
        if working_bubble_mask is not None:
            x1, y1, x2, y2 = crop_bounds
            crop_bubble = working_bubble_mask[y1:y2, x1:x2].copy()
        crop_result = _run_forward_padded(
            inpaint_forward,
            crop_image,
            crop_mask,
            bubble_mask=crop_bubble,
            pad_mod=pad_mod,
        )
        x1, y1, _, _ = crop_bounds
        composite_masked(output, crop_result, crop_mask, x1, y1)
        clear_masked_region(working_mask, crop_mask, x1, y1)

    return output


__all__ = [
    "apply_bubble_fill_fast_path",
    "boxes_from_mask",
    "clear_masked_region",
    "composite_masked",
    "crop_box",
    "crop_windows_from_bboxes",
    "pad_to_modulo",
    "resize_max_side",
    "run_inpaint_crop",
    "run_inpaint_resize",
]
