from flask import Flask, render_template, request, redirect, send_file, jsonify
from flask_socketio import SocketIO, emit
import io
import zipfile
import json
import warnings
import os
import sys

# Needed for old YOLO/Ultralytics .pt checkpoints on PyTorch >= 2.6.
# Only safe if you trust the checkpoint files used by this app.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from detectors import (
    TextRegion,
    detect_page_regions_layout_first,
)
from detectors.runtime_utils import (
    bubble_region_to_crop_data,
    clamp_bbox_to_image,
    crop_bbox,
    expand_bbox,
    text_region_to_crop_data,
    union_text_regions_bbox,
)
from detectors.pp_doclayout_v3 import layout_regions_to_text_regions
from inpainting import (
    LamaMangaInpainter,
    build_bubble_mask,
    build_text_block_crop_windows,
    build_text_block_removal_mask,
    collect_item_inpaint_bboxes,
)
from translator.translator import MangaTranslator
from translator.context_memory import ContextMemory
from ocr.chrome_lens_ocr import ChromeLensOCR
from PIL import Image
import numpy as np
import base64
import cv2
from render_item_utils import (
    build_outside_text_blocks,
    consolidate_render_items,
    dedupe_ocr_items_by_text_and_geometry,
)
from text_rendering import choose_render_style_for_item, render_manga_text_block


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "secret_key")

# Initialize SocketIO with auto-detected async mode
def get_async_mode():
    if getattr(sys, 'frozen', False):
        return 'threading'
    try:
        import eventlet
        return 'eventlet'
    except ImportError:
        pass
    try:
        import gevent
        return 'gevent'
    except ImportError:
        pass
    return 'threading'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode=get_async_mode())

# Control verbose logging (set VERBOSE_LOG=1 to enable debug output)
VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "0") == "1"

def log(msg):
    """Print only if verbose logging is enabled."""
    if VERBOSE_LOG:
        print(msg)

# Default max height for split (1.5x width = landscape-ish ratio)
DEFAULT_SPLIT_HEIGHT_RATIO = 2.0

# Global cache for OCR instances
_OCR_CACHE = {
    "chrome_lens": None,
    "paddleocr_vl": None,
}
_INPAINTER_CACHE = {
    "lama_manga": None,
}
MAX_INPAINT_REGION_RATIO = 0.35
INPAINT_BLOCK_PADDING = 14
RENDER_BLOCK_PADDING = 4

def split_text_regions_by_bubble(text_regions):
    text_regions_by_bubble = {}
    outside_text_regions = []
    for text_region in text_regions:
        if text_region.bubble_id is None:
            outside_text_regions.append(text_region)
            continue
        text_regions_by_bubble.setdefault(text_region.bubble_id, []).append(text_region)
    return text_regions_by_bubble, outside_text_regions


def get_first_layout_first_bubble_crop(image):
    page_detection_result = detect_page_regions_layout_first(image)
    if not page_detection_result.bubbles:
        return None

    crop_data = bubble_region_to_crop_data(
        image,
        page_detection_result.bubbles[0],
        (),
    )
    return crop_data["bubble_crop"]


def get_lama_manga_inpainter():
    if _INPAINTER_CACHE["lama_manga"] is None:
        _INPAINTER_CACHE["lama_manga"] = LamaMangaInpainter()
    return _INPAINTER_CACHE["lama_manga"]


def get_paddleocr_vl_ocr(source_language=None):
    if (
        _OCR_CACHE["paddleocr_vl"] is None
        or getattr(_OCR_CACHE["paddleocr_vl"], "configured_source_language", None) != source_language
    ):
        from ocr.paddleocr_vl_ocr import PaddleOCRVLOCR

        _OCR_CACHE["paddleocr_vl"] = PaddleOCRVLOCR(
            source_language=source_language,
        )
    return _OCR_CACHE["paddleocr_vl"]


def bbox_area(bbox):
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return max(0, x2 - x1) * max(0, y2 - y1)


def is_huge_inpaint_bbox(bbox, image_shape, max_region_ratio=MAX_INPAINT_REGION_RATIO):
    if bbox is None:
        return False
    page_area = max(1, int(image_shape[0]) * int(image_shape[1]))
    return bbox_area(clamp_bbox_to_image(bbox, image_shape)) > (page_area * float(max_region_ratio))


def _select_text_block_bbox(text_regions, image_shape):
    if not text_regions:
        return None, False

    candidate_bbox = union_text_regions_bbox(text_regions, image_shape, padding=0)
    if candidate_bbox is None:
        return None, False
    if not is_huge_inpaint_bbox(candidate_bbox, image_shape):
        return candidate_bbox, False

    smaller_regions = [
        region
        for region in text_regions
        if not is_huge_inpaint_bbox(region.bbox, image_shape)
    ]
    if smaller_regions:
        smaller_bbox = union_text_regions_bbox(smaller_regions, image_shape, padding=0)
        if smaller_bbox is not None and not is_huge_inpaint_bbox(smaller_bbox, image_shape):
            return smaller_bbox, True
    return None, True


def _select_item_inpaint_bbox(
    image_shape,
    *,
    primary_bbox=None,
    fallback_bbox=None,
    block_padding=INPAINT_BLOCK_PADDING,
):
    huge_region_skipped = False
    if primary_bbox is not None:
        padded_primary = expand_bbox(primary_bbox, image_shape, block_padding)
        if not is_huge_inpaint_bbox(padded_primary, image_shape):
            return padded_primary, False, False
        huge_region_skipped = True

    if fallback_bbox is not None:
        clamped_fallback = expand_bbox(
            clamp_bbox_to_image(fallback_bbox, image_shape),
            image_shape,
            max(4, int(block_padding) // 2),
        )
        if not is_huge_inpaint_bbox(clamped_fallback, image_shape):
            return clamped_fallback, huge_region_skipped, True

    return None, huge_region_skipped, True


def conservative_inner_text_bbox(
    bbox,
    image_shape,
    width_ratio=0.86,
    height_ratio=0.70,
    source_hints=None,
):
    x1, y1, x2, y2 = [int(value) for value in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    center_x = x1 + (width / 2.0)
    center_y = y1 + (height / 2.0)
    inner_width = max(12, int(round(width * width_ratio)))
    inner_height = max(12, int(round(height * height_ratio)))
    inner_bbox = expand_bbox(
        (
            int(round(center_x - (inner_width / 2.0))),
            int(round(center_y - (inner_height / 2.0))),
            int(round(center_x + (inner_width / 2.0))),
            int(round(center_y + (inner_height / 2.0))),
        ),
        image_shape,
        0,
    )
    if not source_hints:
        return inner_bbox

    bubble_area = max(1, bbox_area((x1, y1, x2, y2)))
    hint_bboxes = [inner_bbox]
    for hint_bbox in source_hints:
        if hint_bbox is None:
            continue
        clamped_hint = clamp_bbox_to_image(hint_bbox, image_shape)
        if bbox_area(clamped_hint) >= int(bubble_area * 0.92):
            continue
        hint_bboxes.append(clamped_hint)

    merged_x1 = min(box[0] for box in hint_bboxes)
    merged_y1 = min(box[1] for box in hint_bboxes)
    merged_x2 = max(box[2] for box in hint_bboxes)
    merged_y2 = max(box[3] for box in hint_bboxes)
    return clamp_bbox_to_image((merged_x1, merged_y1, merged_x2, merged_y2), image_shape)


def gather_render_item_text_regions(render_items):
    text_regions = []
    for render_item in render_items:
        text_regions.extend(render_item.get("text_regions") or [])
        if (
            render_item.get("kind") == "outside_text"
            and render_item.get("text_region") is not None
            and not render_item.get("text_regions")
        ):
            text_regions.append(render_item["text_region"])
    return text_regions


def render_item_sort_key(render_item, original_index):
    reading_order = render_item.get("reading_order")
    render_bbox = render_item.get("render_bbox") or render_item.get("coords") or (0, 0, 0, 0)
    return (
        reading_order if reading_order is not None else 10**9,
        int(render_bbox[1]),
        int(render_bbox[0]),
        original_index,
    )


def finalize_page_translation(image, render_items, translated_texts, font_path):
    if not render_items:
        return image.copy()

    page_image = image.copy()
    for render_item, translated_text in zip(render_items, translated_texts):
        render_item["translated_text"] = translated_text
        render_item["resolved_inpaint_bboxes"] = collect_item_inpaint_bboxes(
            render_item,
            page_image.shape,
        )

    text_mask = build_text_block_removal_mask(page_image.shape, render_items)
    bubble_mask = build_bubble_mask(page_image.shape, render_items)
    crop_windows = build_text_block_crop_windows(render_items, page_image.shape)
    text_regions = gather_render_item_text_regions(render_items)
    if VERBOSE_LOG:
        page_pixels = max(1, int(page_image.shape[0]) * int(page_image.shape[1]))
        masked_pixels = int(np.count_nonzero(text_mask))
        bubble_mask_pixels = int(np.count_nonzero(bubble_mask))
        block_inpaint_boxes = sum(1 for item in render_items if item.get("inpaint_bbox") is not None)
        fallback_boxes = sum(1 for item in render_items if item.get("inpaint_fallback_used"))
        huge_regions_skipped = sum(1 for item in render_items if item.get("huge_region_skipped"))
        matched_items = sum(1 for item in render_items if item.get("text_regions"))
        unmatched_items = sum(1 for item in render_items if not item.get("text_regions"))
        log(
            "LaMa page prep: "
            f"{len(render_items)} render items, "
            f"{block_inpaint_boxes} block inpaint boxes, "
            f"{masked_pixels}/{page_pixels} masked pixels, "
            f"{bubble_mask_pixels}/{page_pixels} bubble-mask pixels, "
            f"{len(crop_windows)} crop windows, "
            f"{fallback_boxes} fallback inner-bbox masks, "
            f"{huge_regions_skipped} huge regions skipped, "
            f"{matched_items} matched items, "
            f"{unmatched_items} no-matched-text items"
        )
        for item in render_items:
            log(
                "  item "
                f"{item.get('kind')} "
                f"coords={item.get('coords')} "
                f"inpaint_boxes={len(item.get('resolved_inpaint_bboxes') or [])} "
                f"fallback={bool(item.get('inpaint_fallback_used'))} "
                f"matched_text={len(item.get('text_regions') or [])}"
            )

    try:
        final_image = get_lama_manga_inpainter().inpaint(
            page_image,
            text_mask,
            bubble_mask=bubble_mask,
            text_regions=text_regions,
            crop_windows=crop_windows,
        )
    except Exception as exc:
        raise RuntimeError(f"LaMa Manga inpainting failed: {exc}") from exc
    background_reference = final_image.copy()

    for render_item in render_items:
        translated_text = render_item.get("translated_text", "")
        if not translated_text or not translated_text.strip():
            continue

        render_bbox = render_item["render_bbox"]
        render_block = choose_render_style_for_item(
            background_reference,
            render_item,
            render_bbox,
            font_path,
            text=translated_text,
        )

        render_manga_text_block(
            final_image,
            render_block,
        )

    return final_image


def build_bubble_render_item(image, bubble_region, matched_text_regions):
    crop_data = bubble_region_to_crop_data(
        image,
        bubble_region,
        matched_text_regions,
    )
    bubble_bbox = crop_data["bubble_bbox"]
    ocr_crop = crop_data["ocr_crop"].copy()
    text_block_bbox, huge_region_skipped = _select_text_block_bbox(
        matched_text_regions,
        image.shape,
    )
    fallback_inner_bbox = conservative_inner_text_bbox(
        bubble_bbox,
        image.shape,
        source_hints=[crop_data.get("ocr_bbox")],
    )
    if text_block_bbox is not None:
        render_bbox = expand_bbox(text_block_bbox, image.shape, RENDER_BLOCK_PADDING)
    else:
        render_bbox = fallback_inner_bbox
    inpaint_bbox, huge_region_skipped, inpaint_fallback_used = _select_item_inpaint_bbox(
        image.shape,
        primary_bbox=text_block_bbox,
        fallback_bbox=fallback_inner_bbox,
    )
    reading_orders = [
        region.reading_order
        for region in matched_text_regions
        if region.reading_order is not None
    ]
    reading_order = min(reading_orders) if reading_orders else None
    fallback_reason = None
    inpaint_bboxes = []
    if inpaint_bbox is not None:
        inpaint_bboxes.append(inpaint_bbox)
    inpaint_bboxes.append(render_bbox)
    if text_block_bbox is not None:
        inpaint_bboxes.append(text_block_bbox)
        inpaint_bboxes.append(crop_data["ocr_bbox"])
    else:
        fallback_reason = "no_matched_text_regions"
        inpaint_bboxes.append(fallback_inner_bbox)
        if crop_data["ocr_bbox"] != bubble_bbox:
            inpaint_bboxes.append(crop_data["ocr_bbox"])
    if inpaint_fallback_used:
        log(
            "No text block found for bubble; using fallback bubble inner bbox "
            f"for inpaint at {bubble_bbox}"
        )

    return {
        "kind": "bubble",
        "bubble_region": bubble_region,
        "text_regions": list(matched_text_regions),
        "coords": bubble_bbox,
        "render_bbox": render_bbox,
        "inpaint_bbox": inpaint_bbox,
        "inpaint_bboxes": inpaint_bboxes,
        "ocr_bbox": crop_data["ocr_bbox"],
        "ocr_crop": ocr_crop,
        "fallback_text_bbox": fallback_inner_bbox,
        "reading_order": reading_order,
        "inpaint_fallback_used": inpaint_fallback_used,
        "fallback_reason": fallback_reason,
        "huge_region_skipped": huge_region_skipped,
    }


def build_outside_text_render_item(image, outside_block, padding=6):
    if isinstance(outside_block, TextRegion):
        base_region = outside_block
        text_regions = [outside_block]
        raw_bbox = clamp_bbox_to_image(outside_block.bbox, image.shape)
        ocr_bbox = raw_bbox
        outside_source = outside_block.detector or "outside_text"
        reading_order = outside_block.reading_order
        source_direction = None
    else:
        base_region = outside_block.get("text_region")
        text_regions = list(outside_block.get("text_regions") or ([base_region] if base_region is not None else []))
        raw_bbox = clamp_bbox_to_image(
            outside_block.get("container_bbox") or outside_block.get("render_bbox") or base_region.bbox,
            image.shape,
        )
        ocr_bbox = clamp_bbox_to_image(outside_block.get("ocr_bbox") or raw_bbox, image.shape)
        outside_source = outside_block.get("outside_source", "outside_text")
        reading_order = outside_block.get("reading_order", getattr(base_region, "reading_order", None))
        source_direction = outside_block.get("source_direction")

    ocr_crop = crop_bbox(image, ocr_bbox).copy()
    huge_region_skipped = is_huge_inpaint_bbox(raw_bbox, image.shape)
    render_padding = 0 if huge_region_skipped else 2
    render_bbox = expand_bbox(raw_bbox, image.shape, render_padding)
    inpaint_bbox, huge_region_skipped, inpaint_fallback_used = _select_item_inpaint_bbox(
        image.shape,
        primary_bbox=raw_bbox,
        fallback_bbox=None,
    )
    if huge_region_skipped and inpaint_bbox is None:
        log(
            "Skipping huge outside-text inpaint bbox "
            f"at {raw_bbox}; will only use any available region mask refinement"
        )

    return {
        "kind": "outside_text",
        "outside_source": outside_source,
        "text_region": base_region,
        "text_regions": text_regions,
        "coords": raw_bbox,
        "container_bbox": raw_bbox,
        "render_bbox": render_bbox,
        "inpaint_bbox": inpaint_bbox,
        "inpaint_bboxes": [bbox for bbox in (inpaint_bbox, render_bbox, ocr_bbox, raw_bbox) if bbox is not None],
        "ocr_bbox": ocr_bbox,
        "ocr_crop": ocr_crop,
        "reading_order": reading_order,
        "source_direction": source_direction,
        "inpaint_fallback_used": inpaint_fallback_used,
        "huge_region_skipped": huge_region_skipped,
    }


def collect_page_render_items(image, page_detection_result):
    text_regions_by_bubble, _ = split_text_regions_by_bubble(
        page_detection_result.text_regions
    )
    pp_text_regions = layout_regions_to_text_regions(
        page_detection_result.layout_regions,
        image.shape,
    )
    comic_text_regions = [
        text_region
        for text_region in page_detection_result.text_regions
        if text_region.detector == "comic_text_detector"
    ]

    render_items = []
    for bubble_idx, bubble_region in enumerate(page_detection_result.bubbles):
        render_items.append(
            build_bubble_render_item(
                image,
                bubble_region,
                text_regions_by_bubble.get(bubble_idx, []),
            )
        )

    outside_blocks, outside_stats = build_outside_text_blocks(
        pp_text_regions,
        comic_text_regions,
        page_detection_result.bubbles,
        image.shape,
        logger=log if VERBOSE_LOG else None,
    )

    outside_render_items = [
        build_outside_text_render_item(image, outside_block)
        for outside_block in outside_blocks
    ]
    render_items.extend(outside_render_items)

    render_items = consolidate_render_items(
        render_items,
        image.shape,
        logger=log if VERBOSE_LOG else None,
    )

    for render_item in render_items:
        ocr_bbox = (
            render_item.get("ocr_bbox")
            or render_item.get("render_bbox")
            or render_item.get("coords")
        )
        if ocr_bbox is None:
            continue
        ocr_bbox = clamp_bbox_to_image(ocr_bbox, image.shape)
        render_item["ocr_bbox"] = ocr_bbox
        ocr_crop = crop_bbox(image, ocr_bbox)
        render_item["ocr_crop"] = ocr_crop.copy() if hasattr(ocr_crop, "copy") else ocr_crop

    render_items = [
        item
        for index, item in sorted(
            enumerate(render_items),
            key=lambda entry: render_item_sort_key(entry[1], entry[0]),
        )
    ]
    if getattr(page_detection_result, "stats", None) is not None:
        page_detection_result.stats["bubble_items"] = len(page_detection_result.bubbles)
        page_detection_result.stats["pp_outside_blocks"] = outside_stats["pp_outside_blocks"]
        page_detection_result.stats["comic_fallback_outside_blocks"] = outside_stats["comic_fallback_outside_blocks"]
        page_detection_result.stats["outside_text_items"] = sum(
            1 for item in render_items if item.get("kind") == "outside_text"
        )
        page_detection_result.stats["final_render_items"] = len(render_items)

    outside_render_items = [
        item
        for item in render_items
        if item.get("kind") == "outside_text"
    ]

    return render_items, text_regions_by_bubble, outside_render_items

def split_long_image(image: np.ndarray, max_height_ratio: float = DEFAULT_SPLIT_HEIGHT_RATIO) -> list:
    """
    Split a long image into multiple shorter chunks.
    
    Args:
        image: Input image as numpy array (H, W, C)
        max_height_ratio: Maximum height/width ratio before splitting.
                          Images taller than width * ratio will be split.
                          
    Returns:
        List of image chunks (numpy arrays). If image doesn't need splitting,
        returns a list with just the original image.
    """
    height, width = image.shape[:2]
    max_height = int(width * max_height_ratio)
    
    # If image is not too tall, return as-is
    if height <= max_height:
        return [image]
    
    # Split into chunks
    chunks = []
    current_y = 0
    chunk_num = 0
    
    while current_y < height:
        # Calculate chunk end position
        chunk_end = min(current_y + max_height, height)
        
        # Extract chunk
        chunk = image[current_y:chunk_end, :].copy()
        chunks.append(chunk)
        
        current_y = chunk_end
        chunk_num += 1
    
    print(f"  Split image ({width}x{height}) into {len(chunks)} chunks")
    return chunks


@app.route("/")
def home():
    return render_template("index.html")


def process_single_image(image, manga_translator, mocr, selected_translator, selected_font, font_analyzer=None):
    """Process a single image and return the translated version.
    
    Optimized with batch translation for Gemini to reduce API calls.
    Supports auto font matching when font_analyzer is provided and selected_font is 'auto'.
    """
    page_detection_result = detect_page_regions_layout_first(image)
    render_items, _, _ = collect_page_render_items(image, page_detection_result)

    if not render_items:
        return image
    
    # Phase 1: Collect all render item data and OCR texts
    texts_to_translate = []

    for render_item in render_items:
        im = Image.fromarray(render_item["ocr_crop"])
        text = mocr(im)
        texts_to_translate.append(text)

    render_items, texts_to_translate = dedupe_ocr_items_by_text_and_geometry(
        render_items,
        texts_to_translate,
        logger=log if VERBOSE_LOG else None,
    )
    if not render_items:
        return image
    
    # Phase 2: Batch translate
    if selected_translator == "gemini" and len(texts_to_translate) > 1:
        # Use batch translation for Gemini
        try:
            if manga_translator._gemini_translator is None:
                from translator.gemini_translator import GeminiTranslator
                api_key = getattr(manga_translator, '_gemini_api_key', None)
                if not api_key:
                    raise ValueError("Gemini API key not provided")
                custom_prompt = getattr(manga_translator, '_gemini_custom_prompt', None)
                manga_translator._gemini_translator = GeminiTranslator(
                    api_key=api_key, 
                    custom_prompt=custom_prompt
                )
            
            translated_texts = manga_translator._gemini_translator.translate_batch(
                texts_to_translate,
                source=manga_translator.source,
                target=manga_translator.target
            )
        except Exception as e:
            print(f"Batch translation failed, falling back to single: {e}")
            translated_texts = [manga_translator.translate(t, method=selected_translator) for t in texts_to_translate]
    
    elif selected_translator == "copilot" and len(texts_to_translate) > 1:
        # Use batch translation for Local LLM (Ollama, LM Studio, etc.)
        try:
            if not hasattr(manga_translator, '_local_llm_translator') or manga_translator._local_llm_translator is None:
                from translator.local_llm_translator import LocalLLMTranslator
                copilot_server = getattr(manga_translator, '_copilot_server', 'http://localhost:8080')
                copilot_model = getattr(manga_translator, '_copilot_model', 'gpt-4o')
                copilot_custom_prompt = getattr(manga_translator, '_copilot_custom_prompt', None)
                manga_translator._local_llm_translator = LocalLLMTranslator(
                    server_url=copilot_server,
                    model=copilot_model,
                    custom_prompt=copilot_custom_prompt
                )
                print(f"Local LLM translator initialized: {copilot_server} / {copilot_model}")
            
            translated_texts = manga_translator._local_llm_translator.translate_batch(
                texts_to_translate,
                source=manga_translator.source,
                target=manga_translator.target
            )
        except Exception as e:
            print(f"Copilot batch translation failed: {e}")
            translated_texts = texts_to_translate  # Return original on error
    
    elif selected_translator == "deepseek" and len(texts_to_translate) > 1:
        try:
            if not hasattr(manga_translator, "_deepseek_translator") or manga_translator._deepseek_translator is None:
                from translator.deepseek_translator import DeepSeekTranslator

                api_key = getattr(manga_translator, "_deepseek_api_key", None)
                model = getattr(manga_translator, "_deepseek_model", "deepseek-v4-flash")
                custom_prompt = getattr(manga_translator, "_deepseek_custom_prompt", None)
                thinking = getattr(manga_translator, "_deepseek_thinking", False)

                manga_translator._deepseek_translator = DeepSeekTranslator(
                    api_key=api_key,
                    model=model,
                    custom_prompt=custom_prompt,
                    thinking=thinking,
                )

                print(f"DeepSeek translator initialized: {model}")

            translated_texts = manga_translator._deepseek_translator.translate_batch(
                texts_to_translate,
                source=manga_translator.source,
                target=manga_translator.target,
            )

        except Exception as e:
            print(f"DeepSeek batch translation failed: {e}")
            translated_texts = texts_to_translate                       
    
    else:
        # Single translation for other translators
        # Optimized: Use batch translation if available (e.g. for NLLB)
        translated_texts = manga_translator.translate_batch(texts_to_translate, method=selected_translator)
    
    font_path = get_font_path(selected_font)
    return finalize_page_translation(image, render_items, translated_texts, font_path)


def get_font_path(font_name: str) -> str:
    """Get the correct font file path based on font name."""
    # Handle legacy fonts with 'i' suffix
    if font_name in ["animeace_", "arial", "mangat"]:
        return f"fonts/{font_name}i.ttf"
    # Yuki-* fonts use exact name
    elif font_name.startswith("Yuki-") or font_name.startswith("yuki-"):
        return f"fonts/{font_name}.ttf"
    else:
        return f"fonts/{font_name}.ttf"


def process_images_with_batch(images_data, manga_translator, mocr, selected_font, translator_type, batch_size=10, use_context_memory=True):
    """
    Process multiple images with multi-page batching for Copilot or Gemini.
    Collects all texts first, batch translates, then applies translations.
    
    Args:
        images_data: List of dicts with 'image', 'name' keys
        manga_translator: MangaTranslator instance with translator
        mocr: OCR engine
        selected_font: Font to use
        translator_type: 'copilot' or 'gemini'
        batch_size: Number of pages per API call
        use_context_memory: Whether to include context from all pages for better translation
        
    Returns:
        List of processed images with translations applied
    """
    import time
    
    def emit_progress(phase, current, total, message):
        """Emit progress update via WebSocket."""
        try:
            socketio.emit('progress', {
                'phase': phase,
                'current': current,
                'total': total,
                'message': message,
                'percent': int((current / max(total, 1)) * 100)
            })
        except Exception:
            pass  # Silently fail if socket not connected
    
    total_images = len(images_data)
    log(f"Processing {total_images} images... Context Memory: {'ON' if use_context_memory else 'OFF'}")
    
    start_time = time.time()
    
    # Use batch OCR when the selected backend supports it.
    use_batch_ocr = hasattr(mocr, 'process_batch')
    
    # Phase 1a: Detect page regions and collect all OCR items
    print("\n[Phase 1] Detecting page regions...")
    emit_progress('detection', 0, total_images, 'Starting page analysis...')

    all_pages_data = {}
    all_ocr_images = []
    ocr_mapping = []
    total_bubbles = 0
    total_text_regions = 0
    total_outside_text_regions = 0
    
    for idx, img_data in enumerate(images_data):
        image = img_data['image']
        name = img_data['name']
        

        emit_progress('detection', idx + 1, total_images, f'Analyzing page: {name}')
        print(f"  [{idx+1}/{total_images}] {name}", end="", flush=True)
        
        page_detection_result = detect_page_regions_layout_first(image)
        render_items, _, outside_render_items = collect_page_render_items(
            image,
            page_detection_result,
        )
        page_stats = page_detection_result.stats or {}
        bubble_count = sum(1 for item in render_items if item["kind"] == "bubble")
        text_region_count = len(page_detection_result.text_regions)
        outside_count = len(outside_render_items)
        total_bubbles += bubble_count
        total_text_regions += text_region_count
        total_outside_text_regions += outside_count

        all_pages_data[name] = {
            'image': image,
            'render_items': render_items,
            'texts': []
        }

        print(
            " - "
            f"raw_bubbles={page_stats.get('raw_bubbles', bubble_count)}, "
            f"merged_bubbles={page_stats.get('merged_bubbles', bubble_count)}, "
            f"raw_pp_text_regions={page_stats.get('raw_pp_text_regions', text_region_count)}, "
            f"raw_comic_text_regions={page_stats.get('raw_comic_text_regions', text_region_count)}, "
            f"pp_outside_blocks={page_stats.get('pp_outside_blocks', outside_count)}, "
            f"comic_fallback_outside_blocks={page_stats.get('comic_fallback_outside_blocks', 0)}, "
            f"bubble_items={page_stats.get('bubble_items', bubble_count)}, "
            f"outside_text_items={page_stats.get('outside_text_items', outside_count)}, "
            f"final_ocr_items={page_stats.get('final_render_items', len(render_items))}"
        )

        for item_idx, render_item in enumerate(render_items):
            all_ocr_images.append(Image.fromarray(render_item["ocr_crop"]))
            ocr_mapping.append((name, item_idx))
    
    total_text_items = len(all_ocr_images)
    detection_time = time.time() - start_time
    print(
        f"Detected {total_bubbles} bubbles, {total_text_regions} text regions, "
        f"{total_text_items} final OCR items ({total_outside_text_regions} outside text regions)"
    )
    emit_progress(
        'detection',
        total_images,
        total_images,
        f'Detected {total_bubbles} bubbles, {total_text_regions} text regions, {total_text_items} final OCR items',
    )
    print(f"Page analysis completed in {detection_time:.1f}s ({total_text_items} text items)")

    
    # Phase 1b: Batch OCR all text items at once
    if all_ocr_images:
        ocr_start = time.time()
        print(f"Preparing OCR for {len(all_ocr_images)} text items...", end=" ", flush=True)
        emit_progress('ocr', 0, 1, f'OCR processing {len(all_ocr_images)} text items...')
        print(f"\n[Phase 2] OCR processing {len(all_ocr_images)} text items...", end=" ", flush=True)
        
        if use_batch_ocr:
            # Use batch OCR on providers that implement process_batch.
            all_texts = mocr.process_batch(all_ocr_images)
        else:
            # Sequential OCR for simple callable providers.
            all_texts = [mocr(img) for img in all_ocr_images]
        
        # Map texts back to pages
        for (page_name, item_idx), text in zip(ocr_mapping, all_texts):
            all_pages_data[page_name]['texts'].append(text)

        for page_name, page_data in all_pages_data.items():
            deduped_items, deduped_texts = dedupe_ocr_items_by_text_and_geometry(
                page_data['render_items'],
                page_data['texts'],
                logger=log if VERBOSE_LOG else None,
            )
            page_data['render_items'] = deduped_items
            page_data['texts'] = deduped_texts
        total_text_items = sum(len(page_data['render_items']) for page_data in all_pages_data.values())
        
        ocr_time = time.time() - ocr_start
        print(f"({ocr_time:.1f}s)")
        print(f"OCR completed for {len(all_ocr_images)} text items")
        print(f"OCR finished in {ocr_time:.1f}s ({len(all_ocr_images)/max(ocr_time, 1e-6):.1f} text items/sec)")
        emit_progress('ocr', 1, 1, f'OCR complete ({len(all_ocr_images)} text items)')
    
    # Phase 3: Batch translate all pages together

    emit_progress('translation', 0, 1, 'Translating text...')
    pages_texts = {name: data['texts'] for name, data in all_pages_data.items() if data['texts']}
    all_translations = {}
    
    if pages_texts:
        # Get the translator based on type
        if translator_type == "copilot" and hasattr(manga_translator, '_local_llm_translator') and manga_translator._local_llm_translator:
            translator = manga_translator._local_llm_translator
            translator_name = "Local LLM"
        elif translator_type == "gemini" and hasattr(manga_translator, '_gemini_translator') and manga_translator._gemini_translator:
            translator = manga_translator._gemini_translator
            translator_name = "Gemini"
        elif translator_type == "deepseek" and hasattr(manga_translator, "_deepseek_translator") and manga_translator._deepseek_translator:
            translator = manga_translator._deepseek_translator
            translator_name = "DeepSeek"
        else:
            translator = None
            translator_name = "Unknown"
        
        if translator:
            print(f"{translator_name} batch translating {len(pages_texts)} pages in chunks of {batch_size}...")
            
            # Initialize context memory if enabled
            context_memory = None
            if use_context_memory:
                context_memory = ContextMemory()
                print(f"  Context Memory enabled - tracking terms and story context")
            
            # Process in batches
            page_names = list(pages_texts.keys())
            
            for i in range(0, len(page_names), batch_size):
                batch_names = page_names[i:i + batch_size]
                batch_texts = {name: pages_texts[name] for name in batch_names}
                
                print(f"  Translating batch {i//batch_size + 1}: pages {i+1}-{min(i+batch_size, len(page_names))}")
                
                try:
                    translated = translator.translate_pages_batch(
                        batch_texts,
                        source=manga_translator.source,
                        target=manga_translator.target,
                        context_memory=context_memory
                    )
                    all_translations.update(translated)
                    
                    # Update context memory with this batch's translations
                    if context_memory:
                        context_memory.update_from_translation(batch_texts, translated)
                        stats = context_memory.get_stats()
                        print(f"    Context updated: {stats['tracked_words']} terms tracked, {stats['recent_pages']} pages in memory")
                        
                except Exception as e:
                    print(f"  Batch failed: {e}, falling back to individual translation")
                    for name, texts in batch_texts.items():
                        try:
                            all_translations[name] = translator.translate_batch(
                                texts, manga_translator.source, manga_translator.target
                            )
                        except:
                            all_translations[name] = texts  # Return original on error
    
    translation_time = time.time() - start_time - detection_time
    print(f"Translation completed in {translation_time:.1f}s")
    emit_progress('translation', 1, 1, 'Translation complete')
    
    # Phase 4: Inpaint page text and render translations
    emit_progress('rendering', 0, total_images, 'Inpainting pages and rendering translations...')

    render_start = time.time()
    processed_results = []
    font_path = get_font_path(selected_font)
    
    print(f"\n[Phase 4] Inpainting pages and rendering translated text...")
    
    render_idx = 0
    for name, data in all_pages_data.items():
        render_idx += 1
        emit_progress('rendering', render_idx, total_images, f'Inpaint and render page: {name}')
        
        image = data['image']
        render_items = data['render_items']
        translated_texts = all_translations.get(name, data['texts'])  # Fallback to original

        final_image = finalize_page_translation(
            image,
            render_items,
            translated_texts,
            font_path,
        )

        processed_results.append({
            'image': final_image,
            'name': name
        })
    
    render_time = time.time() - render_start
    total_time = time.time() - start_time
    
    print(f"Inpainting and text rendering completed in {render_time:.1f}s")
    print(f"{'='*50}")
    print(f"TOTAL: {total_images} images processed in {total_time:.1f}s ({total_time/total_images:.1f}s/image)")
    print(f"{'='*50}\n")
    
    emit_progress('done', total_images, total_images, f'Completed {total_images} images in {total_time:.1f}s')
    
    return processed_results


@app.route("/translate", methods=["POST"])
def upload_file():
    # Get translator selection
    translator_map = {
        "Opus-mt model": "hf",
        "NLLB": "nllb",
        "Gemini": "gemini",
        "Local LLM": "copilot",
        "DeepSeek": "deepseek"# copilot is internal name for OpenAI-compatible endpoints
    }
    selected_translator = translator_map.get(
        request.form["selected_translator"],
        request.form["selected_translator"].lower()
    )
    
    # Get Local LLM settings if selected (Ollama, LM Studio, etc.)
    copilot_server = request.form.get("copilot_server", "http://localhost:8080")
    copilot_model = request.form.get("copilot_model_input", "gpt-4o")
    
    # Get Gemini API key from form
    gemini_api_key = request.form.get("gemini_api_key", "").strip()
    
    deepseek_api_key = request.form.get("deepseek_api_key", "").strip()
    deepseek_model = request.form.get("deepseek_model", "deepseek-v4-pro").strip()
    deepseek_thinking = request.form.get("deepseek_thinking") == "on"
    
    # Get context memory setting (checkbox - "on" if checked, None if not)
    use_context_memory = request.form.get("context_memory") == "on"

    # Get split long images setting (checkbox - "on" if checked, None if not)
    split_long_images = request.form.get("split_long_images") == "on"

    # Get font selection
    selected_font_raw = request.form["selected_font"]
    selected_font = selected_font_raw.lower()
    
    # Handle special font name mappings
    if selected_font == "auto (match original)":
        selected_font = "auto"
    elif selected_font == "animeace":
        selected_font = "animeace_"
    elif selected_font_raw.startswith("Yuki-"):
        # Keep original case for Yuki fonts
        selected_font = selected_font_raw

    # Get OCR engine
    selected_ocr_raw = request.form.get("selected_ocr", "PaddleOCR-VL Local").strip()
    selected_ocr = {
        "chrome-lens": "chrome-lens",
        "paddleocr-vl local": "paddleocr-vl",
        "paddleocr-vl": "paddleocr-vl",
    }.get(selected_ocr_raw.lower(), selected_ocr_raw.lower())
    
    # Get source language
    source_lang_map = {
        "japanese (manga)": "ja",
        "chinese (manhua)": "zh",
        "korean (manhwa)": "ko",
        "english (comic)": "en"
    }
    selected_source = request.form.get("selected_source_lang", "Japanese (Manga)").lower()
    source_lang = source_lang_map.get(selected_source, "ja")
    
    # Get target language
    target_lang_map = {
        "english": "en",
        "vietnamese": "vi", 
        "chinese": "zh",
        "korean": "ko",
        "thai": "th",
        "indonesian": "id",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "russian": "ru"
    }
    selected_language = request.form.get("selected_language", "Vietnamese").lower()
    target_lang = target_lang_map.get(selected_language, "vi")
    
    # Get translation style/custom prompt
    style_map = {
        "default": "",
        "casual (thÃƒÂ¢n mÃ¡ÂºÂ­t)": "casual",
        "formal (trang trÃ¡Â»Âng)": "formal",
        "keep honorifics (-san, senpai...)": "keep_honorifics",
        "web novel style": "web_novel",
        "action (ngÃ¡ÂºÂ¯n gÃ¡Â»Ân)": "action",
        "literal (sÃƒÂ¡t nghÃ„Â©a)": "literal",
        "custom...": ""
    }
    selected_style = request.form.get("selected_style", "Default").lower()
    style = style_map.get(selected_style, "")
    
    # Get custom prompt if provided
    custom_prompt = request.form.get("custom_prompt", "").strip()
    if custom_prompt:
        style = custom_prompt  # Override style with custom prompt

    # Get multiple files
    files = request.files.getlist("files")
    
    if not files or files[0].filename == '':
        return redirect("/")
    
    # Initialize translator and OCR once for all images
    manga_translator = MangaTranslator(source=source_lang, target=target_lang)
    
    # Set custom prompt for Gemini
    if selected_translator == "gemini" and style:
        manga_translator._gemini_custom_prompt = style
    
    # Set custom prompt for Local LLM
    if selected_translator == "copilot" and style:
        manga_translator._copilot_custom_prompt = style
    
    # Set Gemini API key
    if selected_translator == "gemini" and gemini_api_key:
        manga_translator._gemini_api_key = gemini_api_key
        print(f"Using Gemini API with provided key")
    
    # Set Copilot settings
    if selected_translator == "copilot":
        manga_translator._copilot_server = copilot_server
        manga_translator._copilot_model = copilot_model
        print(f"Using Local LLM: {copilot_server} / model: {copilot_model}")
    
    if selected_translator == "deepseek":
        manga_translator._deepseek_api_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
        manga_translator._deepseek_model = deepseek_model or "deepseek-v4-flash"
        manga_translator._deepseek_thinking = deepseek_thinking

        if style:
            manga_translator._deepseek_custom_prompt = style

        print(f"Using DeepSeek API: model={manga_translator._deepseek_model}, thinking={deepseek_thinking}")
        
    try:
        if selected_ocr == "chrome-lens":
            if _OCR_CACHE["chrome_lens"] is None:
                _OCR_CACHE["chrome_lens"] = ChromeLensOCR()
            mocr = _OCR_CACHE["chrome_lens"]

        elif selected_ocr == "paddleocr-vl":
            mocr = get_paddleocr_vl_ocr(source_language=source_lang)
            print("Using PaddleOCR-VL via llama.cpp")
            print(f"PaddleOCR-VL server: {mocr.server_url}")

        else:
            raise ValueError(
                f"Unsupported OCR provider '{selected_ocr_raw}'. "
                "Supported providers: PaddleOCR-VL Local, Chrome-Lens."
            )
    except Exception as exc:
        error_message = (
            f"OCR provider '{selected_ocr}' failed to initialize: {exc}\n"
            "For PaddleOCR-VL, set PADDLEOCR_VL_MODEL_PATH, PADDLEOCR_VL_MMPROJ_PATH, "
            "and LLAMA_CPP_DIR or PADDLEOCR_VL_SERVER_URL. "
            "Chrome-Lens remains available as the optional fallback provider."
        )
        return error_message, 500
    
    # Initialize font analyzer for auto font matching
    font_analyzer = None
    if selected_font == "auto":
        try:
            from font_analyzer import FontAnalyzer
            # Use same API key as Gemini translator
            api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("Warning: No Gemini API key provided for font analysis")
            font_analyzer = FontAnalyzer(api_key=api_key)
            print("Font analyzer initialized for auto font matching")
        except Exception as e:
            print(f"Failed to initialize font analyzer: {e}")
            selected_font = "animeace_"  # Fallback to default
    
    # Process all images
    processed_images = []
    auto_font_determined = False  # Flag to analyze font only once
    
    # For Local LLM and Gemini: Use multi-page batch processing
    if selected_translator in ["copilot", "gemini", "deepseek"]:
        # First, read all images into memory
        all_images = []
        for file in files:
            if file and file.filename:
                try:
                    file_stream = file.stream
                    file_bytes = np.frombuffer(file_stream.read(), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        continue
                    
                    name = os.path.splitext(file.filename)[0]
                    all_images.append({'image': image, 'name': name})
                except Exception as e:
                    print(f"Error reading {file.filename}: {e}")
        
        if not all_images:
            return redirect("/")
        
        # Auto font: analyze first image
        if selected_font == "auto" and font_analyzer is not None:
            try:
                first_bubble = get_first_layout_first_bubble_crop(all_images[0]['image'])
                if first_bubble is not None and getattr(first_bubble, "size", 0) > 0:
                    selected_font = font_analyzer.analyze_and_match(first_bubble)
                    print(f"Auto font matched: {selected_font}")
                else:
                    selected_font = "animeace_"
            except Exception as e:
                print(f"Font analysis failed: {e}")
                selected_font = "animeace_"
        
        # Initialize translator based on type
        if selected_translator == "copilot":
            if not hasattr(manga_translator, '_local_llm_translator') or manga_translator._local_llm_translator is None:
                from translator.local_llm_translator import LocalLLMTranslator
                # Get custom prompt for Local LLM
                copilot_custom_prompt = style if style else None
                manga_translator._local_llm_translator = LocalLLMTranslator(
                    server_url=copilot_server,
                    model=copilot_model,
                    custom_prompt=copilot_custom_prompt
                )
                print(f"Local LLM translator initialized: {copilot_server} / {copilot_model} (style: {style or 'default'})")
        
        elif selected_translator == "gemini":
            if not hasattr(manga_translator, '_gemini_translator') or manga_translator._gemini_translator is None:
                from translator.gemini_translator import GeminiTranslator
                api_key = gemini_api_key
                if not api_key:
                    raise ValueError("Gemini API key required. Please enter it in the web form.")
                custom_prompt = getattr(manga_translator, '_gemini_custom_prompt', None)
                manga_translator._gemini_translator = GeminiTranslator(
                    api_key=api_key,
                    custom_prompt=custom_prompt
                )
                print("Gemini translator initialized for multi-page batching")
        
        elif selected_translator == "deepseek":
            if not hasattr(manga_translator, "_deepseek_translator") or manga_translator._deepseek_translator is None:
                from translator.deepseek_translator import DeepSeekTranslator

                deepseek_custom_prompt = style if style else None

                manga_translator._deepseek_translator = DeepSeekTranslator(
                    api_key=deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY"),
                    model=deepseek_model or "deepseek-v4-flash",
                    custom_prompt=deepseek_custom_prompt,
                    thinking=deepseek_thinking,
                )

                print(
                    f"DeepSeek translator initialized: "
                    f"{deepseek_model or 'deepseek-v4-flash'} "
                    f"(style: {style or 'default'}, thinking: {deepseek_thinking})"
                )
        
        # Process with multi-page batching (10 pages per API call)
        processed_results = process_images_with_batch(
            all_images, manga_translator, mocr, selected_font, 
            translator_type=selected_translator, batch_size=10,
            use_context_memory=use_context_memory,
        )
        
        # Encode results to base64 (with optional splitting)
        for result in processed_results:
            try:
                image = result['image']
                base_name = result['name']
                
                # Split long images if enabled
                if split_long_images:
                    chunks = split_long_image(image)
                else:
                    chunks = [image]
                
                # Encode each chunk
                for i, chunk in enumerate(chunks):
                    _, buffer = cv2.imencode(".jpg", chunk, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    encoded_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
                    
                    # Add suffix if split into multiple chunks
                    if len(chunks) > 1:
                        chunk_name = f"{base_name}_part{i+1}"
                    else:
                        chunk_name = base_name
                    
                    processed_images.append({
                        "name": chunk_name,
                        "data": encoded_image
                    })
            except Exception as e:
                print(f"Error encoding {result['name']}: {e}")
    
    else:
        # For other translators: Use per-image processing (original flow)
        for file in files:
            if file and file.filename:
                try:
                    # Read image
                    file_stream = file.stream
                    file_bytes = np.frombuffer(file_stream.read(), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        continue
                    
                    # Auto font: analyze FIRST image only
                    if selected_font == "auto" and font_analyzer is not None and not auto_font_determined:
                        try:
                            first_bubble = get_first_layout_first_bubble_crop(image)
                            if first_bubble is not None and getattr(first_bubble, "size", 0) > 0:
                                selected_font = font_analyzer.analyze_and_match(first_bubble)
                                print(f"Auto font matched (once for all images): {selected_font}")
                            else:
                                selected_font = "animeace_"
                        except Exception as e:
                            print(f"Font analysis failed: {e}")
                            selected_font = "animeace_"
                        auto_font_determined = True
                    
                    # Get original filename
                    name = os.path.splitext(file.filename)[0]
                    
                    # Process image
                    processed_image = process_single_image(
                        image, manga_translator, mocr, 
                        selected_translator, selected_font, None,
                    )
                    
                    # Split long images if enabled
                    if split_long_images:
                        chunks = split_long_image(processed_image)
                    else:
                        chunks = [processed_image]
                    
                    # Encode each chunk to base64
                    for i, chunk in enumerate(chunks):
                        _, buffer = cv2.imencode(".jpg", chunk, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        encoded_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
                        
                        # Add suffix if split into multiple chunks
                        if len(chunks) > 1:
                            chunk_name = f"{name}_part{i+1}"
                        else:
                            chunk_name = name
                        
                        processed_images.append({
                            "name": chunk_name,
                            "data": encoded_image
                        })
                    
                except Exception as e:
                    print(f"Error processing {file.filename}: {e}")
                    continue
    
    if not processed_images:
        return redirect("/")
    
    return render_template("translate.html", images=processed_images)


@app.route("/download-zip", methods=["POST"])
def download_zip():
    """Create and download a ZIP file containing all translated images."""
    try:
        images_data = request.form.get("images_data", "[]")
        images = json.loads(images_data)
        
        if not images:
            return redirect("/")
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, img in enumerate(images):
                name = img.get('name', f'image_{i+1}')
                data = img.get('data', '')
                
                # Decode base64 to bytes
                image_bytes = base64.b64decode(data)
                
                # Add to ZIP with proper filename
                filename = f"{name}_translated.png"
                zip_file.writestr(filename, image_bytes)
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='manga_translated.zip'
        )
    
    except Exception as e:
        print(f"Error creating ZIP: {e}")
        return redirect("/")


if __name__ == "__main__":
    socketio.run(app, debug=True)
