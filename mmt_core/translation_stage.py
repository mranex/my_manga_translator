"""Translation stage helpers backed by OCR and translation cache JSON files."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import importlib
from pathlib import Path
from typing import Any

from .canon_state import canon_item_bbox, ensure_canon_state, resolve_canon_item_for_stage_item
from .detection_io import detection_json_path, load_detection_json, save_detection_json
from .ocr_io import load_ocr_json, ocr_json_path
from .translation_io import (
    initialize_translation_from_ocr,
    load_translation_json,
    save_translation_json,
    summarize_translation_json,
    translation_json_path,
)
from .translation_models import TranslationConfig


ProgressCallback = Callable[[dict[str, Any]], None]


def initialize_translation_for_page(
    project,
    image_relative_path,
    config,
    force: bool = False,
    logger: Callable[[str], None] | None = None,
) -> Path:
    """Create or refresh translation JSON for one page from cached OCR results."""

    translation_config = TranslationConfig.from_value(config)
    relative_path = Path(str(image_relative_path))
    _log(logger, f"Initializing translation cache for {relative_path.name}")
    output_path = initialize_translation_from_ocr(
        project,
        relative_path,
        translation_config,
        force=force,
    )
    _log(logger, f"Saved translation cache: {output_path}")
    return output_path


def run_translation_for_page(
    project,
    image_relative_path,
    config,
    force: bool = False,
    selected_item_ids: Sequence[int] | None = None,
    logger: Callable[[str], None] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Translate one page from cached OCR text into cached translation JSON."""

    translation_config = TranslationConfig.from_value(config)
    adapter = _TranslationAdapter(translation_config, logger=logger)
    context_memory = _create_context_memory(translation_config, logger=logger)
    return _run_translation_for_page_with_adapter(
        project,
        image_relative_path,
        translation_config,
        adapter,
        force=force,
        selected_item_ids=selected_item_ids,
        logger=logger,
        progress_callback=progress_callback,
        context_memory=context_memory,
    )


def run_translation_for_pages(
    project,
    image_relative_paths,
    config,
    force: bool = False,
    logger: Callable[[str], None] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    """Translate many pages while loading only one page chunk of OCR/translation JSON at a time."""

    translation_config = TranslationConfig.from_value(config)
    adapter = _TranslationAdapter(translation_config, logger=logger)
    context_memory = _create_context_memory(translation_config, logger=logger)
    batch_helpers = _load_batch_helpers()

    ordered_relative_paths = [str(path) for path in image_relative_paths if str(path).strip()]
    if not ordered_relative_paths:
        raise ValueError("No pages were provided for translation.")

    chunk_size = max(1, int(translation_config.batch_size_pages))
    chunks = batch_helpers["build_page_chunks"](ordered_relative_paths, chunk_size, default_size=chunk_size)
    output_paths: list[Path] = []
    last_error: Exception | None = None
    total_chunks = len(chunks)

    for chunk_index, chunk_paths in enumerate(chunks, start=1):
        _emit_progress(
            progress_callback,
            {
                "event": "chunk_start",
                "chunk_index": chunk_index,
                "chunk_total": total_chunks,
                "page_count": len(chunk_paths),
                "page_names": list(chunk_paths),
                "message": f"Starting translation chunk {chunk_index}/{total_chunks}",
            },
        )
        _log(
            logger,
            f"Starting translation chunk {chunk_index}/{total_chunks}: "
            + ", ".join(Path(page_name).name for page_name in chunk_paths),
        )

        if translation_config.supports_page_batch:
            chunk_paths_saved, chunk_error = _run_translation_chunk_with_page_batch(
                project,
                chunk_paths,
                translation_config,
                adapter,
                force=force,
                logger=logger,
                progress_callback=progress_callback,
                context_memory=context_memory,
            )
            output_paths.extend(chunk_paths_saved)
            if chunk_error is not None:
                last_error = chunk_error
        else:
            for page_offset, image_relative_path in enumerate(chunk_paths, start=1):
                try:
                    output_path = _run_translation_for_page_with_adapter(
                        project,
                        image_relative_path,
                        translation_config,
                        adapter,
                        force=force,
                        selected_item_ids=None,
                        logger=logger,
                        progress_callback=progress_callback,
                        context_memory=context_memory,
                    )
                except Exception as exc:
                    last_error = exc
                    _emit_progress(
                        progress_callback,
                        {
                            "event": "page_error",
                            "image_relative_path": str(image_relative_path),
                            "message": str(exc),
                        },
                    )
                    _log(logger, f"Translation failed for {Path(image_relative_path).name}: {exc}")
                    continue

                output_paths.append(output_path)
                _emit_progress(
                    progress_callback,
                    {
                        "event": "page_done",
                        "image_relative_path": str(image_relative_path),
                        "page_index": page_offset,
                        "page_total": len(chunk_paths),
                        "message": f"Translated {Path(image_relative_path).name}",
                        "output_path": str(output_path),
                    },
                )

        _emit_progress(
            progress_callback,
            {
                "event": "chunk_done",
                "chunk_index": chunk_index,
                "chunk_total": total_chunks,
                "page_count": len(chunk_paths),
                "page_names": list(chunk_paths),
                "message": f"Finished translation chunk {chunk_index}/{total_chunks}",
            },
        )

    if not output_paths and last_error is not None:
        raise RuntimeError(str(last_error))
    if not output_paths:
        raise RuntimeError("Translation did not produce any cache files.")
    return output_paths


def validate_translation_config(
    config,
    *,
    logger: Callable[[str], None] | None = None,
) -> TranslationConfig:
    """Validate translator settings and fail early on provider configuration errors."""

    translation_config = TranslationConfig.from_value(config)
    adapter = _TranslationAdapter(translation_config, logger=logger)
    adapter._ensure_translator()
    return translation_config


def _run_translation_chunk_with_page_batch(
    project,
    chunk_paths: list[str],
    config: TranslationConfig,
    adapter: "_TranslationAdapter",
    *,
    force: bool,
    logger: Callable[[str], None] | None,
    progress_callback: ProgressCallback | None,
    context_memory: Any,
) -> tuple[list[Path], Exception | None]:
    page_payloads: dict[str, dict[str, Any]] = {}
    page_json_paths: dict[str, Path] = {}
    page_entry_indices: dict[str, list[int]] = {}
    pages_texts: dict[str, list[str]] = {}
    output_paths: list[Path] = []

    try:
        for image_relative_path in chunk_paths:
            json_path, payload = _ensure_translation_payload(project, image_relative_path, config)
            page_key = str(image_relative_path)
            entries = _collect_translation_entries(payload, force=force, selected_item_ids=None)
            for entry in entries:
                _mark_item_running(payload["items"][entry["index"]], config)
            save_translation_json(json_path, payload)

            page_payloads[page_key] = payload
            page_json_paths[page_key] = json_path
            page_entry_indices[page_key] = [entry["index"] for entry in entries]
            pages_texts[page_key] = [entry["item"]["source_text"] for entry in entries]
            output_paths.append(json_path)

            if not entries:
                _emit_progress(
                    progress_callback,
                    {
                        "event": "page_done",
                        "image_relative_path": page_key,
                        "output_path": str(json_path),
                        "message": f"No translation needed for {Path(page_key).name}",
                    },
                )

        translatable_pages_texts = {
            page_key: texts
            for page_key, texts in pages_texts.items()
            if texts
        }
        if not translatable_pages_texts:
            return output_paths, None

        translated_pages = adapter.translate_pages(
            translatable_pages_texts,
            context_memory=context_memory,
            batch_size=len(translatable_pages_texts),
        )
    except Exception as exc:
        for page_key, payload in page_payloads.items():
            entries = page_entry_indices.get(page_key, [])
            for item_index in entries:
                _mark_item_error(payload["items"][item_index], config, str(exc))
            json_path = page_json_paths.get(page_key)
            if json_path is not None:
                _touch_translation_payload(payload, config)
                save_translation_json(json_path, payload)
        return output_paths, exc

    for page_key, translated_lines in translated_pages.items():
        payload = page_payloads.get(page_key)
        json_path = page_json_paths.get(page_key)
        entry_indices = page_entry_indices.get(page_key, [])
        if payload is None or json_path is None:
            continue

        repaired_lines = _repair_translation_list(translated_lines, len(entry_indices))
        for line_index, item_index in enumerate(entry_indices):
            item = payload["items"][item_index]
            item["translated_text"] = repaired_lines[line_index]
            item["status"] = "done"
            item["error"] = ""
            item["updated_at"] = _timestamp()
            item["translator"] = config.translator
            _touch_translation_payload(payload, config)
            save_translation_json(json_path, payload)
            _emit_progress(
                progress_callback,
                {
                    "event": "item_done",
                    "image_relative_path": page_key,
                    "item_id": int(item.get("id", line_index)),
                    "message": f"Translated item {item.get('id', line_index)} on {Path(page_key).name}",
                    "output_path": str(json_path),
                },
            )

        _emit_progress(
            progress_callback,
            {
                "event": "page_done",
                "image_relative_path": page_key,
                "output_path": str(json_path),
                "summary": summarize_translation_json(payload),
                "message": f"Translated {Path(page_key).name}",
            },
        )

    missing_pages = set(translatable_pages_texts.keys()) - set(translated_pages.keys())
    for page_key in missing_pages:
        payload = page_payloads.get(page_key)
        json_path = page_json_paths.get(page_key)
        if payload is None or json_path is None:
            continue

        for item_index in page_entry_indices.get(page_key, []):
            _mark_item_error(
                payload["items"][item_index],
                config,
                f"{config.translator} did not return a result for this page.",
            )
        _touch_translation_payload(payload, config)
        save_translation_json(json_path, payload)
        _emit_progress(
            progress_callback,
            {
                "event": "page_error",
                "image_relative_path": page_key,
                "output_path": str(json_path),
                "message": f"{config.translator} did not return a result for {Path(page_key).name}",
            },
        )

    return output_paths, None


def _run_translation_for_page_with_adapter(
    project,
    image_relative_path,
    config: TranslationConfig,
    adapter: "_TranslationAdapter",
    *,
    force: bool,
    selected_item_ids: Sequence[int] | None,
    logger: Callable[[str], None] | None,
    progress_callback: ProgressCallback | None,
    context_memory: Any,
) -> Path:
    relative_path = Path(str(image_relative_path))
    json_path, payload = _ensure_translation_payload(project, relative_path, config)
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid translation cache: items must be a list in {json_path}")

    entries = _collect_translation_entries(
        payload,
        force=force,
        selected_item_ids=selected_item_ids,
    )

    if not entries:
        items = payload.get("items", [])
        has_source_text = any(
            str(item.get("source_text", "") or "").strip()
            for item in items
            if isinstance(item, dict)
        )
        if not has_source_text:
            _touch_translation_payload(payload, config)
            save_translation_json(json_path, payload)
            raise RuntimeError(
                f"OCR text is empty for all items on {relative_path.name}. Prepare and run OCR first."
            )
        _touch_translation_payload(payload, config)
        save_translation_json(json_path, payload)
        _log(logger, f"No translation needed for {relative_path.name}")
        return json_path

    texts_to_translate = [entry["item"]["source_text"] for entry in entries]
    for entry in entries:
        _mark_item_running(items[entry["index"]], config)
    _touch_translation_payload(payload, config)
    save_translation_json(json_path, payload)

    _log(logger, f"Translating {len(entries)} item(s) for {relative_path.name}")
    _emit_progress(
        progress_callback,
        {
            "event": "page_start",
            "image_relative_path": str(relative_path),
            "item_total": len(entries),
            "message": f"Translating {relative_path.name}",
        },
    )

    try:
        translated_lines = adapter.translate_texts(texts_to_translate)
    except Exception as exc:
        for entry in entries:
            _mark_item_error(items[entry["index"]], config, str(exc))
        _touch_translation_payload(payload, config)
        save_translation_json(json_path, payload)
        raise

    repaired_lines = _repair_translation_list(translated_lines, len(entries))
    for line_index, entry in enumerate(entries, start=1):
        item = items[entry["index"]]
        item["translated_text"] = repaired_lines[line_index - 1]
        item["status"] = "done"
        item["error"] = ""
        item["updated_at"] = _timestamp()
        item["translator"] = config.translator
        _touch_translation_payload(payload, config)
        save_translation_json(json_path, payload)
        _emit_progress(
            progress_callback,
            {
                "event": "item_done",
                "image_relative_path": str(relative_path),
                "item_id": int(item.get("id", line_index - 1)),
                "item_index": line_index,
                "item_total": len(entries),
                "message": f"Translated item {item.get('id', line_index - 1)}",
                "output_path": str(json_path),
            },
        )

    if context_memory is not None:
        try:
            context_memory.update_from_translation(
                {str(relative_path): texts_to_translate},
                {str(relative_path): repaired_lines},
            )
        except Exception as exc:
            _log(logger, f"Context memory update failed for {relative_path.name}: {exc}")

    _emit_progress(
        progress_callback,
        {
            "event": "page_done",
            "image_relative_path": str(relative_path),
            "summary": summarize_translation_json(payload),
            "message": f"Translated {relative_path.name}",
            "output_path": str(json_path),
        },
    )
    return json_path


def _ensure_translation_payload(project, image_relative_path, config: TranslationConfig) -> tuple[Path, dict[str, Any]]:
    relative_path = Path(str(image_relative_path))
    json_path = translation_json_path(project, relative_path)
    if not json_path.exists():
        initialize_translation_from_ocr(project, relative_path, config, force=False)
    payload = load_translation_json(json_path)
    payload, payload_changed = _refresh_translation_payload_from_canon(project, relative_path, payload)
    _touch_translation_payload(payload, config, touch_created=False)
    if payload_changed:
        save_translation_json(json_path, payload)
    return json_path, payload


def _collect_translation_entries(
    payload: dict[str, Any],
    *,
    force: bool,
    selected_item_ids: Sequence[int] | None,
) -> list[dict[str, Any]]:
    items = payload.get("items", [])
    selected_ids = _normalize_selected_ids(selected_item_ids)
    entries: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if bool(item.get("excluded", False)):
            continue

        source_text = str(item.get("source_text", "") or "").strip()
        status = str(item.get("status", "pending") or "pending").strip().lower()
        translated_text = str(item.get("translated_text", "") or "").strip()
        item_id = _coerce_int(item.get("id"), index)
        ocr_item_id = _coerce_int(item.get("ocr_item_id"), item_id)

        if not source_text:
            item["status"] = "skipped"
            item["translated_text"] = ""
            item["error"] = ""
            item["updated_at"] = _timestamp()
            continue

        if selected_ids is not None and item_id not in selected_ids and ocr_item_id not in selected_ids:
            continue

        if force:
            entries.append({"index": index, "item": item})
            continue

        if status == "manually_edited":
            continue
        if status == "done" and translated_text:
            continue
        if status == "skipped":
            continue
        if status == "error" and translated_text:
            continue
        if translated_text and status not in {"pending", "running", "error"}:
            continue

        entries.append({"index": index, "item": item})

    return entries


def _refresh_translation_payload_from_canon(
    project,
    image_relative_path: Path,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    detection_path = detection_json_path(project, image_relative_path)
    if not detection_path.exists():
        raise RuntimeError(
            f"Detection cache is missing for {image_relative_path.name}. Run Detection first."
        )

    detection_data = load_detection_json(detection_path)
    had_canon_state = isinstance(detection_data.get("canon_state"), dict)
    ensure_canon_state(detection_data)
    if not had_canon_state:
        save_detection_json(detection_path, detection_data)
    canon_state = detection_data["canon_state"]

    ocr_path = ocr_json_path(project, image_relative_path)
    ocr_data = load_ocr_json(ocr_path) if ocr_path.exists() else {"items": []}
    ocr_items_by_id = {
        int(item.get("id", index)): dict(item)
        for index, item in enumerate(ocr_data.get("items", []))
        if isinstance(item, dict)
    }

    changed = False
    for index, item in enumerate(payload.get("items", [])):
        if not isinstance(item, dict):
            continue
        ocr_item_id = int(item.get("ocr_item_id", index) or index)
        ocr_item = ocr_items_by_id.get(ocr_item_id, {})
        canon_item = resolve_canon_item_for_stage_item(canon_state, item, active_only=False)
        if canon_item is None and ocr_item:
            canon_item = resolve_canon_item_for_stage_item(canon_state, ocr_item, active_only=False)
        if canon_item is None:
            continue

        canon_id = str(canon_item.get("canon_id", "") or "")
        bbox = canon_item_bbox(canon_item, "bbox")
        ocr_bbox = canon_item_bbox(canon_item, "ocr_bbox")
        excluded = not bool(canon_item.get("enabled", True))
        if (
            str(item.get("canon_id", "") or "") != canon_id
            or item.get("bbox") != bbox
            or item.get("ocr_bbox") != ocr_bbox
            or bool(item.get("excluded", False)) != excluded
        ):
            changed = True
        item["canon_id"] = canon_id
        # Snapshot only. canon_state remains the source of truth.
        item["bbox"] = bbox
        item["ocr_bbox"] = ocr_bbox
        item["excluded"] = excluded
    return payload, changed


def _mark_item_running(item: dict[str, Any], config: TranslationConfig) -> None:
    item["status"] = "running"
    item["error"] = ""
    item["updated_at"] = _timestamp()
    item["translator"] = config.translator


def _mark_item_error(item: dict[str, Any], config: TranslationConfig, error_message: str) -> None:
    item["status"] = "error"
    item["error"] = str(error_message)
    item["updated_at"] = _timestamp()
    item["translator"] = config.translator


def _touch_translation_payload(
    payload: dict[str, Any],
    config: TranslationConfig,
    *,
    touch_created: bool = True,
) -> None:
    prompt_result = config.prompt_build_result
    if touch_created and not str(payload.get("created_at", "")).strip():
        payload["created_at"] = _timestamp()
    payload["updated_at"] = _timestamp()
    payload["source_language"] = config.source_language
    payload["target_language"] = config.target_language
    payload["translator"] = config.translator
    payload["provider"] = config.translator
    payload["style"] = prompt_result.style_name
    payload["custom_prompt"] = prompt_result.prompt_text
    payload["prompt_style_id"] = prompt_result.style_id
    payload["prompt_mode"] = prompt_result.prompt_mode
    payload["user_instructions"] = config.user_instructions
    payload["full_custom_prompt"] = config.full_custom_prompt
    payload["project_translation_notes"] = config.project_translation_notes
    payload["output_contract"] = prompt_result.output_contract
    payload["prompt_preview"] = prompt_result.prompt_preview
    payload["prompt_hash"] = prompt_result.prompt_hash


def _normalize_selected_ids(selected_item_ids: Sequence[int] | None) -> set[int] | None:
    if selected_item_ids is None:
        return None
    normalized = {_coerce_int(value, -1) for value in selected_item_ids}
    normalized.discard(-1)
    return normalized


def _repair_translation_list(translated_lines: Any, expected_length: int) -> list[str]:
    if not isinstance(translated_lines, list):
        translated_lines = []
    repaired = [str(item) for item in translated_lines[:expected_length]]
    while len(repaired) < expected_length:
        repaired.append("")
    return repaired


def _create_context_memory(
    config: TranslationConfig,
    *,
    logger: Callable[[str], None] | None,
) -> Any:
    if not config.use_context_memory:
        return None
    try:
        context_module = importlib.import_module("translator.context_memory")
        return context_module.ContextMemory()
    except Exception as exc:
        _log(logger, f"Context memory unavailable; continuing without it: {exc}")
        return None


def _load_batch_helpers() -> dict[str, Any]:
    module = importlib.import_module("translator.batch_orchestrator")
    return {
        "build_page_chunks": module.build_page_chunks,
        "translate_pages_chunked_with_recovery": module.translate_pages_chunked_with_recovery,
    }


class _TranslationAdapter:
    def __init__(
        self,
        config: TranslationConfig,
        *,
        logger: Callable[[str], None] | None,
    ) -> None:
        self.config = config
        self.logger = logger
        self._translator: Any = None

    def translate_texts(self, texts: list[str]) -> list[str]:
        translator = self._ensure_translator()
        if self.config.translator_key in {"google", "baidu", "bing", "nllb"}:
            return self._translate_texts_with_generic_translator(translator, texts)
        return self._call_with_supported_kwargs(
            translator.translate_batch,
            texts,
            source=self.config.source_language,
            target=self.config.target_language,
            custom_prompt=self.config.effective_prompt(),
        )

    def translate_pages(
        self,
        pages_texts: dict[str, list[str]],
        *,
        context_memory: Any,
        batch_size: int,
    ) -> dict[str, list[str]]:
        translator = self._ensure_translator()
        if not self.config.supports_page_batch:
            raise RuntimeError(f"{self.config.translator} does not support multi-page batch translation.")

        batch_helpers = _load_batch_helpers()
        return batch_helpers["translate_pages_chunked_with_recovery"](
            translator,
            pages_texts,
            source=self.config.source_language,
            target=self.config.target_language,
            custom_prompt=self.config.effective_prompt(),
            context_memory=context_memory,
            batch_size=batch_size,
            logger=self.logger or (lambda _message: None),
        )

    def _ensure_translator(self) -> Any:
        if self._translator is not None:
            return self._translator

        translator_key = self.config.translator_key
        prompt = self.config.effective_prompt()

        try:
            if translator_key == "gemini":
                module = importlib.import_module("translator.gemini_translator")
                self._translator = module.GeminiTranslator(
                    api_key=self.config.gemini_api_key or None,
                    custom_prompt=prompt,
                    style="default",
                )
            elif translator_key == "local_llm":
                module = importlib.import_module("translator.local_llm_translator")
                self._translator = module.LocalLLMTranslator(
                    server_url=self.config.local_llm_server_url,
                    model=self.config.local_llm_model,
                    custom_prompt=prompt,
                    style="default",
                )
                if hasattr(self._translator, "test_connection") and not self._translator.test_connection():
                    raise RuntimeError(
                        "Local LLM server is not reachable. Check the Translation tab server URL and model."
                    )
            elif translator_key == "deepseek":
                module = importlib.import_module("translator.deepseek_translator")
                self._translator = module.DeepSeekTranslator(
                    api_key=self.config.deepseek_api_key or None,
                    model=self.config.deepseek_model,
                    custom_prompt=prompt,
                    style="default",
                    thinking=self.config.deepseek_thinking,
                )
            elif translator_key == "openai_compatible":
                module = importlib.import_module("translator.openai_compatible_translator")
                self._translator = module.OpenAICompatibleTranslator(
                    provider_preset=self.config.openai_compatible_provider_preset,
                    base_url=self.config.openai_compatible_base_url,
                    api_key=self.config.openai_compatible_api_key or None,
                    model=self.config.openai_compatible_model,
                    custom_prompt=prompt,
                    style="default",
                    temperature=self.config.openai_compatible_temperature,
                    max_tokens=self.config.openai_compatible_max_tokens,
                    timeout=self.config.openai_compatible_timeout,
                    json_mode=self.config.openai_compatible_json_mode,
                )
            else:
                module = importlib.import_module("translator.translator")
                self._translator = module.MangaTranslator(
                    source=self.config.source_language,
                    target=self.config.target_language,
                    gemini_api_key=self.config.gemini_api_key or None,
                )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Translator dependency missing for {self.config.translator}: {exc.name}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize {self.config.translator} translator: {exc}") from exc

        return self._translator

    def _translate_texts_with_generic_translator(self, translator: Any, texts: list[str]) -> list[str]:
        try:
            return translator.translate_batch(texts, method=self.config.translator_key)
        except Exception as exc:
            if self.config.translator_key == "nllb":
                raise RuntimeError(f"NLLB translation failed: {exc}") from exc
            raise RuntimeError(f"{self.config.translator} translation failed: {exc}") from exc

    def _call_with_supported_kwargs(self, callable_obj, texts: list[str], **kwargs: Any) -> list[str]:
        import inspect

        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            signature = None

        if signature is None:
            return callable_obj(texts)

        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
        return callable_obj(texts, **filtered_kwargs)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _emit_progress(callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback is not None:
        callback(payload)


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


__all__ = [
    "initialize_translation_for_page",
    "run_translation_for_page",
    "run_translation_for_pages",
    "validate_translation_config",
]
