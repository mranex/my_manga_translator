"""Resident OCR service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mmt_core import OCRConfig
from mmt_core.ocr_io import load_ocr_json, ocr_json_path, summarize_ocr_items
from mmt_core.ocr_models import DEFAULT_OCR_PROVIDER, provider_label
from mmt_core.ocr_providers import OCRProvider, create_ocr_provider
from mmt_core.ocr_stage import prepare_ocr_items_for_image, run_ocr_for_page
from mmt_gui.workers import (
    OCRInferencePageResult,
    OCRInferenceTask,
    OCRInferenceWorkerResult,
    OCRPreparationPageResult,
    OCRPreparationTask,
    OCRPreparationWorkerResult,
)

from .base_service import BaseService, WorkerSignalsBridge
from .models import ServiceCommand


class OCRService(BaseService):
    def __init__(self, *, scheduler: Any | None = None, startup_options: dict | None = None) -> None:
        super().__init__("ocr", scheduler=scheduler, startup_options=startup_options)
        self._provider: OCRProvider | None = None
        self._provider_key: tuple[Any, ...] | None = None

    def on_initialize(self) -> None:
        self._emit_status("loading", "Starting OCR worker...")
        self._emit_log("info", "OCR service is ready. Local OCR server remains user-controlled.")

    def on_shutdown(self) -> None:
        self._close_provider()

    def on_restart(self) -> None:
        self._close_provider()
        self.on_initialize()

    def execute_command(self, command: ServiceCommand, bridge: WorkerSignalsBridge) -> Any:
        task = command.task
        if isinstance(task, OCRPreparationTask):
            return self._run_prepare_task(command, task, bridge)
        if isinstance(task, OCRInferenceTask):
            return self._run_inference_task(command, task, bridge)
        if command.action == "check_provider":
            config = OCRConfig.from_value(command.config)
            provider = self._ensure_provider(config)
            provider.validate()
            return {"provider": provider.provider_key, "label": provider.provider_label}
        if command.action == "reload_provider":
            self._close_provider()
            return {"reloaded": True}
        if command.action == "status":
            return {"provider": None if self._provider is None else self._provider.provider_key}
        raise RuntimeError(f"OCR service does not support action '{command.action}'.")

    def _run_prepare_task(
        self,
        command: ServiceCommand,
        task: OCRPreparationTask,
        bridge: WorkerSignalsBridge,
    ) -> OCRPreparationWorkerResult:
        if task.project is None:
            raise ValueError("No project was provided for OCR preparation.")
        if not task.image_relative_paths:
            raise ValueError("No project pages were provided for OCR preparation.")

        total_pages = len(task.image_relative_paths)
        page_results: list[OCRPreparationPageResult] = []
        bridge.message.emit(f"Starting OCR item preparation for {total_pages} page(s).")
        bridge.progress.emit(0)

        for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
            self._check_canceled(command, message="OCR preparation canceled before the next page.")
            page_name = Path(image_relative_path).name
            bridge.event.emit(
                {
                    "stage": "ocr_prepare",
                    "event": "page_start",
                    "image_relative_path": str(image_relative_path),
                    "page_index": index,
                    "page_total": total_pages,
                }
            )
            bridge.message.emit(f"[{index}/{total_pages}] Preparing OCR items for {page_name}")
            try:
                json_path = prepare_ocr_items_for_image(
                    task.project,
                    image_relative_path,
                    force=task.force,
                    save_crops=task.save_crops,
                    logger=bridge.message.emit,
                )
            except Exception as exc:
                readable_error = f"{page_name}: {exc}"
                bridge.message.emit(f"OCR preparation failed: {readable_error}")
                page_results.append(
                    OCRPreparationPageResult(
                        image_relative_path=str(image_relative_path),
                        json_path=None,
                        error=str(exc),
                    )
                )
                bridge.event.emit(
                    {
                        "stage": "ocr_prepare",
                        "event": "page_error",
                        "image_relative_path": str(image_relative_path),
                        "page_index": index,
                        "page_total": total_pages,
                        "message": str(exc),
                        "error": str(exc),
                    }
                )
            else:
                page_results.append(
                    OCRPreparationPageResult(
                        image_relative_path=str(image_relative_path),
                        json_path=json_path,
                        error=None,
                    )
                )
                bridge.message.emit(f"OCR items prepared: {json_path}")
                bridge.event.emit(
                    {
                        "stage": "ocr_prepare",
                        "event": "page_done",
                        "image_relative_path": str(image_relative_path),
                        "output_path": str(json_path),
                        "page_index": index,
                        "page_total": total_pages,
                    }
                )
            bridge.progress.emit(int((index / total_pages) * 100))

        successful_pages = [result for result in page_results if result.json_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown OCR preparation failure."
            raise RuntimeError(f"OCR item preparation failed for all pages. {first_error}")
        return OCRPreparationWorkerResult(page_results=page_results)

    def _run_inference_task(
        self,
        command: ServiceCommand,
        task: OCRInferenceTask,
        bridge: WorkerSignalsBridge,
    ) -> OCRInferenceWorkerResult:
        if task.project is None:
            raise ValueError("No project was provided for OCR inference.")
        if not task.image_relative_paths:
            raise ValueError("No OCR-prepared pages were provided for OCR inference.")

        ocr_config = OCRConfig.from_value(task.config)
        if not str(ocr_config.server_url).strip() and str(task.server_url).strip():
            ocr_config.server_url = str(task.server_url).strip()
        if float(task.timeout) > 0:
            ocr_config.timeout = float(task.timeout)

        total_pages = len(task.image_relative_paths)
        page_results: list[OCRInferencePageResult] = []
        bridge.message.emit(f"Running OCR with {provider_label(ocr_config.ocr_provider)}")
        bridge.message.emit(f"Starting OCR inference for {total_pages} page(s).")
        bridge.progress.emit(0)

        provider = self._ensure_provider(ocr_config)
        provider.validate()
        for page_index, image_relative_path in enumerate(task.image_relative_paths, start=1):
            self._check_canceled(command, message="OCR canceled before the next page.")
            page_name = Path(image_relative_path).name
            bridge.event.emit(
                {
                    "stage": "ocr",
                    "event": "page_start",
                    "image_relative_path": str(image_relative_path),
                    "page_index": page_index,
                    "page_total": total_pages,
                }
            )
            bridge.message.emit(f"[{page_index}/{total_pages}] Running OCR for {page_name}")

            def on_item_progress(current: int, total: int, item_info: dict[str, Any]) -> None:
                safe_total = max(int(total), 1)
                progress_ratio = ((page_index - 1) + (min(int(current), safe_total) / safe_total)) / total_pages
                bridge.progress.emit(int(progress_ratio * 100))
                item_message = str(item_info.get("message", "") or "").strip()
                if item_message:
                    bridge.message.emit(
                        f"[{page_index}/{total_pages}] {page_name} item {min(int(current), safe_total)}/{safe_total}: {item_message}"
                    )

            try:
                json_path = run_ocr_for_page(
                    task.project,
                    image_relative_path,
                    ocr_config.server_url,
                    ocr_provider=ocr_config.ocr_provider,
                    provider_config=ocr_config,
                    provider_instance=provider,
                    force=task.force,
                    selected_item_ids=task.selected_item_ids_by_page.get(str(image_relative_path)),
                    timeout=ocr_config.timeout,
                    logger=bridge.message.emit,
                    progress_callback=on_item_progress,
                )
                ocr_data = load_ocr_json(json_path)
                summary = summarize_ocr_items(ocr_data.get("items", []))
            except Exception as exc:
                readable_error = f"{page_name}: {exc}"
                bridge.message.emit(f"OCR failed: {readable_error}")
                page_results.append(
                    OCRInferencePageResult(
                        image_relative_path=str(image_relative_path),
                        json_path=None,
                        error=str(exc),
                        summary={},
                    )
                )
                bridge.event.emit(
                    {
                        "stage": "ocr",
                        "event": "page_error",
                        "image_relative_path": str(image_relative_path),
                        "page_index": page_index,
                        "page_total": total_pages,
                        "message": str(exc),
                        "error": str(exc),
                    }
                )
            else:
                page_results.append(
                    OCRInferencePageResult(
                        image_relative_path=str(image_relative_path),
                        json_path=json_path,
                        error=None,
                        summary=summary,
                    )
                )
                bridge.event.emit(
                    {
                        "stage": "ocr",
                        "event": "page_done",
                        "image_relative_path": str(image_relative_path),
                        "page_index": page_index,
                        "page_total": total_pages,
                        "output_path": str(json_path),
                        "summary": summary,
                        "message": f"OCR complete for {page_name}",
                    }
                )
            finally:
                self._clear_provider_page_cache(
                    provider,
                    page_name=page_name,
                    logger=bridge.message.emit,
                )
            bridge.progress.emit(int((page_index / total_pages) * 100))

        successful_pages = [result for result in page_results if result.json_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown OCR inference failure."
            raise RuntimeError(f"OCR inference failed for all pages. {first_error}")
        return OCRInferenceWorkerResult(page_results=page_results)

    def _ensure_provider(self, config: OCRConfig) -> OCRProvider:
        provider_key = (
            str(config.ocr_provider or DEFAULT_OCR_PROVIDER),
            str(config.server_url or ""),
            float(config.timeout),
            bool(config.chrome_lens_headless),
            str(config.chrome_lens_chrome_path or ""),
            str(config.chrome_lens_user_data_dir or ""),
            str(config.chrome_lens_language or ""),
            int(config.chrome_lens_max_retries),
        )
        if self._provider is not None and self._provider_key == provider_key:
            return self._provider
        self._close_provider()
        self._provider = create_ocr_provider(config)
        self._provider_key = provider_key
        return self._provider

    def _clear_provider_page_cache(
        self,
        provider: OCRProvider,
        *,
        page_name: str,
        logger: Any,
    ) -> None:
        clear_after_page = getattr(provider, "clear_after_page", None)
        if not callable(clear_after_page):
            return

        try:
            result = clear_after_page()
        except Exception as exc:
            logger(f"DeepSeek OCR slot cache clear warning after page {page_name}: {exc}")
            return

        if not isinstance(result, dict):
            logger(f"DeepSeek OCR slot cache clear warning after page {page_name}: unexpected result.")
            return

        errors = result.get("errors")
        if not isinstance(errors, list):
            errors = []

        if not bool(result.get("supported", False)):
            if str(result.get("reason", "")).strip() == "unsupported":
                logger("llama.cpp /slots erase endpoint is unavailable; skipping DeepSeek OCR slot cache clear.")
            elif errors:
                logger(f"DeepSeek OCR slot cache clear warning after page {page_name}: {errors[0]}")
            return

        logger(
            "Cleared DeepSeek OCR llama.cpp slot cache after page "
            f"{page_name}: cleared={int(result.get('cleared', 0))}, "
            f"skipped_processing={int(result.get('skipped_processing', 0))}"
        )
        if errors:
            logger(f"DeepSeek OCR slot cache clear warning after page {page_name}: {errors[0]}")

    def _close_provider(self) -> None:
        if self._provider is None:
            self._provider_key = None
            return
        try:
            self._provider.close()
        finally:
            self._provider = None
            self._provider_key = None


__all__ = ["OCRService"]
