"""Resident inpaint service with thread-owned LaMa model lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mmt_core.inpaint_io import inpaint_json_path, load_inpaint_json, summarize_inpaint_json
from mmt_core.inpaint_stage import (
    DEFAULT_CROP_MARGIN,
    DEFAULT_CROP_TRIGGER_SIZE,
    DEFAULT_PAD_MOD,
    DEFAULT_RESIZE_LIMIT,
    LamaInpainterManager,
    load_lama_model,
    prepare_inpaint_mask_for_page,
    run_inpaint_for_page,
    unload_lama_model,
)
from mmt_gui.workers import (
    InpaintMaskPageResult,
    InpaintMaskTask,
    InpaintMaskWorkerResult,
    InpaintPageResult,
    InpaintTask,
    InpaintWorkerResult,
    LamaModelTask,
    LamaModelTaskResult,
)

from .base_service import BaseService, ServiceCanceledError, WorkerSignalsBridge
from .models import ServiceCommand
from .resource_scheduler import ResourceScheduler


class InpaintService(BaseService):
    def __init__(self, *, scheduler: ResourceScheduler | None = None, startup_options: dict | None = None) -> None:
        super().__init__("inpaint", scheduler=scheduler, startup_options=startup_options)
        self._manager = LamaInpainterManager()

    def on_initialize(self) -> None:
        if not bool(self.startup_options.get("preload_inpaint", True)):
            self._emit_status("error", "Inpaint preload is disabled. Enable startup preload or reload the service.")
            return
        self._emit_status("loading", "Preloading LaMa Manga model...")
        device = str(self.startup_options.get("device", "") or "")
        result = load_lama_model(
            device=device,
            crop_trigger_size=DEFAULT_CROP_TRIGGER_SIZE,
            crop_margin=DEFAULT_CROP_MARGIN,
            resize_limit=DEFAULT_RESIZE_LIMIT,
            pad_mod=DEFAULT_PAD_MOD,
            logger=lambda message: self._emit_log("info", message),
            manager=self._manager,
        )
        self._emit_log("info", "LaMa model loaded once at service startup.")
        self._emit_log("info", str(result.get("message", "") or "LaMa Manga model ready."))

    def on_shutdown(self) -> None:
        try:
            unload_lama_model(logger=lambda message: self._emit_log("info", message), manager=self._manager)
            self._emit_log("info", "LaMa unloaded on service shutdown.")
        except Exception as exc:
            self._emit_log("warning", f"Inpaint service shutdown cleanup failed: {exc}")

    def on_restart(self) -> None:
        self.on_shutdown()
        self.on_initialize()

    def execute_command(self, command: ServiceCommand, bridge: WorkerSignalsBridge) -> Any:
        task = command.task
        if isinstance(task, InpaintMaskTask):
            return self._run_mask_task(command, task, bridge)
        if isinstance(task, InpaintTask):
            return self._run_inpaint_task(command, task, bridge)
        if isinstance(task, LamaModelTask):
            return self._run_model_task(command, task, bridge)
        if command.action == "status":
            return self._manager.status()
        raise RuntimeError(f"Inpaint service does not support action '{command.action}'.")

    def _run_mask_task(
        self,
        command: ServiceCommand,
        task: InpaintMaskTask,
        bridge: WorkerSignalsBridge,
    ) -> InpaintMaskWorkerResult:
        if task.project is None:
            raise ValueError("No project was provided for inpaint mask preparation.")
        if not task.image_relative_paths:
            raise ValueError("No pages were provided for inpaint mask preparation.")

        total_pages = len(task.image_relative_paths)
        page_results: list[InpaintMaskPageResult] = []
        bridge.message.emit(f"Preparing inpaint masks for {total_pages} page(s).")
        bridge.progress.emit(0)

        for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
            self._check_canceled(command, message="Inpaint mask preparation canceled before the next page.")
            page_name = Path(image_relative_path).name
            bridge.event.emit(
                {
                    "event": "page_start",
                    "image_relative_path": str(image_relative_path),
                    "page_index": index,
                    "page_total": total_pages,
                }
            )
            bridge.message.emit(f"[{index}/{total_pages}] Preparing mask for {page_name}")
            try:
                mask_path = prepare_inpaint_mask_for_page(
                    task.project,
                    image_relative_path,
                    mask_padding=task.mask_padding,
                    use_bubble_mask=task.use_bubble_mask,
                    force=task.force,
                    logger=bridge.message.emit,
                )
                metadata = load_inpaint_json(inpaint_json_path(task.project, image_relative_path))
                summary = summarize_inpaint_json(metadata)
            except Exception as exc:
                readable_error = f"{page_name}: {exc}"
                bridge.message.emit(f"Inpaint mask preparation failed: {readable_error}")
                page_results.append(
                    InpaintMaskPageResult(
                        image_relative_path=str(image_relative_path),
                        mask_path=None,
                        error=str(exc),
                        summary={},
                    )
                )
                bridge.event.emit(
                    {
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
                    InpaintMaskPageResult(
                        image_relative_path=str(image_relative_path),
                        mask_path=mask_path,
                        error=None,
                        summary=summary,
                    )
                )
                bridge.message.emit(f"Inpaint mask ready: {mask_path}")
                bridge.event.emit(
                    {
                        "event": "mask_ready",
                        "image_relative_path": str(image_relative_path),
                        "page_index": index,
                        "page_total": total_pages,
                        "output_path": str(mask_path),
                        "summary": summary,
                        "message": f"Inpaint mask ready for {page_name}",
                    }
                )
            bridge.progress.emit(int((index / total_pages) * 100))

        successful_pages = [result for result in page_results if result.mask_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown inpaint mask failure."
            raise RuntimeError(f"Inpaint mask preparation failed for all pages. {first_error}")

        failed_count = len([result for result in page_results if result.error is not None])
        if failed_count:
            bridge.message.emit(
                f"Inpaint mask preparation finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
            )
        else:
            bridge.message.emit(
                f"Inpaint mask preparation finished successfully for {len(successful_pages)} page(s)."
            )
        return InpaintMaskWorkerResult(page_results=page_results)

    def _run_inpaint_task(
        self,
        command: ServiceCommand,
        task: InpaintTask,
        bridge: WorkerSignalsBridge,
    ) -> InpaintWorkerResult:
        if task.project is None:
            raise ValueError("No project was provided for inpainting.")
        if not task.image_relative_paths:
            raise ValueError("No pages were provided for inpainting.")

        total_pages = len(task.image_relative_paths)
        page_results: list[InpaintPageResult] = []
        bridge.message.emit(f"Starting inpaint for {total_pages} page(s).")
        bridge.progress.emit(0)
        model_status = self._manager.status()
        bridge.message.emit(
            f"Using resident LaMa model on {str(model_status.get('device', '') or 'auto')} "
            f"(reload_count={int(model_status.get('reload_count', 0) or 0)})."
        )

        for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
            self._check_canceled(command, message="Inpaint canceled before the next page.")
            page_name = Path(image_relative_path).name
            bridge.event.emit(
                {
                    "event": "batch_page_start" if total_pages > 1 else "page_start",
                    "image_relative_path": str(image_relative_path),
                    "page_index": index,
                    "page_total": total_pages,
                    "message": f"[{index}/{total_pages}] Inpainting {page_name}",
                }
            )
            try:
                image_path = run_inpaint_for_page(
                    task.project,
                    image_relative_path,
                    force=task.force,
                    mask_padding=task.mask_padding,
                    use_bubble_mask=task.use_bubble_mask,
                    use_crop_windows=task.use_crop_windows,
                    device=task.device,
                    logger=bridge.message.emit,
                    progress_callback=bridge.event.emit,
                    manager=self._manager,
                    require_loaded_model=True,
                )
                metadata = load_inpaint_json(inpaint_json_path(task.project, image_relative_path))
                summary = summarize_inpaint_json(metadata)
            except Exception as exc:
                readable_error = f"{page_name}: {exc}"
                bridge.message.emit(f"Inpaint failed: {readable_error}")
                page_results.append(
                    InpaintPageResult(
                        image_relative_path=str(image_relative_path),
                        image_path=None,
                        error=str(exc),
                        summary={},
                    )
                )
                bridge.event.emit(
                    {
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
                    InpaintPageResult(
                        image_relative_path=str(image_relative_path),
                        image_path=image_path,
                        error=None,
                        summary=summary,
                    )
                )
            bridge.progress.emit(int((index / total_pages) * 100))

        successful_pages = [result for result in page_results if result.image_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown inpaint failure."
            raise RuntimeError(f"Inpaint failed for all pages. {first_error}")

        failed_count = len([result for result in page_results if result.error is not None])
        if failed_count:
            bridge.message.emit(
                f"Inpaint finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
            )
        else:
            bridge.message.emit(f"Inpaint finished successfully for {len(successful_pages)} page(s).")
        return InpaintWorkerResult(page_results=page_results)

    def _run_model_task(
        self,
        command: ServiceCommand,
        task: LamaModelTask,
        bridge: WorkerSignalsBridge,
    ) -> LamaModelTaskResult:
        action = str(task.action or "").strip().lower()
        if action in {"load", "reload"}:
            if action == "reload":
                self._emit_log("info", "LaMa reload requested explicitly.", command=command)
            result = load_lama_model(
                device=task.device,
                crop_trigger_size=DEFAULT_CROP_TRIGGER_SIZE,
                crop_margin=DEFAULT_CROP_MARGIN,
                resize_limit=DEFAULT_RESIZE_LIMIT,
                pad_mod=DEFAULT_PAD_MOD,
                explicit_reload=(action == "reload"),
                logger=bridge.message.emit,
                manager=self._manager,
            )
        elif action == "unload":
            result = unload_lama_model(logger=bridge.message.emit, manager=self._manager)
        else:
            raise RuntimeError(f"Unsupported LaMa model action: {task.action}")

        return LamaModelTaskResult(
            loaded=bool(result.get("loaded", False)),
            device=str(result.get("device", "") or ""),
            message=str(result.get("message", "") or ""),
        )


__all__ = ["InpaintService"]
