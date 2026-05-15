"""Resident process orchestration service that delegates to resident stage services."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mmt_core import OCRConfig, RenderConfig, TranslationConfig
from mmt_core.process_stage import (
    PROCESS_PIPELINE_STEPS,
    ProcessPageFailure,
    ProcessPipelineResult,
)
from mmt_gui.workers import (
    DetectionTask,
    InpaintMaskTask,
    InpaintTask,
    OCRInferenceTask,
    OCRPreparationTask,
    ProcessTask,
    RenderPreparationTask,
    RenderTask,
    TranslationInitializationTask,
    TranslationTask,
)

from .base_service import BaseService, ServiceCanceledError, WorkerSignalsBridge
from .models import ServiceCommand, ServiceCommandResult
from .resource_scheduler import ResourceScheduler


SyncDispatch = Callable[[Any, ServiceCommand], ServiceCommandResult]


class ProcessService(BaseService):
    def __init__(
        self,
        *,
        scheduler: ResourceScheduler,
        sync_dispatch: SyncDispatch,
        startup_options: dict | None = None,
    ) -> None:
        super().__init__("process", scheduler=scheduler, startup_options=startup_options)
        self._sync_dispatch = sync_dispatch

    def on_initialize(self) -> None:
        self._emit_log("info", "Process service is ready.")

    def execute_command(self, command: ServiceCommand, bridge: WorkerSignalsBridge) -> Any:
        task = command.task
        if not isinstance(task, ProcessTask):
            raise RuntimeError("Process service received an unexpected task type.")
        return self._run_process(command, task, bridge)

    def _run_process(
        self,
        command: ServiceCommand,
        task: ProcessTask,
        bridge: WorkerSignalsBridge,
    ) -> ProcessPipelineResult:
        if task.project is None:
            raise ValueError("No project was provided for one-click processing.")
        if not task.image_relative_paths:
            raise ValueError("No pages were provided for one-click processing.")

        ordered_paths = [str(Path(path).as_posix()) for path in task.image_relative_paths if str(path).strip()]
        if not ordered_paths:
            raise ValueError("No pages were provided for one-click processing.")

        normalized_scope = "current" if str(task.scope or "").strip().lower() == "current" else "chapter"
        step_statuses = {step.key: "pending" for step in PROCESS_PIPELINE_STEPS}
        result = ProcessPipelineResult(
            scope=normalized_scope,
            force=bool(task.force),
            image_relative_paths=list(ordered_paths),
            step_statuses=step_statuses,
        )

        bridge.message.emit(
            f"Starting one-click process for {len(ordered_paths)} page(s) "
            f"({'current page' if normalized_scope == 'current' else 'chapter'}, force={bool(task.force)}). "
            "Export is excluded."
        )
        bridge.progress.emit(0)

        failed_pages: dict[str, ProcessPageFailure] = {}
        total_pages = len(ordered_paths)
        total_steps = len(PROCESS_PIPELINE_STEPS)
        total_units = max(1, total_pages * total_steps)

        try:
            for step_index, step in enumerate(PROCESS_PIPELINE_STEPS, start=1):
                self._check_canceled(
                    command,
                    message=f"Canceled by user before {step.display_name} could start.",
                )
                result.current_stage = step.key
                result.unfinished_stage = step.key
                active_paths = [page for page in ordered_paths if page not in failed_pages]
                skipped_count = total_pages - len(active_paths)
                step_progress_base = (step_index - 1) * total_pages

                if not active_paths:
                    step_statuses[step.key] = "skipped"
                    payload = self._process_payload(
                        command,
                        step.key,
                        step.workflow_stage,
                        "process_stage_completed",
                        step_index=step_index,
                        step_total=total_steps,
                        page_total=total_pages,
                        overall_progress=self._progress_value(step_progress_base + total_pages, total_units),
                        status="skipped",
                        message=f"Skipped {step.display_name} because all remaining pages had already failed.",
                    )
                    bridge.event.emit(payload)
                    bridge.progress.emit(int(payload.get("overall_progress", 0) or 0))
                    result.completed_steps.append(step.key)
                    continue

                payload = self._process_payload(
                    command,
                    step.key,
                    step.workflow_stage,
                    "process_stage_started",
                    step_index=step_index,
                    step_total=total_steps,
                    page_total=total_pages,
                    active_page_total=len(active_paths),
                    overall_progress=self._progress_value(step_progress_base, total_units),
                    status="running",
                    message=f"{step.display_name} started.",
                )
                bridge.event.emit(payload)
                bridge.progress.emit(int(payload.get("overall_progress", 0) or 0))
                stage_had_error = False

                for page_offset, image_relative_path in enumerate(active_paths, start=1):
                    self._check_canceled(
                        command,
                        message=f"Canceled by user before {step.display_name} could continue.",
                    )
                    page_name = Path(image_relative_path).name
                    result.current_page = image_relative_path
                    payload = self._process_payload(
                        command,
                        step.key,
                        step.workflow_stage,
                        "page_start",
                        image_relative_path=image_relative_path,
                        step_index=step_index,
                        step_total=total_steps,
                        page_index=page_offset,
                        page_total=len(active_paths),
                        overall_progress=self._progress_value(
                            step_progress_base + skipped_count + page_offset - 1,
                            total_units,
                        ),
                        message=f"{step.display_name}: {page_name}",
                    )
                    bridge.event.emit(payload)
                    bridge.progress.emit(int(payload.get("overall_progress", 0) or 0))

                    stage_task = self._build_stage_task(task, step.key, image_relative_path)
                    try:
                        stage_result = self._sync_dispatch(stage_task, command)
                    except Exception as exc:
                        stage_had_error = True
                        result.last_error = str(exc)
                        failed_pages[image_relative_path] = ProcessPageFailure(
                            image_relative_path=image_relative_path,
                            process_stage=step.key,
                            error=str(exc),
                        )
                        payload = self._process_payload(
                            command,
                            step.key,
                            step.workflow_stage,
                            "page_error",
                            image_relative_path=image_relative_path,
                            step_index=step_index,
                            step_total=total_steps,
                            page_index=page_offset,
                            page_total=len(active_paths),
                            overall_progress=self._progress_value(
                                step_progress_base + skipped_count + page_offset,
                                total_units,
                            ),
                            message=str(exc),
                            error=str(exc),
                        )
                        bridge.event.emit(payload)
                        bridge.progress.emit(int(payload.get("overall_progress", 0) or 0))
                        if normalized_scope == "current":
                            result.stopped_early = True
                            result.page_failures = list(failed_pages.values())
                            step_statuses[step.key] = "error"
                            completed_payload = self._process_payload(
                                command,
                                step.key,
                                step.workflow_stage,
                                "process_stage_completed",
                                step_index=step_index,
                                step_total=total_steps,
                                page_total=total_pages,
                                overall_progress=self._progress_value(
                                    step_progress_base + skipped_count + page_offset,
                                    total_units,
                                ),
                                status="error",
                                message=f"{step.display_name} failed for {page_name}.",
                            )
                            bridge.event.emit(completed_payload)
                            bridge.progress.emit(int(completed_payload.get("overall_progress", 0) or 0))
                            return result
                        continue

                    self._check_canceled(
                        command,
                        message=f"Canceled by user after {step.display_name} finished the current page.",
                    )
                    result.last_completed_stage = step.display_name
                    result.last_completed_page = image_relative_path
                    page_event_name = "mask_ready" if step.key == "inpaint_mask" else "page_done"
                    payload = self._process_payload(
                        command,
                        step.key,
                        step.workflow_stage,
                        page_event_name,
                        image_relative_path=image_relative_path,
                        step_index=step_index,
                        step_total=total_steps,
                        page_index=page_offset,
                        page_total=len(active_paths),
                        overall_progress=self._progress_value(
                            step_progress_base + skipped_count + page_offset,
                            total_units,
                        ),
                        message=f"{step.display_name} complete for {page_name}.",
                        summary=self._summarize_stage_result(stage_result),
                    )
                    bridge.event.emit(payload)
                    bridge.progress.emit(int(payload.get("overall_progress", 0) or 0))

                step_statuses[step.key] = "error" if stage_had_error else "done"
                result.completed_steps.append(step.key)
                payload = self._process_payload(
                    command,
                    step.key,
                    step.workflow_stage,
                    "process_stage_completed",
                    step_index=step_index,
                    step_total=total_steps,
                    page_total=total_pages,
                    overall_progress=self._progress_value(step_index * total_pages, total_units),
                    status=step_statuses[step.key],
                    message=(
                        f"{step.display_name} completed with errors."
                        if stage_had_error
                        else f"{step.display_name} completed."
                    ),
                )
                bridge.event.emit(payload)
                bridge.progress.emit(int(payload.get("overall_progress", 0) or 0))
        except ServiceCanceledError as exc:
            result.canceled = True
            result.cancel_requested = True
            result.cancel_message = str(exc)
            result.page_failures = list(failed_pages.values())
            payload = self._process_payload(
                command,
                result.current_stage or "",
                self._workflow_stage_for_step(result.current_stage or ""),
                "process_canceled",
                image_relative_path=result.current_page or "",
                overall_progress=self._progress_value(
                    len(result.completed_steps) * total_pages,
                    total_units,
                ),
                message=str(exc),
                status="canceled",
            )
            bridge.event.emit(payload)
            bridge.progress.emit(int(payload.get("overall_progress", 0) or 0))
            return result

        result.page_failures = list(failed_pages.values())
        result.current_stage = ""
        result.unfinished_stage = ""
        result.current_page = ""
        return result

    def _build_stage_task(self, process_task: ProcessTask, step_key: str, image_relative_path: str) -> Any:
        project = process_task.project
        if step_key == "detection":
            return DetectionTask(
                name=f"Detection {Path(image_relative_path).name}",
                stage="detection",
                image_paths=[project.root_dir / image_relative_path],
                detection_cache_dir=project.cache_dir / "detection",
                masks_cache_dir=project.cache_dir / "masks",
                force=bool(process_task.force),
            )
        if step_key == "ocr_prepare":
            return OCRPreparationTask(
                name=f"OCR Prepare {Path(image_relative_path).name}",
                stage="ocr",
                project=project,
                image_relative_paths=[image_relative_path],
                force=bool(process_task.force),
                save_crops=True,
            )
        if step_key == "ocr":
            ocr_config = OCRConfig.from_value(process_task.ocr_config)
            return OCRInferenceTask(
                name=f"OCR {Path(image_relative_path).name}",
                stage="ocr",
                project=project,
                image_relative_paths=[image_relative_path],
                config=ocr_config.to_metadata(),
                server_url=ocr_config.server_url,
                force=bool(process_task.force),
                selected_item_ids_by_page={},
                timeout=float(ocr_config.timeout),
            )
        if step_key == "translation_init":
            return TranslationInitializationTask(
                name=f"Translation Init {Path(image_relative_path).name}",
                stage="translation",
                project=project,
                image_relative_paths=[image_relative_path],
                config=TranslationConfig.from_value(process_task.translation_config),
                force=bool(process_task.force),
            )
        if step_key == "translation":
            return TranslationTask(
                name=f"Translation {Path(image_relative_path).name}",
                stage="translation",
                project=project,
                image_relative_paths=[image_relative_path],
                config=TranslationConfig.from_value(process_task.translation_config),
                force=bool(process_task.force),
                selected_item_ids_by_page={},
            )
        if step_key == "inpaint_mask":
            settings = dict(process_task.inpaint_settings or {})
            return InpaintMaskTask(
                name=f"Inpaint Mask {Path(image_relative_path).name}",
                stage="inpaint",
                project=project,
                image_relative_paths=[image_relative_path],
                force=bool(settings.get("force", process_task.force)),
                mask_padding=int(settings.get("mask_padding", 0) or 0),
                use_bubble_mask=bool(settings.get("use_bubble_mask", True)),
            )
        if step_key == "inpaint":
            settings = dict(process_task.inpaint_settings or {})
            return InpaintTask(
                name=f"Inpaint {Path(image_relative_path).name}",
                stage="inpaint",
                project=project,
                image_relative_paths=[image_relative_path],
                force=bool(settings.get("force", process_task.force)),
                mask_padding=int(settings.get("mask_padding", 0) or 0),
                use_bubble_mask=bool(settings.get("use_bubble_mask", True)),
                use_crop_windows=bool(settings.get("use_crop_windows", True)),
                device=str(settings.get("device", "") or ""),
            )
        if step_key == "render_prepare":
            return RenderPreparationTask(
                name=f"Render Prepare {Path(image_relative_path).name}",
                stage="render",
                project=project,
                image_relative_paths=[image_relative_path],
                force=bool(process_task.force),
            )
        if step_key == "render":
            render_config = RenderConfig.from_value(process_task.render_config)
            render_config.force = bool(process_task.force)
            return RenderTask(
                name=f"Render {Path(image_relative_path).name}",
                stage="render",
                project=project,
                image_relative_paths=[image_relative_path],
                config=render_config.to_metadata(),
                force=render_config.force,
            )
        raise RuntimeError(f"Unsupported process step: {step_key}")

    def _summarize_stage_result(self, service_result: ServiceCommandResult) -> dict[str, Any]:
        result = service_result.result
        if result is None:
            return {}
        page_results = getattr(result, "page_results", None)
        if isinstance(page_results, list) and page_results:
            first_page_result = page_results[0]
            summary = getattr(first_page_result, "summary", None)
            if isinstance(summary, dict):
                return dict(summary)
            output_path = getattr(first_page_result, "json_path", None) or getattr(first_page_result, "image_path", None)
            if output_path is not None:
                return {"output_path": str(output_path)}
        manifest = getattr(result, "manifest", None)
        if isinstance(manifest, dict):
            return dict(manifest)
        return {}

    def _process_payload(
        self,
        command: ServiceCommand,
        process_stage: str,
        workflow_stage: str,
        event_name: str,
        *,
        image_relative_path: str = "",
        step_index: int = 0,
        step_total: int = 0,
        page_index: int = 0,
        page_total: int = 0,
        active_page_total: int = 0,
        overall_progress: int = 0,
        status: str = "",
        message: str = "",
        summary: dict[str, Any] | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        payload = {
            "command_id": command.command_id,
            "service_name": self.service_name,
            "action": command.action,
            "stage": "process",
            "event": event_name,
            "process_stage": process_stage,
            "workflow_stage": workflow_stage,
            "display_name": self._display_name_for_step(process_stage),
            "overall_progress": max(0, min(100, int(overall_progress))),
            "message": message,
        }
        if image_relative_path:
            payload["image_relative_path"] = image_relative_path
        if step_index:
            payload["step_index"] = int(step_index)
        if step_total:
            payload["step_total"] = int(step_total)
        if page_index:
            payload["page_index"] = int(page_index)
        if page_total:
            payload["page_total"] = int(page_total)
        if active_page_total:
            payload["active_page_total"] = int(active_page_total)
        if status:
            payload["status"] = status
        if summary:
            payload["summary"] = dict(summary)
        if error:
            payload["error"] = error
        return payload

    def _display_name_for_step(self, step_key: str) -> str:
        for step in PROCESS_PIPELINE_STEPS:
            if step.key == step_key:
                return step.display_name
        return str(step_key or "").replace("_", " ").title()

    def _workflow_stage_for_step(self, step_key: str) -> str:
        for step in PROCESS_PIPELINE_STEPS:
            if step.key == step_key:
                return step.workflow_stage
        return ""

    def _progress_value(self, completed_units: int, total_units: int) -> int:
        if total_units <= 0:
            return 0
        return max(0, min(100, int((completed_units / total_units) * 100)))


__all__ = ["ProcessService"]
