"""One-click process orchestration for Detection through Render."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .detection_config import DetectionConfig
from .detection_stage import run_detection_for_image
from .inpaint_stage import prepare_inpaint_mask_for_page, run_inpaint_for_page
from .ocr_models import OCRConfig
from .ocr_providers import create_ocr_provider, validate_ocr_provider_config
from .ocr_stage import prepare_ocr_items_for_image, run_ocr_for_page
from .render_models import RenderConfig
from .render_stage import prepare_render_for_page, run_render_for_page
from .translation_models import TranslationConfig
from .translation_stage import (
    initialize_translation_for_page,
    run_translation_for_page,
    run_translation_for_pages,
    validate_translation_config,
)

Logger = Callable[[str], None]
ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


class ProcessCanceledError(RuntimeError):
    """Raised when one-click processing is canceled at a safe point."""


@dataclass(slots=True, frozen=True)
class ProcessStepDefinition:
    """Describes one orchestration step in the one-click process pipeline."""

    key: str
    display_name: str
    workflow_stage: str


PROCESS_PIPELINE_STEPS: tuple[ProcessStepDefinition, ...] = (
    ProcessStepDefinition("detection", "Detection", "detection"),
    ProcessStepDefinition("ocr_prepare", "OCR Prepare", "ocr"),
    ProcessStepDefinition("ocr", "OCR", "ocr"),
    ProcessStepDefinition("translation_init", "Translation Init", "translation"),
    ProcessStepDefinition("translation", "Translation", "translation"),
    ProcessStepDefinition("inpaint_mask", "Mask Prepare", "inpaint"),
    ProcessStepDefinition("inpaint", "Inpaint", "inpaint"),
    ProcessStepDefinition("render_prepare", "Render Prepare", "render"),
    ProcessStepDefinition("render", "Render", "render"),
)
PROCESS_STEP_KEYS: tuple[str, ...] = tuple(step.key for step in PROCESS_PIPELINE_STEPS)
PROCESS_WORKFLOW_STAGE_BY_STEP: dict[str, str] = {
    step.key: step.workflow_stage for step in PROCESS_PIPELINE_STEPS
}


@dataclass(slots=True)
class ProcessPageFailure:
    """Captures one page-specific process failure."""

    image_relative_path: str
    process_stage: str
    error: str


@dataclass(slots=True)
class ProcessPipelineResult:
    """Aggregated outcome for a one-click process run."""

    scope: str
    force: bool
    image_relative_paths: list[str]
    step_statuses: dict[str, str]
    page_failures: list[ProcessPageFailure] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    stopped_early: bool = False
    fatal_error: bool = False
    last_error: str = ""
    canceled: bool = False
    cancel_requested: bool = False
    cancel_message: str = ""
    current_stage: str = ""
    unfinished_stage: str = ""
    current_page: str = ""
    last_completed_stage: str = ""
    last_completed_page: str = ""

    @property
    def pages_processed(self) -> int:
        return len(self.image_relative_paths)

    @property
    def failed_pages(self) -> int:
        return len(self.page_failures)

    @property
    def succeeded_pages(self) -> int:
        return max(0, self.pages_processed - self.failed_pages)

    @property
    def stages_completed(self) -> int:
        return len([status for status in self.step_statuses.values() if status in {"done", "error"}])

    @property
    def completed_with_errors(self) -> bool:
        return self.final_state == "completed_with_errors"

    @property
    def final_state(self) -> str:
        if self.canceled:
            return "canceled"
        if self.fatal_error or (self.stopped_early and self.scope == "current"):
            return "failed"
        if self.page_failures or self.stopped_early:
            return "completed_with_errors"
        return "completed"


def run_process_pipeline(
    project,
    image_relative_paths: Sequence[str | Path],
    *,
    scope: str = "chapter",
    force: bool = False,
    detection_config: DetectionConfig | dict[str, Any] | None = None,
    ocr_config: OCRConfig | dict[str, Any] | None = None,
    translation_config: TranslationConfig | dict[str, Any] | None = None,
    inpaint_settings: dict[str, Any] | None = None,
    render_config: RenderConfig | dict[str, Any] | None = None,
    logger: Logger | None = None,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> ProcessPipelineResult:
    """Run Detection -> OCR -> Translation -> Inpaint -> Render without export."""

    if project is None:
        raise ValueError("No project was provided for processing.")

    ordered_paths = [
        str(Path(str(image_relative_path)).as_posix())
        for image_relative_path in image_relative_paths
        if str(image_relative_path or "").strip()
    ]
    if not ordered_paths:
        raise ValueError("No pages were provided for processing.")

    normalized_scope = "current" if str(scope or "").strip().lower() == "current" else "chapter"
    continue_on_page_error = normalized_scope == "chapter"

    normalized_detection_config = DetectionConfig.from_value(detection_config)
    normalized_ocr_config = validate_ocr_provider_config(ocr_config)
    normalized_translation_config = validate_translation_config(translation_config, logger=logger)
    normalized_inpaint_settings = _normalize_inpaint_settings(inpaint_settings, force=force)
    normalized_render_config = RenderConfig.from_value(render_config)
    normalized_render_config.force = bool(force)

    ocr_provider = create_ocr_provider(normalized_ocr_config)
    try:
        ocr_provider.validate()
    except Exception as exc:
        ocr_provider.close()
        provider_label = normalized_ocr_config.provider_label
        if normalized_ocr_config.requires_llama_server:
            raise RuntimeError(
                f"{provider_label} server is not reachable. Start/check it from OCR tab first. "
                f"Details: {exc}"
            ) from exc
        raise RuntimeError(f"{provider_label} is unavailable: {exc}") from exc
    finally:
        try:
            ocr_provider.close()
        except Exception:
            pass

    step_statuses = {step.key: "pending" for step in PROCESS_PIPELINE_STEPS}
    result = ProcessPipelineResult(
        scope=normalized_scope,
        force=bool(force),
        image_relative_paths=list(ordered_paths),
        step_statuses=step_statuses,
    )
    failed_pages: dict[str, ProcessPageFailure] = {}
    total_pages = len(ordered_paths)
    total_units = max(1, total_pages * len(PROCESS_PIPELINE_STEPS))
    completed_units = 0

    _raise_if_cancel_requested(should_cancel)

    _log(
        logger,
        f"Starting {'chapter' if normalized_scope == 'chapter' else 'current page'} process for "
        f"{total_pages} page(s). Export is intentionally excluded.",
    )

    for step_index, step in enumerate(PROCESS_PIPELINE_STEPS, start=1):
        result.current_stage = step.key
        result.unfinished_stage = step.key
        active_paths = [page_path for page_path in ordered_paths if page_path not in failed_pages]
        skipped_for_step = total_pages - len(active_paths)

        try:
            _raise_if_cancel_requested(should_cancel)
        except ProcessCanceledError:
            return _finalize_canceled_result(
                result,
                step=step,
                step_index=step_index,
                total_pages=total_pages,
                active_page_total=len(active_paths),
                completed_units=completed_units,
                total_units=total_units,
                progress_callback=progress_callback,
                logger=logger,
                image_relative_path=result.current_page or result.last_completed_page,
            )

        if not active_paths:
            step_statuses[step.key] = "skipped"
            completed_units += total_pages
            _emit(
                progress_callback,
                {
                    "event": "process_stage_completed",
                    "process_stage": step.key,
                    "workflow_stage": step.workflow_stage,
                    "display_name": step.display_name,
                    "status": "skipped",
                    "step_index": step_index,
                    "step_total": len(PROCESS_PIPELINE_STEPS),
                    "page_total": total_pages,
                    "message": f"Skipped {step.display_name} because all remaining pages had already failed.",
                    "overall_progress": _progress_value(completed_units, total_units),
                },
            )
            continue

        _log(logger, f"Process step started: {step.display_name}")
        _emit(
            progress_callback,
            {
                "event": "process_stage_started",
                "process_stage": step.key,
                "workflow_stage": step.workflow_stage,
                "display_name": step.display_name,
                "status": "running",
                "step_index": step_index,
                "step_total": len(PROCESS_PIPELINE_STEPS),
                "page_total": total_pages,
                "active_page_total": len(active_paths),
                "message": f"{step.display_name} started.",
                "overall_progress": _progress_value(completed_units, total_units),
            },
        )

        stage_had_error = False
        stage_canceled = False
        last_completed_page = ""

        if step.key == "translation" and normalized_scope == "chapter" and len(active_paths) > 1:
            stage_had_error, completed_units, stage_canceled, last_completed_page = _run_translation_batch_step(
                project,
                active_paths,
                normalized_translation_config,
                force=bool(force),
                logger=logger,
                progress_callback=progress_callback,
                total_units=total_units,
                completed_units=completed_units,
                failed_pages=failed_pages,
                should_cancel=should_cancel,
            )
        else:
            stage_had_error, completed_units, stop_current, stage_canceled, last_completed_page = _run_sequential_step(
                project,
                step,
                active_paths,
                force=bool(force),
                detection_config=normalized_detection_config,
                ocr_config=normalized_ocr_config,
                translation_config=normalized_translation_config,
                inpaint_settings=normalized_inpaint_settings,
                render_config=normalized_render_config,
                logger=logger,
                progress_callback=progress_callback,
                total_units=total_units,
                completed_units=completed_units,
                failed_pages=failed_pages,
                stop_on_error=not continue_on_page_error,
                should_cancel=should_cancel,
            )
            if last_completed_page:
                result.last_completed_page = last_completed_page
            if stage_canceled:
                step_statuses[step.key] = "canceled"
                return _finalize_canceled_result(
                    result,
                    step=step,
                    step_index=step_index,
                    total_pages=total_pages,
                    active_page_total=len(active_paths),
                    completed_units=completed_units,
                    total_units=total_units,
                    progress_callback=progress_callback,
                    logger=logger,
                    image_relative_path=result.current_page or result.last_completed_page,
                )
            if stop_current:
                stage_had_error = True
                step_statuses[step.key] = "error"
                result.completed_steps.append(step.key)
                completed_units += skipped_for_step
                _emit(
                    progress_callback,
                    {
                        "event": "process_stage_completed",
                        "process_stage": step.key,
                        "workflow_stage": step.workflow_stage,
                        "display_name": step.display_name,
                        "status": "error",
                        "step_index": step_index,
                        "step_total": len(PROCESS_PIPELINE_STEPS),
                        "page_total": total_pages,
                        "active_page_total": len(active_paths),
                        "message": f"{step.display_name} stopped after an error.",
                        "overall_progress": _progress_value(completed_units, total_units),
                    },
                )
                for remaining_step in PROCESS_PIPELINE_STEPS[step_index:]:
                    step_statuses[remaining_step.key] = "skipped"
                result.page_failures = list(failed_pages.values())
                result.stopped_early = True
                result.last_error = result.page_failures[-1].error if result.page_failures else ""
                return result
        if last_completed_page:
            result.last_completed_page = last_completed_page
        if stage_canceled:
            step_statuses[step.key] = "canceled"
            return _finalize_canceled_result(
                result,
                step=step,
                step_index=step_index,
                total_pages=total_pages,
                active_page_total=len(active_paths),
                completed_units=completed_units,
                total_units=total_units,
                progress_callback=progress_callback,
                logger=logger,
                image_relative_path=result.current_page or result.last_completed_page,
            )

        completed_units += skipped_for_step
        step_statuses[step.key] = "error" if stage_had_error else "done"
        result.completed_steps.append(step.key)
        result.last_completed_stage = step.key
        result.current_stage = ""
        result.unfinished_stage = ""
        _emit(
            progress_callback,
            {
                "event": "process_stage_completed",
                "process_stage": step.key,
                "workflow_stage": step.workflow_stage,
                "display_name": step.display_name,
                "status": step_statuses[step.key],
                "step_index": step_index,
                "step_total": len(PROCESS_PIPELINE_STEPS),
                "page_total": total_pages,
                "active_page_total": len(active_paths),
                "message": (
                    f"{step.display_name} completed with page errors."
                    if stage_had_error
                    else f"{step.display_name} completed."
                ),
                "overall_progress": _progress_value(completed_units, total_units),
            },
        )

    result.page_failures = list(failed_pages.values())
    if result.page_failures:
        result.last_error = result.page_failures[-1].error
        _log(
            logger,
            f"Process completed with {result.succeeded_pages} success(es) and {result.failed_pages} failed page(s).",
        )
    else:
        _log(logger, f"Process completed successfully for {result.succeeded_pages} page(s).")
    return result


def _run_sequential_step(
    project,
    step: ProcessStepDefinition,
    active_paths: list[str],
    *,
    force: bool,
    detection_config: DetectionConfig,
    ocr_config: OCRConfig,
    translation_config: TranslationConfig,
    inpaint_settings: dict[str, Any],
    render_config: RenderConfig,
    logger: Logger | None,
    progress_callback: ProgressCallback | None,
    total_units: int,
    completed_units: int,
    failed_pages: dict[str, ProcessPageFailure],
    stop_on_error: bool,
    should_cancel: CancelCheck | None,
) -> tuple[bool, int, bool, bool, str]:
    stage_had_error = False
    last_completed_page = ""

    ocr_provider = None
    if step.key == "ocr":
        ocr_provider = create_ocr_provider(ocr_config)
        ocr_provider.validate()

    try:
        for page_index, image_relative_path in enumerate(active_paths, start=1):
            try:
                _raise_if_cancel_requested(should_cancel)
            except ProcessCanceledError:
                return stage_had_error, completed_units, False, True, last_completed_page
            page_name = Path(image_relative_path).name
            _log(logger, f"{step.display_name}: {page_name}")
            _emit(
                progress_callback,
                {
                    "event": "page_start",
                    "process_stage": step.key,
                    "workflow_stage": step.workflow_stage,
                    "display_name": step.display_name,
                    "image_relative_path": image_relative_path,
                    "page_index": page_index,
                    "page_total": len(active_paths),
                    "message": f"{step.display_name}: {page_name}",
                    "overall_progress": _progress_value(completed_units, total_units),
                },
            )
            try:
                page_result = _run_step_for_page(
                    project,
                    step.key,
                    image_relative_path,
                    force=force,
                    detection_config=detection_config,
                    ocr_config=ocr_config,
                    ocr_provider=ocr_provider,
                    translation_config=translation_config,
                    inpaint_settings=inpaint_settings,
                    render_config=render_config,
                    logger=logger,
                    progress_callback=progress_callback,
                    should_cancel=should_cancel,
                )
            except ProcessCanceledError:
                return stage_had_error, completed_units, False, True, last_completed_page
            except Exception as exc:
                failure = ProcessPageFailure(
                    image_relative_path=image_relative_path,
                    process_stage=step.key,
                    error=str(exc),
                )
                failed_pages[image_relative_path] = failure
                stage_had_error = True
                completed_units += 1
                _emit(
                    progress_callback,
                    {
                        "event": "page_error",
                        "process_stage": step.key,
                        "workflow_stage": step.workflow_stage,
                        "display_name": step.display_name,
                        "image_relative_path": image_relative_path,
                        "page_index": page_index,
                        "page_total": len(active_paths),
                        "error": str(exc),
                        "message": f"{step.display_name} failed for {page_name}: {exc}",
                        "overall_progress": _progress_value(completed_units, total_units),
                    },
                )
                _log(logger, f"{step.display_name} failed for {page_name}: {exc}")
                if stop_on_error:
                    return stage_had_error, completed_units, True, False, last_completed_page
                continue

            completed_units += 1
            last_completed_page = image_relative_path
            payload = {
                "event": "page_done",
                "process_stage": step.key,
                "workflow_stage": step.workflow_stage,
                "display_name": step.display_name,
                "image_relative_path": image_relative_path,
                "page_index": page_index,
                "page_total": len(active_paths),
                "message": f"{step.display_name} complete for {page_name}",
                "overall_progress": _progress_value(completed_units, total_units),
            }
            payload.update(page_result)
            _emit(progress_callback, payload)
            if page_index < len(active_paths):
                try:
                    _raise_if_cancel_requested(should_cancel)
                except ProcessCanceledError:
                    return stage_had_error, completed_units, False, True, last_completed_page
    finally:
        if ocr_provider is not None:
            ocr_provider.close()

    return stage_had_error, completed_units, False, False, last_completed_page


def _run_translation_batch_step(
    project,
    active_paths: list[str],
    translation_config: TranslationConfig,
    *,
    force: bool,
    logger: Logger | None,
    progress_callback: ProgressCallback | None,
    total_units: int,
    completed_units: int,
    failed_pages: dict[str, ProcessPageFailure],
    should_cancel: CancelCheck | None,
) -> tuple[bool, int, bool, str]:
    stage_had_error = False
    processed_paths: set[str] = set()
    last_completed_page = ""
    page_indices = {page_path: index for index, page_path in enumerate(active_paths, start=1)}

    def on_progress(event: dict[str, Any]) -> None:
        nonlocal completed_units, last_completed_page
        payload = dict(event)
        payload["process_stage"] = "translation"
        payload["workflow_stage"] = "translation"
        payload["display_name"] = "Translation"
        image_relative_path = str(payload.get("image_relative_path", "") or "").strip()
        if image_relative_path:
            payload.setdefault("page_index", page_indices.get(image_relative_path, 1))
            payload.setdefault("page_total", len(active_paths))

        event_name = str(payload.get("event", "") or "").strip().lower()
        if image_relative_path and event_name in {"page_done", "page_error"} and image_relative_path not in processed_paths:
            processed_paths.add(image_relative_path)
            completed_units += 1
            if event_name == "page_done":
                last_completed_page = image_relative_path
            payload["overall_progress"] = _progress_value(completed_units, total_units)
            if event_name == "page_error":
                failed_pages[image_relative_path] = ProcessPageFailure(
                    image_relative_path=image_relative_path,
                    process_stage="translation",
                    error=str(payload.get("message", "") or "Translation failed."),
                )
        elif "overall_progress" not in payload:
            payload["overall_progress"] = _progress_value(completed_units, total_units)
        _emit(progress_callback, payload)
        if event_name in {"page_done", "page_error"}:
            _raise_if_cancel_requested(should_cancel)

    try:
        _raise_if_cancel_requested(should_cancel)
    except ProcessCanceledError:
        return stage_had_error, completed_units, True, last_completed_page

    try:
        run_translation_for_pages(
            project,
            active_paths,
            translation_config,
            force=force,
            logger=logger,
            progress_callback=on_progress,
        )
    except ProcessCanceledError:
        return stage_had_error, completed_units, True, last_completed_page
    except Exception as exc:
        stage_had_error = True
        unprocessed_paths = [page_path for page_path in active_paths if page_path not in processed_paths]
        for image_relative_path in unprocessed_paths:
            completed_units += 1
            failed_pages[image_relative_path] = ProcessPageFailure(
                image_relative_path=image_relative_path,
                process_stage="translation",
                error=str(exc),
            )
            _emit(
                progress_callback,
                {
                    "event": "page_error",
                    "process_stage": "translation",
                    "workflow_stage": "translation",
                    "display_name": "Translation",
                    "image_relative_path": image_relative_path,
                    "page_index": page_indices.get(image_relative_path, 1),
                    "page_total": len(active_paths),
                    "error": str(exc),
                    "message": f"Translation failed for {Path(image_relative_path).name}: {exc}",
                    "overall_progress": _progress_value(completed_units, total_units),
                },
            )
    else:
        if any(page_path in failed_pages for page_path in active_paths):
            stage_had_error = True

    return stage_had_error, completed_units, False, last_completed_page


def _run_step_for_page(
    project,
    process_stage: str,
    image_relative_path: str,
    *,
    force: bool,
    detection_config: DetectionConfig,
    ocr_config: OCRConfig,
    ocr_provider,
    translation_config: TranslationConfig,
    inpaint_settings: dict[str, Any],
    render_config: RenderConfig,
    logger: Logger | None,
    progress_callback: ProgressCallback | None,
    should_cancel: CancelCheck | None,
) -> dict[str, Any]:
    if process_stage == "detection":
        image_path = project.root_dir / image_relative_path
        output_path = run_detection_for_image(
            image_path,
            project.cache_dir / "detection",
            project.cache_dir / "masks",
            force=force,
            detection_config=detection_config,
            logger=logger,
        )
        return {"output_path": str(output_path)}

    if process_stage == "ocr_prepare":
        output_path = prepare_ocr_items_for_image(
            project,
            image_relative_path,
            force=force,
            save_crops=True,
            logger=logger,
        )
        return {"output_path": str(output_path)}

    if process_stage == "ocr":
        output_path = run_ocr_for_page(
            project,
            image_relative_path,
            ocr_config.server_url,
            ocr_provider=ocr_config.ocr_provider,
            provider_config=ocr_config,
            provider_instance=ocr_provider,
            force=force,
            selected_item_ids=None,
            timeout=ocr_config.timeout,
            logger=logger,
            progress_callback=lambda current, total, item_info: _emit_nested_ocr_progress(
                progress_callback,
                should_cancel=should_cancel,
                image_relative_path=image_relative_path,
                current=current,
                total=total,
                item_info=item_info,
            ),
        )
        return {"output_path": str(output_path)}

    if process_stage == "translation_init":
        output_path = initialize_translation_for_page(
            project,
            image_relative_path,
            translation_config,
            force=force,
            logger=logger,
        )
        return {"output_path": str(output_path)}

    if process_stage == "translation":
        output_path = run_translation_for_page(
            project,
            image_relative_path,
            translation_config,
            force=force,
            selected_item_ids=None,
            logger=logger,
            progress_callback=lambda payload: _relay_nested_process_event(
                progress_callback,
                payload,
                process_stage="translation",
                workflow_stage="translation",
                display_name="Translation",
                skip_page_events=True,
                should_cancel=should_cancel,
                cancel_event_names={"item_done"},
            ),
        )
        return {"output_path": str(output_path)}

    if process_stage == "inpaint_mask":
        mask_path = prepare_inpaint_mask_for_page(
            project,
            image_relative_path,
            mask_padding=int(inpaint_settings["mask_padding"]),
            use_bubble_mask=bool(inpaint_settings["use_bubble_mask"]),
            force=force,
            logger=logger,
        )
        return {"output_path": str(mask_path)}

    if process_stage == "inpaint":
        image_path = run_inpaint_for_page(
            project,
            image_relative_path,
            force=force,
            mask_padding=int(inpaint_settings["mask_padding"]),
            use_bubble_mask=bool(inpaint_settings["use_bubble_mask"]),
            use_crop_windows=bool(inpaint_settings["use_crop_windows"]),
            device=inpaint_settings["device"],
            logger=logger,
            progress_callback=lambda payload: _relay_nested_process_event(
                progress_callback,
                payload,
                process_stage="inpaint",
                workflow_stage="inpaint",
                display_name="Inpaint",
                skip_page_events=True,
                should_cancel=None,
                cancel_event_names=set(),
            ),
        )
        return {"output_path": str(image_path)}

    if process_stage == "render_prepare":
        output_path = prepare_render_for_page(
            project,
            image_relative_path,
            force=force,
            logger=logger,
        )
        return {"output_path": str(output_path)}

    if process_stage == "render":
        image_path = run_render_for_page(
            project,
            image_relative_path,
            force=force,
            font_name=render_config.font_name,
            font_path=render_config.font_path,
            min_font_size=int(render_config.min_font_size),
            max_font_size=int(render_config.max_font_size),
            stroke_enabled=bool(render_config.stroke_enabled),
            stroke_width=render_config.stroke_width,
            text_color=render_config.text_color,
            stroke_color=render_config.stroke_color,
            auto_color=bool(render_config.auto_color),
            auto_direction=bool(render_config.auto_direction),
            vertical_cjk=bool(render_config.vertical_cjk),
            save_sprites=bool(render_config.save_sprites),
            logger=logger,
            progress_callback=lambda payload: _relay_nested_process_event(
                progress_callback,
                payload,
                process_stage="render",
                workflow_stage="render",
                display_name="Render",
                skip_page_events=True,
                should_cancel=None,
                cancel_event_names=set(),
            ),
        )
        return {"output_path": str(image_path)}

    raise ValueError(f"Unsupported process stage: {process_stage}")


def _normalize_inpaint_settings(
    inpaint_settings: dict[str, Any] | None,
    *,
    force: bool,
) -> dict[str, Any]:
    payload = dict(inpaint_settings or {})
    return {
        "mask_padding": int(payload.get("mask_padding", 0) or 0),
        "use_bubble_mask": bool(payload.get("use_bubble_mask", True)),
        "use_crop_windows": bool(payload.get("use_crop_windows", True)),
        "device": payload.get("device"),
        "force": bool(force),
    }


def _progress_value(completed_units: int, total_units: int) -> int:
    if total_units <= 0:
        return 0
    return max(0, min(100, int((completed_units / total_units) * 100)))


def _emit(callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback is not None:
        callback(payload)


def _emit_nested_ocr_progress(
    callback: ProgressCallback | None,
    *,
    should_cancel: CancelCheck | None,
    image_relative_path: str,
    current: int,
    total: int,
    item_info: dict[str, Any],
) -> None:
    _emit(
        callback,
        {
            "event": "item_done",
            "process_stage": "ocr",
            "workflow_stage": "ocr",
            "display_name": "OCR",
            "image_relative_path": image_relative_path,
            "item_index": int(current),
            "item_total": int(total),
            "message": str(item_info.get("message", "") or "").strip(),
        },
    )
    _raise_if_cancel_requested(should_cancel)


def _relay_nested_process_event(
    callback: ProgressCallback | None,
    payload: dict[str, Any],
    *,
    process_stage: str,
    workflow_stage: str,
    display_name: str,
    skip_page_events: bool,
    should_cancel: CancelCheck | None,
    cancel_event_names: set[str],
) -> None:
    normalized_payload = dict(payload)
    event_name = str(normalized_payload.get("event", "") or "").strip().lower()
    if skip_page_events and event_name in {"page_start", "page_done", "page_error"}:
        return
    normalized_payload["process_stage"] = process_stage
    normalized_payload["workflow_stage"] = workflow_stage
    normalized_payload["display_name"] = display_name
    _emit(callback, normalized_payload)
    if event_name in cancel_event_names:
        _raise_if_cancel_requested(should_cancel)


def _raise_if_cancel_requested(should_cancel: CancelCheck | None) -> None:
    if should_cancel is None:
        return
    try:
        is_canceled = bool(should_cancel())
    except Exception:
        is_canceled = False
    if is_canceled:
        raise ProcessCanceledError("Process canceled by user.")


def _finalize_canceled_result(
    result: ProcessPipelineResult,
    *,
    step: ProcessStepDefinition,
    step_index: int,
    total_pages: int,
    active_page_total: int,
    completed_units: int,
    total_units: int,
    progress_callback: ProgressCallback | None,
    logger: Logger | None,
    image_relative_path: str | None,
) -> ProcessPipelineResult:
    result.canceled = True
    result.cancel_requested = True
    result.cancel_message = "Process canceled by user."
    result.current_stage = ""
    result.unfinished_stage = step.key
    result.current_page = str(image_relative_path or result.current_page or "")
    result.step_statuses[step.key] = "canceled"
    for remaining_step in PROCESS_PIPELINE_STEPS[step_index:]:
        if remaining_step.key == step.key:
            continue
        if result.step_statuses.get(remaining_step.key, "pending") == "pending":
            result.step_statuses[remaining_step.key] = "skipped"
    _log(
        logger,
        f"Process canceled by user at {step.display_name}"
        + (f" for {Path(result.current_page).name}." if result.current_page else "."),
    )
    _emit(
        progress_callback,
        {
            "event": "process_canceled",
            "process_stage": step.key,
            "workflow_stage": step.workflow_stage,
            "display_name": step.display_name,
            "status": "canceled",
            "image_relative_path": result.current_page,
            "step_index": step_index,
            "step_total": len(PROCESS_PIPELINE_STEPS),
            "page_total": total_pages,
            "active_page_total": active_page_total,
            "message": "Process canceled by user.",
            "overall_progress": _progress_value(completed_units, total_units),
        },
    )
    return result


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None:
        logger(message)


__all__ = [
    "PROCESS_PIPELINE_STEPS",
    "PROCESS_STEP_KEYS",
    "PROCESS_WORKFLOW_STAGE_BY_STEP",
    "ProcessCanceledError",
    "ProcessPageFailure",
    "ProcessPipelineResult",
    "ProcessStepDefinition",
    "run_process_pipeline",
]
