"""Background workers for GUI-triggered pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
import traceback
from typing import Any

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

from mmt_core import (
    OCRConfig,
    ExportConfig,
    ProcessPipelineResult,
    TranslationConfig,
    inpaint_json_path,
    load_ocr_json,
    load_inpaint_json,
    load_render_json,
    load_translation_json,
    load_lama_model,
    render_json_path,
    run_process_pipeline,
    summarize_ocr_items,
    summarize_inpaint_json,
    summarize_render_json,
    summarize_translation_json,
    unload_lama_model,
)
from mmt_core.ocr_models import provider_label
from mmt_core.ocr_providers import create_ocr_provider
from mmt_core.export_stage import run_export
from mmt_core.detection_stage import run_detection_for_image
from mmt_core.inpaint_stage import prepare_inpaint_mask_for_page, run_inpaint_for_page, run_inpaint_for_pages
from mmt_core.llama_server import LlamaServerManager, LlamaServerStatus
from mmt_core.ocr_stage import prepare_ocr_items_for_image, run_ocr_for_page
from mmt_core.translation_stage import (
    initialize_translation_for_page,
    run_translation_for_page,
    run_translation_for_pages,
)
from mmt_core.render_stage import prepare_render_for_page, run_render_for_page, run_render_for_pages


@dataclass(slots=True)
class PipelineTask:
    """Base descriptor for a background pipeline task."""

    name: str
    stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectionTask(PipelineTask):
    """Configuration for one or more detection runs."""

    image_paths: list[Path] = field(default_factory=list)
    detection_cache_dir: Path = Path(".")
    masks_cache_dir: Path = Path(".")
    force: bool = False
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DetectionPageResult:
    """Per-page detection outcome."""

    image_path: Path
    json_path: Path | None = None
    error: str | None = None


@dataclass(slots=True)
class DetectionWorkerResult:
    """Aggregated result emitted when a detection task finishes."""

    page_results: list[DetectionPageResult] = field(default_factory=list)

    @property
    def json_paths(self) -> list[Path]:
        return [result.json_path for result in self.page_results if result.json_path is not None]

    @property
    def failures(self) -> list[DetectionPageResult]:
        return [result for result in self.page_results if result.error is not None]


@dataclass(slots=True)
class MangaDetectorTask(PipelineTask):
    """Configuration for Manga RT-DETR detector lifecycle actions."""

    action: str = "status"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MangaDetectorTaskResult:
    """Structured result for a Manga RT-DETR detector lifecycle action."""

    loaded: bool
    device: str
    model_id: str
    message: str
    detection_config: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass(slots=True)
class OCRPreparationTask(PipelineTask):
    """Configuration for preparing OCR items from detection caches."""

    project: Any = None
    image_relative_paths: list[str] = field(default_factory=list)
    force: bool = False
    save_crops: bool = True


@dataclass(slots=True)
class OCRPreparationPageResult:
    """Per-page OCR preparation outcome."""

    image_relative_path: str
    json_path: Path | None = None
    error: str | None = None


@dataclass(slots=True)
class OCRPreparationWorkerResult:
    """Aggregated OCR preparation result."""

    page_results: list[OCRPreparationPageResult] = field(default_factory=list)

    @property
    def json_paths(self) -> list[Path]:
        return [result.json_path for result in self.page_results if result.json_path is not None]

    @property
    def failures(self) -> list[OCRPreparationPageResult]:
        return [result for result in self.page_results if result.error is not None]


@dataclass(slots=True)
class OCRInferenceTask(PipelineTask):
    """Configuration for OCR inference over prepared OCR items."""

    project: Any = None
    image_relative_paths: list[str] = field(default_factory=list)
    config: Any = None
    server_url: str = ""
    force: bool = False
    selected_item_ids_by_page: dict[str, list[int]] = field(default_factory=dict)
    timeout: float = 120.0


@dataclass(slots=True)
class OCRInferencePageResult:
    """Per-page OCR inference outcome."""

    image_relative_path: str
    json_path: Path | None = None
    error: str | None = None
    summary: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class OCRInferenceWorkerResult:
    """Aggregated OCR inference result."""

    page_results: list[OCRInferencePageResult] = field(default_factory=list)

    @property
    def json_paths(self) -> list[Path]:
        return [result.json_path for result in self.page_results if result.json_path is not None]

    @property
    def failures(self) -> list[OCRInferencePageResult]:
        return [result for result in self.page_results if result.error is not None]


@dataclass(slots=True)
class TranslationInitializationTask(PipelineTask):
    """Configuration for initializing translation JSON from OCR JSON."""

    project: Any = None
    image_relative_paths: list[str] = field(default_factory=list)
    config: Any = None
    force: bool = False


@dataclass(slots=True)
class TranslationInitializationPageResult:
    """Per-page translation initialization outcome."""

    image_relative_path: str
    json_path: Path | None = None
    error: str | None = None
    summary: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationInitializationWorkerResult:
    """Aggregated translation initialization result."""

    page_results: list[TranslationInitializationPageResult] = field(default_factory=list)

    @property
    def json_paths(self) -> list[Path]:
        return [result.json_path for result in self.page_results if result.json_path is not None]

    @property
    def failures(self) -> list[TranslationInitializationPageResult]:
        return [result for result in self.page_results if result.error is not None]


@dataclass(slots=True)
class TranslationTask(PipelineTask):
    """Configuration for translation execution from cached OCR text."""

    project: Any = None
    image_relative_paths: list[str] = field(default_factory=list)
    config: Any = None
    force: bool = False
    selected_item_ids_by_page: dict[str, list[int]] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationPageResult:
    """Per-page translation execution outcome."""

    image_relative_path: str
    json_path: Path | None = None
    error: str | None = None
    summary: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationWorkerResult:
    """Aggregated translation execution result."""

    page_results: list[TranslationPageResult] = field(default_factory=list)

    @property
    def json_paths(self) -> list[Path]:
        return [result.json_path for result in self.page_results if result.json_path is not None]

    @property
    def failures(self) -> list[TranslationPageResult]:
        return [result for result in self.page_results if result.error is not None]


@dataclass(slots=True)
class InpaintMaskTask(PipelineTask):
    """Configuration for preparing inpaint masks from cached OCR data."""

    project: Any = None
    image_relative_paths: list[str] = field(default_factory=list)
    force: bool = False
    mask_padding: int = 0
    use_bubble_mask: bool = True


@dataclass(slots=True)
class InpaintMaskPageResult:
    """Per-page inpaint mask preparation outcome."""

    image_relative_path: str
    mask_path: Path | None = None
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InpaintMaskWorkerResult:
    """Aggregated inpaint mask preparation results."""

    page_results: list[InpaintMaskPageResult] = field(default_factory=list)

    @property
    def mask_paths(self) -> list[Path]:
        return [result.mask_path for result in self.page_results if result.mask_path is not None]

    @property
    def failures(self) -> list[InpaintMaskPageResult]:
        return [result for result in self.page_results if result.error is not None]


@dataclass(slots=True)
class InpaintTask(PipelineTask):
    """Configuration for running page inpainting."""

    project: Any = None
    image_relative_paths: list[str] = field(default_factory=list)
    force: bool = False
    mask_padding: int = 0
    use_bubble_mask: bool = True
    use_crop_windows: bool = True
    device: str | None = None


@dataclass(slots=True)
class InpaintPageResult:
    """Per-page inpainting outcome."""

    image_relative_path: str
    image_path: Path | None = None
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InpaintWorkerResult:
    """Aggregated inpainting results."""

    page_results: list[InpaintPageResult] = field(default_factory=list)

    @property
    def image_paths(self) -> list[Path]:
        return [result.image_path for result in self.page_results if result.image_path is not None]

    @property
    def failures(self) -> list[InpaintPageResult]:
        return [result for result in self.page_results if result.error is not None]


@dataclass(slots=True)
class RenderPreparationTask(PipelineTask):
    """Configuration for preparing render metadata from translation caches."""

    project: Any = None
    image_relative_paths: list[str] = field(default_factory=list)
    force: bool = False


@dataclass(slots=True)
class RenderPreparationPageResult:
    """Per-page render preparation outcome."""

    image_relative_path: str
    json_path: Path | None = None
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RenderPreparationWorkerResult:
    """Aggregated render preparation results."""

    page_results: list[RenderPreparationPageResult] = field(default_factory=list)

    @property
    def json_paths(self) -> list[Path]:
        return [result.json_path for result in self.page_results if result.json_path is not None]

    @property
    def failures(self) -> list[RenderPreparationPageResult]:
        return [result for result in self.page_results if result.error is not None]


@dataclass(slots=True)
class RenderTask(PipelineTask):
    """Configuration for rendering translated text onto inpainted pages."""

    project: Any = None
    image_relative_paths: list[str] = field(default_factory=list)
    config: Any = None
    force: bool = False


@dataclass(slots=True)
class RenderPageResult:
    """Per-page render execution outcome."""

    image_relative_path: str
    image_path: Path | None = None
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RenderWorkerResult:
    """Aggregated render execution results."""

    page_results: list[RenderPageResult] = field(default_factory=list)

    @property
    def image_paths(self) -> list[Path]:
        return [result.image_path for result in self.page_results if result.image_path is not None]

    @property
    def failures(self) -> list[RenderPageResult]:
        return [result for result in self.page_results if result.error is not None]


@dataclass(slots=True)
class LlamaServerTask(PipelineTask):
    """Configuration for a llama.cpp server action."""

    manager: LlamaServerManager | None = None
    action: str = "check"
    timeout_seconds: float = 30.0


@dataclass(slots=True)
class LlamaServerTaskResult:
    """Structured result for a llama.cpp server action."""

    state: str
    message: str
    is_alive: bool
    managed: bool


@dataclass(slots=True)
class LamaModelTask(PipelineTask):
    """Configuration for LaMa model lifecycle actions."""

    action: str = "status"
    device: str | None = None


@dataclass(slots=True)
class LamaModelTaskResult:
    """Structured result for a LaMa model lifecycle action."""

    loaded: bool
    device: str
    message: str


@dataclass(slots=True)
class ExportTask(PipelineTask):
    """Configuration for exporting cached output files."""

    project: Any = None
    config: Any = None
    current_page: str | None = None
    selected_pages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExportWorkerResult:
    """Structured result for a completed export run."""

    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def exported_count(self) -> int:
        return int(self.manifest.get("exported_count", 0) or 0)

    @property
    def skipped_count(self) -> int:
        return int(self.manifest.get("skipped_count", 0) or 0)

    @property
    def error_count(self) -> int:
        return int(self.manifest.get("error_count", 0) or 0)


@dataclass(slots=True)
class ProcessTask(PipelineTask):
    """Configuration for a one-click Detection -> Render process run."""

    project: Any = None
    image_relative_paths: list[str] = field(default_factory=list)
    scope: str = "chapter"
    force: bool = False
    ocr_config: Any = None
    translation_config: Any = None
    detection_config: Any = None
    inpaint_settings: dict[str, Any] = field(default_factory=dict)
    render_config: Any = None
    cancel_token: Any = None


class WorkerSignals(QObject):
    """Signals shared by background task runners."""

    started = pyqtSignal(str)
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    event = pyqtSignal(object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)


WorkerCallback = Callable[[PipelineTask, WorkerSignals], Any]


class TaskWorker(QRunnable):
    """Runs a callback in Qt's thread pool."""

    def __init__(
        self,
        task: PipelineTask,
        callback: WorkerCallback | None = None,
        *,
        cancel_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.task = task
        self.callback = callback
        self.signals = WorkerSignals()
        self._cancel_event = cancel_event or threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    @pyqtSlot()
    def run(self) -> None:
        self.signals.started.emit(self.task.name)

        try:
            if self.callback is None:
                self.signals.message.emit(
                    f"{self.task.stage} worker placeholder is ready for future pipeline integration."
                )
                result: Any = None
            else:
                result = self.callback(self.task, self.signals)
        except Exception as exc:  # pragma: no cover - defensive UI plumbing.
            thread_name = threading.current_thread().name
            self.signals.message.emit(
                f"{self.task.name} failed on worker thread {thread_name}: {exc}"
            )
            self.signals.message.emit(traceback.format_exc())
            self.signals.failed.emit(str(exc))
            return

        self.signals.finished.emit(result)


def create_detection_worker(task: DetectionTask) -> TaskWorker:
    """Create a QRunnable that executes detection off the UI thread."""

    return TaskWorker(task=task, callback=_run_detection_task)


def create_ocr_preparation_worker(task: OCRPreparationTask) -> TaskWorker:
    """Create a QRunnable that prepares OCR items off the UI thread."""

    return TaskWorker(task=task, callback=_run_ocr_preparation_task)


def create_ocr_inference_worker(task: OCRInferenceTask) -> TaskWorker:
    """Create a QRunnable that runs OCR inference off the UI thread."""

    return TaskWorker(task=task, callback=_run_ocr_inference_task)


def create_translation_initialization_worker(task: TranslationInitializationTask) -> TaskWorker:
    """Create a QRunnable that initializes translation caches off the UI thread."""

    return TaskWorker(task=task, callback=_run_translation_initialization_task)


def create_translation_worker(task: TranslationTask) -> TaskWorker:
    """Create a QRunnable that runs translation off the UI thread."""

    return TaskWorker(task=task, callback=_run_translation_task)


def create_inpaint_mask_worker(task: InpaintMaskTask) -> TaskWorker:
    """Create a QRunnable that prepares inpaint masks off the UI thread."""

    return TaskWorker(task=task, callback=_run_inpaint_mask_task)


def create_inpaint_worker(task: InpaintTask) -> TaskWorker:
    """Create a QRunnable that runs inpainting off the UI thread."""

    return TaskWorker(task=task, callback=_run_inpaint_task)


def create_render_preparation_worker(task: RenderPreparationTask) -> TaskWorker:
    """Create a QRunnable that prepares render metadata off the UI thread."""

    return TaskWorker(task=task, callback=_run_render_preparation_task)


def create_render_worker(task: RenderTask) -> TaskWorker:
    """Create a QRunnable that renders translated pages off the UI thread."""

    return TaskWorker(task=task, callback=_run_render_task)


def create_llama_server_worker(task: LlamaServerTask) -> TaskWorker:
    """Create a QRunnable that manages the llama.cpp server off the UI thread."""

    return TaskWorker(task=task, callback=_run_llama_server_task)


def create_lama_model_worker(task: LamaModelTask) -> TaskWorker:
    """Create a QRunnable that manages LaMa model load/unload off the UI thread."""

    return TaskWorker(task=task, callback=_run_lama_model_task)


def create_export_worker(task: ExportTask) -> TaskWorker:
    """Create a QRunnable that exports cached outputs off the UI thread."""

    return TaskWorker(task=task, callback=_run_export_task)


def create_process_worker(task: ProcessTask) -> TaskWorker:
    """Create a QRunnable that runs the one-click process pipeline off the UI thread."""

    cancel_event = threading.Event()
    task.cancel_token = cancel_event
    return TaskWorker(task=task, callback=_run_process_task, cancel_event=cancel_event)


def _run_detection_task(task: PipelineTask, signals: WorkerSignals) -> DetectionWorkerResult:
    if not isinstance(task, DetectionTask):
        raise TypeError("Detection worker received an unexpected task type.")

    if not task.image_paths:
        raise ValueError("No source images were provided for detection.")

    total_pages = len(task.image_paths)
    page_results: list[DetectionPageResult] = []
    signals.message.emit(f"Starting detection for {total_pages} page(s).")
    signals.progress.emit(0)

    for index, image_path in enumerate(task.image_paths, start=1):
        signals.event.emit(
            {
                "stage": "detection",
                "event": "page_start",
                "image_path": str(image_path),
                "page_index": index,
                "page_total": total_pages,
            }
        )
        signals.message.emit(f"[{index}/{total_pages}] Detecting {Path(image_path).name}")

        try:
            json_path = run_detection_for_image(
                Path(image_path),
                task.detection_cache_dir,
                task.masks_cache_dir,
                force=task.force,
                detection_config=task.config,
                logger=signals.message.emit,
            )
        except Exception as exc:
            readable_error = f"{Path(image_path).name}: {exc}"
            signals.message.emit(f"Detection failed: {readable_error}")
            page_results.append(
                DetectionPageResult(
                    image_path=Path(image_path),
                    json_path=None,
                    error=str(exc),
                )
            )
        else:
            page_results.append(
                DetectionPageResult(
                    image_path=Path(image_path),
                    json_path=json_path,
                    error=None,
                )
            )
            signals.message.emit(f"Detection complete: {json_path}")
            signals.event.emit(
                {
                    "stage": "detection",
                    "event": "page_done",
                    "image_path": str(image_path),
                    "output_path": str(json_path),
                    "page_index": index,
                    "page_total": total_pages,
                }
            )

        signals.progress.emit(int((index / total_pages) * 100))

    successful_pages = [result for result in page_results if result.json_path is not None]
    if not successful_pages:
        first_error = page_results[0].error if page_results else "Unknown detection failure."
        raise RuntimeError(f"Detection failed for all pages. {first_error}")

    failed_count = len([result for result in page_results if result.error is not None])
    if failed_count:
        signals.message.emit(
            f"Detection finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
        )
    else:
        signals.message.emit(f"Detection finished successfully for {len(successful_pages)} page(s).")

    return DetectionWorkerResult(page_results=page_results)


def _run_ocr_preparation_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> OCRPreparationWorkerResult:
    if not isinstance(task, OCRPreparationTask):
        raise TypeError("OCR preparation worker received an unexpected task type.")

    if task.project is None:
        raise ValueError("No project was provided for OCR preparation.")

    if not task.image_relative_paths:
        raise ValueError("No project pages were provided for OCR preparation.")

    total_pages = len(task.image_relative_paths)
    page_results: list[OCRPreparationPageResult] = []
    signals.message.emit(f"Starting OCR item preparation for {total_pages} page(s).")
    signals.progress.emit(0)

    for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
        signals.event.emit(
            {
                "stage": "ocr_prepare",
                "event": "page_start",
                "image_relative_path": str(image_relative_path),
                "page_index": index,
                "page_total": total_pages,
            }
        )
        signals.message.emit(f"[{index}/{total_pages}] Preparing OCR items for {Path(image_relative_path).name}")
        try:
            json_path = prepare_ocr_items_for_image(
                task.project,
                image_relative_path,
                force=task.force,
                save_crops=task.save_crops,
                logger=signals.message.emit,
            )
        except Exception as exc:
            readable_error = f"{Path(image_relative_path).name}: {exc}"
            signals.message.emit(f"OCR preparation failed: {readable_error}")
            page_results.append(
                OCRPreparationPageResult(
                    image_relative_path=str(image_relative_path),
                    json_path=None,
                    error=str(exc),
                )
            )
        else:
            page_results.append(
                OCRPreparationPageResult(
                    image_relative_path=str(image_relative_path),
                    json_path=json_path,
                    error=None,
                )
            )
            signals.message.emit(f"OCR items prepared: {json_path}")
            signals.event.emit(
                {
                    "stage": "ocr_prepare",
                    "event": "page_done",
                    "image_relative_path": str(image_relative_path),
                    "output_path": str(json_path),
                    "page_index": index,
                    "page_total": total_pages,
                }
            )

        signals.progress.emit(int((index / total_pages) * 100))

    successful_pages = [result for result in page_results if result.json_path is not None]
    if not successful_pages:
        first_error = page_results[0].error if page_results else "Unknown OCR preparation failure."
        raise RuntimeError(f"OCR item preparation failed for all pages. {first_error}")

    failed_count = len([result for result in page_results if result.error is not None])
    if failed_count:
        signals.message.emit(
            f"OCR item preparation finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
        )
    else:
        signals.message.emit(f"OCR item preparation finished successfully for {len(successful_pages)} page(s).")

    return OCRPreparationWorkerResult(page_results=page_results)


def _run_ocr_inference_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> OCRInferenceWorkerResult:
    if not isinstance(task, OCRInferenceTask):
        raise TypeError("OCR inference worker received an unexpected task type.")

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
    signals.message.emit(f"Running OCR with {provider_label(ocr_config.ocr_provider)}")
    signals.message.emit(f"Starting OCR inference for {total_pages} page(s).")
    signals.progress.emit(0)

    provider = create_ocr_provider(ocr_config)
    try:
        provider.validate()

        for page_index, image_relative_path in enumerate(task.image_relative_paths, start=1):
            page_name = Path(image_relative_path).name
            signals.event.emit(
                {
                    "stage": "ocr",
                    "event": "page_start",
                    "image_relative_path": str(image_relative_path),
                    "page_index": page_index,
                    "page_total": total_pages,
                }
            )
            signals.message.emit(f"[{page_index}/{total_pages}] Running OCR for {page_name}")

            def on_item_progress(current: int, total: int, item_info: dict[str, Any]) -> None:
                safe_total = max(int(total), 1)
                progress_ratio = ((page_index - 1) + (min(int(current), safe_total) / safe_total)) / total_pages
                signals.progress.emit(int(progress_ratio * 100))
                item_message = str(item_info.get("message", "") or "").strip()
                if item_message:
                    signals.message.emit(
                        f"[{page_index}/{total_pages}] {page_name} "
                        f"item {min(int(current), safe_total)}/{safe_total}: {item_message}"
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
                    logger=signals.message.emit,
                    progress_callback=on_item_progress,
                )
                ocr_data = load_ocr_json(json_path)
                summary = summarize_ocr_items(ocr_data.get("items", []))
            except Exception as exc:
                readable_error = f"{page_name}: {exc}"
                signals.message.emit(f"OCR inference failed: {readable_error}")
                page_results.append(
                    OCRInferencePageResult(
                        image_relative_path=str(image_relative_path),
                        json_path=None,
                        error=str(exc),
                        summary={},
                    )
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
                signals.message.emit(
                    f"OCR complete for {page_name}: "
                    f"{summary.get('done', 0)} done, {summary.get('error', 0)} error, "
                    f"{summary.get('prepared', 0)} prepared."
                )
                signals.event.emit(
                    {
                        "stage": "ocr",
                        "event": "page_done",
                        "image_relative_path": str(image_relative_path),
                        "output_path": str(json_path),
                        "page_index": page_index,
                        "page_total": total_pages,
                        "summary": dict(summary),
                    }
                )

            signals.progress.emit(int((page_index / total_pages) * 100))
    finally:
        provider.close()

    successful_pages = [result for result in page_results if result.json_path is not None]
    if not successful_pages:
        first_error = page_results[0].error if page_results else "Unknown OCR inference failure."
        raise RuntimeError(f"OCR inference failed for all pages. {first_error}")

    failed_count = len([result for result in page_results if result.error is not None])
    if failed_count:
        signals.message.emit(
            f"OCR inference finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
        )
    else:
        signals.message.emit(f"OCR inference finished successfully for {len(successful_pages)} page(s).")

    return OCRInferenceWorkerResult(page_results=page_results)


def _run_translation_initialization_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> TranslationInitializationWorkerResult:
    if not isinstance(task, TranslationInitializationTask):
        raise TypeError("Translation initialization worker received an unexpected task type.")

    if task.project is None:
        raise ValueError("No project was provided for translation initialization.")

    if not task.image_relative_paths:
        raise ValueError("No pages were provided for translation initialization.")

    total_pages = len(task.image_relative_paths)
    page_results: list[TranslationInitializationPageResult] = []
    signals.message.emit(f"Starting translation initialization for {total_pages} page(s).")
    signals.progress.emit(0)

    for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
        page_name = Path(image_relative_path).name
        signals.message.emit(f"[{index}/{total_pages}] Initializing translation for {page_name}")
        try:
            json_path = initialize_translation_for_page(
                task.project,
                image_relative_path,
                task.config,
                force=task.force,
                logger=signals.message.emit,
            )
            translation_data = load_translation_json(json_path)
            summary = summarize_translation_json(translation_data)
        except Exception as exc:
            readable_error = f"{page_name}: {exc}"
            signals.message.emit(f"Translation initialization failed: {readable_error}")
            page_results.append(
                TranslationInitializationPageResult(
                    image_relative_path=str(image_relative_path),
                    json_path=None,
                    error=str(exc),
                    summary={},
                )
            )
        else:
            page_results.append(
                TranslationInitializationPageResult(
                    image_relative_path=str(image_relative_path),
                    json_path=json_path,
                    error=None,
                    summary=summary,
                )
            )
            signals.message.emit(f"Translation cache ready: {json_path}")

        signals.progress.emit(int((index / total_pages) * 100))

    successful_pages = [result for result in page_results if result.json_path is not None]
    if not successful_pages:
        first_error = page_results[0].error if page_results else "Unknown translation initialization failure."
        raise RuntimeError(f"Translation initialization failed for all pages. {first_error}")

    failed_count = len([result for result in page_results if result.error is not None])
    if failed_count:
        signals.message.emit(
            f"Translation initialization finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
        )
    else:
        signals.message.emit(
            f"Translation initialization finished successfully for {len(successful_pages)} page(s)."
        )

    return TranslationInitializationWorkerResult(page_results=page_results)


def _run_translation_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> TranslationWorkerResult:
    if not isinstance(task, TranslationTask):
        raise TypeError("Translation worker received an unexpected task type.")

    if task.project is None:
        raise ValueError("No project was provided for translation.")

    if not task.image_relative_paths:
        raise ValueError("No pages were provided for translation.")

    total_pages = len(task.image_relative_paths)
    translation_config = TranslationConfig.from_value(task.config)
    page_results_map: dict[str, TranslationPageResult] = {
        str(image_relative_path): TranslationPageResult(image_relative_path=str(image_relative_path))
        for image_relative_path in task.image_relative_paths
    }

    def on_progress(event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("stage", "translation")
        signals.event.emit(payload)
        event_name = str(event.get("event", "") or "")
        message = str(event.get("message", "") or "").strip()
        image_relative_path = str(event.get("image_relative_path", "") or "")
        if message:
            signals.message.emit(message)

        if event_name == "chunk_start":
            chunk_index = int(event.get("chunk_index", 1) or 1)
            chunk_total = max(int(event.get("chunk_total", 1) or 1), 1)
            signals.progress.emit(int(((chunk_index - 1) / chunk_total) * 100))
            return

        if event_name in {"page_done", "page_error"} and image_relative_path:
            processed_pages = len(
                [
                    result
                    for result in page_results_map.values()
                    if result.json_path is not None or result.error is not None
                ]
            )
            signals.progress.emit(int((processed_pages / max(total_pages, 1)) * 100))

            page_result = page_results_map.setdefault(
                image_relative_path,
                TranslationPageResult(image_relative_path=image_relative_path),
            )
            output_path = event.get("output_path")
            if isinstance(output_path, str) and output_path.strip():
                page_result.json_path = Path(output_path)
            summary = event.get("summary")
            if isinstance(summary, dict):
                page_result.summary = {str(key): int(value) for key, value in summary.items() if isinstance(value, int)}
            if event_name == "page_error":
                page_result.error = message or "Unknown translation failure."

    signals.message.emit(
        f"Starting translation for {total_pages} page(s) with {translation_config.translator}."
    )
    signals.progress.emit(0)

    if len(task.image_relative_paths) == 1:
        image_relative_path = task.image_relative_paths[0]
        page_name = Path(image_relative_path).name
        signals.message.emit(f"[1/1] Translating {page_name}")
        try:
            json_path = run_translation_for_page(
                task.project,
                image_relative_path,
                task.config,
                force=task.force,
                selected_item_ids=task.selected_item_ids_by_page.get(str(image_relative_path)),
                logger=signals.message.emit,
                progress_callback=on_progress,
            )
            translation_data = load_translation_json(json_path)
            summary = summarize_translation_json(translation_data)
        except Exception as exc:
            readable_error = f"{page_name}: {exc}"
            signals.message.emit(f"Translation failed: {readable_error}")
            page_results_map[str(image_relative_path)] = TranslationPageResult(
                image_relative_path=str(image_relative_path),
                json_path=None,
                error=str(exc),
                summary={},
            )
        else:
            page_results_map[str(image_relative_path)] = TranslationPageResult(
                image_relative_path=str(image_relative_path),
                json_path=json_path,
                error=None,
                summary=summary,
            )
        signals.progress.emit(100)
    else:
        try:
            output_paths = run_translation_for_pages(
                task.project,
                task.image_relative_paths,
                task.config,
                force=task.force,
                logger=signals.message.emit,
                progress_callback=on_progress,
            )
        except Exception as exc:
            signals.message.emit(f"Translation failed: {exc}")
        else:
            for output_path in output_paths:
                try:
                    translation_data = load_translation_json(output_path)
                except Exception:
                    continue
                source_image = str(translation_data.get("source_image", "") or "")
                page_result = page_results_map.setdefault(
                    source_image,
                    TranslationPageResult(image_relative_path=source_image),
                )
                page_result.json_path = output_path
                page_result.summary = summarize_translation_json(translation_data)
                if page_result.error is None and page_result.summary.get("error", 0) > 0:
                    page_result.error = "One or more translation items failed."

        for image_relative_path in task.image_relative_paths:
            page_key = str(image_relative_path)
            page_result = page_results_map.setdefault(
                page_key,
                TranslationPageResult(image_relative_path=page_key),
            )
            if page_result.json_path is not None or page_result.error is not None:
                continue
            try:
                json_path = task.project.cache_dir / "translation" / f"{Path(page_key).stem}.json"
                if json_path.exists():
                    translation_data = load_translation_json(json_path)
                    page_result.json_path = json_path
                    page_result.summary = summarize_translation_json(translation_data)
            except Exception as exc:
                page_result.error = str(exc)

        signals.progress.emit(100)

    page_results = [page_results_map[str(image_relative_path)] for image_relative_path in task.image_relative_paths]
    successful_pages = [result for result in page_results if result.json_path is not None]
    if not successful_pages:
        first_error = page_results[0].error if page_results else "Unknown translation failure."
        raise RuntimeError(f"Translation failed for all pages. {first_error}")

    failed_count = len([result for result in page_results if result.error is not None])
    if failed_count:
        signals.message.emit(
            f"Translation finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
        )
    else:
        signals.message.emit(f"Translation finished successfully for {len(successful_pages)} page(s).")

    return TranslationWorkerResult(page_results=page_results)


def _run_inpaint_mask_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> InpaintMaskWorkerResult:
    if not isinstance(task, InpaintMaskTask):
        raise TypeError("Inpaint mask worker received an unexpected task type.")

    if task.project is None:
        raise ValueError("No project was provided for inpaint mask preparation.")

    if not task.image_relative_paths:
        raise ValueError("No pages were provided for inpaint mask preparation.")

    total_pages = len(task.image_relative_paths)
    page_results: list[InpaintMaskPageResult] = []
    signals.message.emit(f"Preparing inpaint masks for {total_pages} page(s).")
    signals.progress.emit(0)

    for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
        page_name = Path(image_relative_path).name
        signals.message.emit(f"[{index}/{total_pages}] Preparing mask for {page_name}")
        try:
            mask_path = prepare_inpaint_mask_for_page(
                task.project,
                image_relative_path,
                mask_padding=task.mask_padding,
                use_bubble_mask=task.use_bubble_mask,
                force=task.force,
                logger=signals.message.emit,
            )
            metadata = load_inpaint_json(inpaint_json_path(task.project, image_relative_path))
            summary = summarize_inpaint_json(metadata)
        except Exception as exc:
            readable_error = f"{page_name}: {exc}"
            signals.message.emit(f"Inpaint mask preparation failed: {readable_error}")
            page_results.append(
                InpaintMaskPageResult(
                    image_relative_path=str(image_relative_path),
                    mask_path=None,
                    error=str(exc),
                    summary={},
                )
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
            signals.message.emit(f"Inpaint mask ready: {mask_path}")

        signals.progress.emit(int((index / total_pages) * 100))

    successful_pages = [result for result in page_results if result.mask_path is not None]
    if not successful_pages:
        first_error = page_results[0].error if page_results else "Unknown inpaint mask failure."
        raise RuntimeError(f"Inpaint mask preparation failed for all pages. {first_error}")

    failed_count = len([result for result in page_results if result.error is not None])
    if failed_count:
        signals.message.emit(
            f"Inpaint mask preparation finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
        )
    else:
        signals.message.emit(
            f"Inpaint mask preparation finished successfully for {len(successful_pages)} page(s)."
        )

    return InpaintMaskWorkerResult(page_results=page_results)


def _run_inpaint_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> InpaintWorkerResult:
    if not isinstance(task, InpaintTask):
        raise TypeError("Inpaint worker received an unexpected task type.")

    if task.project is None:
        raise ValueError("No project was provided for inpainting.")

    if not task.image_relative_paths:
        raise ValueError("No pages were provided for inpainting.")

    total_pages = len(task.image_relative_paths)
    page_results_map: dict[str, InpaintPageResult] = {
        str(image_relative_path): InpaintPageResult(image_relative_path=str(image_relative_path))
        for image_relative_path in task.image_relative_paths
    }

    def on_progress(event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("stage", "inpaint")
        signals.event.emit(payload)
        event_name = str(event.get("event", "") or "")
        message = str(event.get("message", "") or "").strip()
        image_relative_path = str(event.get("image_relative_path", "") or "")
        if message:
            signals.message.emit(message)

        if event_name == "batch_page_start":
            page_index = max(int(event.get("page_index", 1) or 1), 1)
            page_total = max(int(event.get("page_total", total_pages) or total_pages), 1)
            signals.progress.emit(int(((page_index - 1) / page_total) * 100))
            return

        if event_name in {"page_done", "page_error"} and image_relative_path:
            page_result = page_results_map.setdefault(
                image_relative_path,
                InpaintPageResult(image_relative_path=image_relative_path),
            )
            output_path = event.get("output_path")
            if isinstance(output_path, str) and output_path.strip():
                page_result.image_path = Path(output_path)
            summary = event.get("summary")
            if isinstance(summary, dict):
                page_result.summary = dict(summary)
            if event_name == "page_error":
                page_result.error = message or "Unknown inpaint failure."

            processed_pages = len(
                [
                    result
                    for result in page_results_map.values()
                    if result.image_path is not None or result.error is not None
                ]
            )
            signals.progress.emit(int((processed_pages / max(total_pages, 1)) * 100))

    signals.message.emit(f"Starting inpaint for {total_pages} page(s).")
    signals.progress.emit(0)

    if len(task.image_relative_paths) == 1:
        image_relative_path = task.image_relative_paths[0]
        page_name = Path(image_relative_path).name
        signals.message.emit(f"[1/1] Inpainting {page_name}")
        try:
            image_path = run_inpaint_for_page(
                task.project,
                image_relative_path,
                force=task.force,
                mask_padding=task.mask_padding,
                use_bubble_mask=task.use_bubble_mask,
                use_crop_windows=task.use_crop_windows,
                device=task.device,
                logger=signals.message.emit,
                progress_callback=on_progress,
            )
            metadata = load_inpaint_json(inpaint_json_path(task.project, image_relative_path))
            summary = summarize_inpaint_json(metadata)
        except Exception as exc:
            readable_error = f"{page_name}: {exc}"
            signals.message.emit(f"Inpaint failed: {readable_error}")
            page_results_map[str(image_relative_path)] = InpaintPageResult(
                image_relative_path=str(image_relative_path),
                image_path=None,
                error=str(exc),
                summary={},
            )
        else:
            page_results_map[str(image_relative_path)] = InpaintPageResult(
                image_relative_path=str(image_relative_path),
                image_path=image_path,
                error=None,
                summary=summary,
            )
        signals.progress.emit(100)
    else:
        try:
            output_paths = run_inpaint_for_pages(
                task.project,
                task.image_relative_paths,
                force=task.force,
                mask_padding=task.mask_padding,
                use_bubble_mask=task.use_bubble_mask,
                use_crop_windows=task.use_crop_windows,
                device=task.device,
                logger=signals.message.emit,
                progress_callback=on_progress,
            )
        except Exception as exc:
            signals.message.emit(f"Inpaint failed: {exc}")
        else:
            for output_path in output_paths:
                # The metadata file is the canonical source of the original page mapping.
                metadata_path = output_path.parent / f"{output_path.stem}.json"
                try:
                    metadata = load_inpaint_json(metadata_path)
                except Exception:
                    continue
                source_image = str(metadata.get("source_image", "") or "")
                page_result = page_results_map.setdefault(
                    source_image,
                    InpaintPageResult(image_relative_path=source_image),
                )
                page_result.image_path = output_path
                page_result.summary = summarize_inpaint_json(metadata)
                if page_result.error is None and str(metadata.get("status", "")).lower() == "error":
                    page_result.error = str(metadata.get("error", "") or "Inpaint failed.")

        for image_relative_path in task.image_relative_paths:
            page_key = str(image_relative_path)
            page_result = page_results_map.setdefault(
                page_key,
                InpaintPageResult(image_relative_path=page_key),
            )
            if page_result.image_path is not None or page_result.error is not None:
                continue
            try:
                metadata_path = inpaint_json_path(task.project, page_key)
                if metadata_path.exists():
                    metadata = load_inpaint_json(metadata_path)
                    output_path = task.project.root_dir / str(metadata.get("output_image_path", "") or "")
                    if output_path.exists():
                        page_result.image_path = output_path
                    page_result.summary = summarize_inpaint_json(metadata)
                    if str(metadata.get("status", "")).lower() == "error":
                        page_result.error = str(metadata.get("error", "") or "Inpaint failed.")
            except Exception as exc:
                page_result.error = str(exc)

        signals.progress.emit(100)

    page_results = [page_results_map[str(image_relative_path)] for image_relative_path in task.image_relative_paths]
    successful_pages = [result for result in page_results if result.image_path is not None]
    if not successful_pages:
        first_error = page_results[0].error if page_results else "Unknown inpaint failure."
        raise RuntimeError(f"Inpaint failed for all pages. {first_error}")

    failed_count = len([result for result in page_results if result.error is not None])
    if failed_count:
        signals.message.emit(
            f"Inpaint finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
        )
    else:
        signals.message.emit(f"Inpaint finished successfully for {len(successful_pages)} page(s).")

    return InpaintWorkerResult(page_results=page_results)


def _run_render_preparation_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> RenderPreparationWorkerResult:
    if not isinstance(task, RenderPreparationTask):
        raise TypeError("Render preparation worker received an unexpected task type.")

    if task.project is None:
        raise ValueError("No project was provided for render preparation.")

    if not task.image_relative_paths:
        raise ValueError("No pages were provided for render preparation.")

    total_pages = len(task.image_relative_paths)
    page_results: list[RenderPreparationPageResult] = []
    signals.message.emit(f"Preparing render metadata for {total_pages} page(s).")
    signals.progress.emit(0)

    for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
        page_name = Path(image_relative_path).name
        signals.message.emit(f"[{index}/{total_pages}] Preparing render metadata for {page_name}")
        try:
            json_path = prepare_render_for_page(
                task.project,
                image_relative_path,
                force=task.force,
                logger=signals.message.emit,
            )
            render_data = load_render_json(json_path)
            summary = summarize_render_json(render_data)
        except Exception as exc:
            readable_error = f"{page_name}: {exc}"
            signals.message.emit(f"Render preparation failed: {readable_error}")
            page_results.append(
                RenderPreparationPageResult(
                    image_relative_path=str(image_relative_path),
                    json_path=None,
                    error=str(exc),
                    summary={},
                )
            )
        else:
            page_results.append(
                RenderPreparationPageResult(
                    image_relative_path=str(image_relative_path),
                    json_path=json_path,
                    error=None,
                    summary=summary,
                )
            )
            signals.message.emit(f"Render metadata ready: {json_path}")

        signals.progress.emit(int((index / total_pages) * 100))

    successful_pages = [result for result in page_results if result.json_path is not None]
    if not successful_pages:
        first_error = page_results[0].error if page_results else "Unknown render preparation failure."
        raise RuntimeError(f"Render preparation failed for all pages. {first_error}")

    failed_count = len([result for result in page_results if result.error is not None])
    if failed_count:
        signals.message.emit(
            f"Render preparation finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
        )
    else:
        signals.message.emit(
            f"Render preparation finished successfully for {len(successful_pages)} page(s)."
        )

    return RenderPreparationWorkerResult(page_results=page_results)


def _run_render_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> RenderWorkerResult:
    if not isinstance(task, RenderTask):
        raise TypeError("Render worker received an unexpected task type.")

    if task.project is None:
        raise ValueError("No project was provided for rendering.")

    if not task.image_relative_paths:
        raise ValueError("No pages were provided for rendering.")

    if not isinstance(task.config, dict):
        raise ValueError("Render worker requires a configuration dictionary.")

    total_pages = len(task.image_relative_paths)
    page_results_map: dict[str, RenderPageResult] = {
        str(image_relative_path): RenderPageResult(image_relative_path=str(image_relative_path))
        for image_relative_path in task.image_relative_paths
    }

    def on_progress(event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("stage", "render")
        signals.event.emit(payload)
        event_name = str(event.get("event", "") or "")
        message = str(event.get("message", "") or "").strip()
        image_relative_path = str(event.get("image_relative_path", "") or "")
        if message:
            signals.message.emit(message)

        if event_name == "batch_page_start":
            page_index = max(int(event.get("page_index", 1) or 1), 1)
            page_total = max(int(event.get("page_total", total_pages) or total_pages), 1)
            signals.progress.emit(int(((page_index - 1) / page_total) * 100))
            return

        if event_name == "page_done" and image_relative_path:
            page_result = page_results_map.setdefault(
                image_relative_path,
                RenderPageResult(image_relative_path=image_relative_path),
            )
            output_path = event.get("output_path")
            if isinstance(output_path, str) and output_path.strip():
                page_result.image_path = Path(output_path)
            summary = event.get("summary")
            if isinstance(summary, dict):
                page_result.summary = dict(summary)

            processed_pages = len(
                [
                    result
                    for result in page_results_map.values()
                    if result.image_path is not None or result.error is not None
                ]
            )
            signals.progress.emit(int((processed_pages / max(total_pages, 1)) * 100))

    signals.message.emit(f"Starting render for {total_pages} page(s).")
    signals.progress.emit(0)

    if len(task.image_relative_paths) == 1:
        image_relative_path = task.image_relative_paths[0]
        page_name = Path(image_relative_path).name
        signals.message.emit(f"[1/1] Rendering {page_name}")
        try:
            image_path = run_render_for_page(
                task.project,
                image_relative_path,
                force=task.force,
                font_name=task.config.get("font_name"),
                font_path=task.config.get("font_path"),
                min_font_size=int(task.config.get("min_font_size", 12) or 12),
                max_font_size=int(task.config.get("max_font_size", 72) or 72),
                stroke_enabled=bool(task.config.get("stroke_enabled", True)),
                stroke_width=task.config.get("stroke_width"),
                text_color=task.config.get("text_color"),
                stroke_color=task.config.get("stroke_color"),
                auto_color=bool(task.config.get("auto_color", True)),
                auto_direction=bool(task.config.get("auto_direction", True)),
                vertical_cjk=bool(task.config.get("vertical_cjk", True)),
                save_sprites=bool(task.config.get("save_sprites", True)),
                logger=signals.message.emit,
                progress_callback=on_progress,
            )
            render_data = load_render_json(render_json_path(task.project, image_relative_path))
            summary = summarize_render_json(render_data)
        except Exception as exc:
            readable_error = f"{page_name}: {exc}"
            signals.message.emit(f"Render failed: {readable_error}")
            page_results_map[str(image_relative_path)] = RenderPageResult(
                image_relative_path=str(image_relative_path),
                image_path=None,
                error=str(exc),
                summary={},
            )
        else:
            page_results_map[str(image_relative_path)] = RenderPageResult(
                image_relative_path=str(image_relative_path),
                image_path=image_path,
                error=None,
                summary=summary,
            )
        signals.progress.emit(100)
    else:
        try:
            output_paths = run_render_for_pages(
                task.project,
                task.image_relative_paths,
                force=task.force,
                font_name=task.config.get("font_name"),
                font_path=task.config.get("font_path"),
                min_font_size=int(task.config.get("min_font_size", 12) or 12),
                max_font_size=int(task.config.get("max_font_size", 72) or 72),
                stroke_enabled=bool(task.config.get("stroke_enabled", True)),
                stroke_width=task.config.get("stroke_width"),
                text_color=task.config.get("text_color"),
                stroke_color=task.config.get("stroke_color"),
                auto_color=bool(task.config.get("auto_color", True)),
                auto_direction=bool(task.config.get("auto_direction", True)),
                vertical_cjk=bool(task.config.get("vertical_cjk", True)),
                save_sprites=bool(task.config.get("save_sprites", True)),
                logger=signals.message.emit,
                progress_callback=on_progress,
            )
        except Exception as exc:
            signals.message.emit(f"Render failed: {exc}")
        else:
            for output_path in output_paths:
                metadata_path = output_path.parent / f"{output_path.stem}.json"
                try:
                    render_data = load_render_json(metadata_path)
                except Exception:
                    continue
                source_image = str(render_data.get("source_image", "") or "")
                page_result = page_results_map.setdefault(
                    source_image,
                    RenderPageResult(image_relative_path=source_image),
                )
                page_result.image_path = output_path
                page_result.summary = summarize_render_json(render_data)
                if page_result.error is None and str(render_data.get("status", "")).lower() == "error":
                    page_result.error = str(render_data.get("error", "") or "Render failed.")

        for image_relative_path in task.image_relative_paths:
            page_key = str(image_relative_path)
            page_result = page_results_map.setdefault(
                page_key,
                RenderPageResult(image_relative_path=page_key),
            )
            if page_result.image_path is not None or page_result.error is not None:
                continue
            try:
                metadata_path = render_json_path(task.project, page_key)
                if metadata_path.exists():
                    render_data = load_render_json(metadata_path)
                    output_path = task.project.root_dir / str(render_data.get("output_image_path", "") or "")
                    if output_path.exists():
                        page_result.image_path = output_path
                    page_result.summary = summarize_render_json(render_data)
                    if str(render_data.get("status", "")).lower() == "error":
                        page_result.error = str(render_data.get("error", "") or "Render failed.")
            except Exception as exc:
                page_result.error = str(exc)

        signals.progress.emit(100)

    page_results = [page_results_map[str(image_relative_path)] for image_relative_path in task.image_relative_paths]
    successful_pages = [result for result in page_results if result.image_path is not None]
    if not successful_pages:
        first_error = page_results[0].error if page_results else "Unknown render failure."
        raise RuntimeError(f"Render failed for all pages. {first_error}")

    failed_count = len([result for result in page_results if result.error is not None])
    if failed_count:
        signals.message.emit(
            f"Render finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
        )
    else:
        signals.message.emit(f"Render finished successfully for {len(successful_pages)} page(s).")

    return RenderWorkerResult(page_results=page_results)


def _run_llama_server_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> LlamaServerTaskResult:
    if not isinstance(task, LlamaServerTask):
        raise TypeError("llama.cpp server worker received an unexpected task type.")

    if task.manager is None:
        raise ValueError("No llama.cpp server manager was provided.")

    action = str(task.action).strip().lower()
    signals.message.emit(f"Running llama.cpp server action: {action}")

    if action == "check":
        status = task.manager.check_server(timeout=min(float(task.timeout_seconds), 5.0))
    elif action == "start":
        status = task.manager.start_server(
            timeout=float(task.timeout_seconds),
            logger=signals.message.emit,
        )
    elif action == "stop":
        status = task.manager.stop_server(
            timeout=float(task.timeout_seconds),
            logger=signals.message.emit,
        )
    else:
        raise ValueError(f"Unsupported llama.cpp server action: {task.action}")

    if status.state == "Error":
        raise RuntimeError(status.message)

    return _status_to_task_result(status)


def _run_lama_model_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> LamaModelTaskResult:
    if not isinstance(task, LamaModelTask):
        raise TypeError("LaMa model worker received an unexpected task type.")

    action = str(task.action).strip().lower()
    signals.message.emit(f"Running LaMa model action: {action}")

    if action == "status":
        from mmt_core.inpaint_stage import get_lama_model_manager

        result = get_lama_model_manager().status()
    elif action == "load":
        result = load_lama_model(device=task.device, logger=signals.message.emit)
    elif action == "unload":
        result = unload_lama_model(logger=signals.message.emit)
    else:
        raise ValueError(f"Unsupported LaMa model action: {task.action}")

    return LamaModelTaskResult(
        loaded=bool(result.get("loaded", False)),
        device=str(result.get("device", "") or ""),
        message=str(result.get("message", "") or ""),
    )


def _run_export_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> ExportWorkerResult:
    if not isinstance(task, ExportTask):
        raise TypeError("Export worker received an unexpected task type.")

    if task.project is None:
        raise ValueError("No project was provided for export.")

    config = ExportConfig.from_value(task.config)
    signals.message.emit(
        f"Starting export: source={config.export_source}, scope={config.page_scope}, format={config.output_format}"
    )
    signals.progress.emit(0)

    def on_progress(event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("stage", "export")
        signals.event.emit(payload)
        message = str(event.get("message", "") or "").strip()
        if message:
            signals.message.emit(message)
        try:
            progress_value = int(event.get("progress", 0) or 0)
        except Exception:
            progress_value = 0
        signals.progress.emit(max(0, min(100, progress_value)))

    manifest = run_export(
        task.project,
        current_page=task.current_page,
        selected_pages=task.selected_pages,
        config=config,
        logger=signals.message.emit,
        progress_callback=on_progress,
    )
    signals.progress.emit(100)
    signals.message.emit(
        f"Export completed. Exported: {int(manifest.get('exported_count', 0) or 0)}, "
        f"skipped: {int(manifest.get('skipped_count', 0) or 0)}, "
        f"errors: {int(manifest.get('error_count', 0) or 0)}"
    )
    return ExportWorkerResult(manifest=manifest)


def _run_process_task(
    task: PipelineTask,
    signals: WorkerSignals,
) -> ProcessPipelineResult:
    if not isinstance(task, ProcessTask):
        raise TypeError("Process worker received an unexpected task type.")

    if task.project is None:
        raise ValueError("No project was provided for one-click processing.")

    if not task.image_relative_paths:
        raise ValueError("No pages were provided for one-click processing.")

    scope_label = "chapter" if str(task.scope or "").strip().lower() != "current" else "current page"
    thread_name = threading.current_thread().name
    signals.message.emit(
        f"Starting one-click process for {len(task.image_relative_paths)} page(s) "
        f"({scope_label}, force={bool(task.force)}) on {thread_name}. Export is excluded."
    )
    signals.progress.emit(0)

    cancel_token = task.cancel_token if hasattr(task, "cancel_token") else None
    cancel_check = cancel_token.is_set if hasattr(cancel_token, "is_set") else None

    def on_progress(event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("stage", "process")
        signals.event.emit(payload)
        try:
            progress_value = int(payload.get("overall_progress", 0) or 0)
        except Exception:
            progress_value = 0
        signals.progress.emit(max(0, min(100, progress_value)))

    result = run_process_pipeline(
        task.project,
        task.image_relative_paths,
        scope=task.scope,
        force=task.force,
        detection_config=task.detection_config,
        ocr_config=task.ocr_config,
        translation_config=task.translation_config,
        inpaint_settings=task.inpaint_settings,
        render_config=task.render_config,
        logger=signals.message.emit,
        progress_callback=on_progress,
        should_cancel=cancel_check,
    )
    if not result.canceled:
        signals.progress.emit(100)

    if result.canceled:
        signals.message.emit("One-click process canceled by user.")
    elif result.completed_with_errors:
        signals.message.emit(
            f"One-click process completed with errors. "
            f"Succeeded: {result.succeeded_pages}, failed: {result.failed_pages}."
        )
    else:
        signals.message.emit(
            f"One-click process completed successfully for {result.succeeded_pages} page(s)."
        )
    return result


def _status_to_task_result(status: LlamaServerStatus) -> LlamaServerTaskResult:
    return LlamaServerTaskResult(
        state=status.state,
        message=status.message,
        is_alive=status.is_alive,
        managed=status.managed,
    )
