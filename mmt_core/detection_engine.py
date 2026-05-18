"""Resident detection engine for preloaded detector ownership."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import os
from typing import Any

from detectors import (
    OgkMangaRTDetrManager,
    PageDetectionResult,
    get_pp_doclayout_v3_detector,
    get_yolov8_seg_bubble_detector,
)
from detectors.page_detector import detect_page_regions_layout_first
from .crash_logging import write_crash_breadcrumb
from .detection_config import (
    DETECTION_ENGINE_CLASSIC,
    DETECTION_ENGINE_MANGA_RTDETR,
    DetectionConfig,
)


Logger = Callable[[str], None] | None
StatusCallback = Callable[[str], None] | None


@dataclass(slots=True)
class DetectionEngine:
    """Owns the detector instances used by the desktop studio runtime."""

    bubble_detector: Any | None = None
    layout_detector: Any | None = None
    manga_detector_manager: OgkMangaRTDetrManager = field(default_factory=OgkMangaRTDetrManager)
    disable_pp_layout_for_debug: bool = field(init=False, default=False)
    disable_yolo_for_debug: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.disable_pp_layout_for_debug = _env_flag("MMT_DISABLE_PP_LAYOUT")
        self.disable_yolo_for_debug = _env_flag("MMT_DISABLE_YOLO_BUBBLE")

    def preload(
        self,
        detection_config: DetectionConfig | dict[str, Any] | None = None,
        *,
        logger: Logger = None,
        status_callback: StatusCallback = None,
    ) -> None:
        active_config = DetectionConfig.from_value(detection_config)
        if active_config.engine == DETECTION_ENGINE_CLASSIC:
            _status(status_callback, "Loading Classic PP+YOLO detection models...")
            _log(logger, "Loading Classic PP+YOLO detection models...")
            try:
                self._ensure_classic_detectors_loaded(logger=logger, status_callback=status_callback)
            except Exception as exc:
                failure_message = (
                    f"Classic PP+YOLO startup load failed: {exc}. "
                    "Switch to Manga RT-DETR or run Classic detection again after fixing the runtime."
                )
                _status(status_callback, failure_message)
                _log(logger, failure_message)
            return

        _log(logger, "Classic PP+YOLO detection is optional and will load when selected.")
        _status(status_callback, "Loading default Manga RT-DETR detector...")
        _log(logger, "Loading default Manga RT-DETR detector...")
        try:
            payload = self.load_manga_detector(active_config, logger=logger)
        except Exception as exc:
            failure_message = (
                f"Manga RT-DETR startup load failed: {exc}. "
                "Switch to Classic or reload Manga detector in Config."
            )
            self.manga_detector_manager.set_error(failure_message)
            _status(status_callback, failure_message)
            _log(logger, failure_message)
            return
        device = str(payload.get("device", active_config.manga_device) or active_config.manga_device or "auto")
        success_message = f"Manga RT-DETR detector loaded on {device}."
        _status(status_callback, success_message)
        _log(logger, success_message)

    def is_ready(self) -> bool:
        return True

    def missing_detectors(self) -> list[str]:
        missing: list[str] = []
        if self.layout_detector is None:
            missing.append("PPLayout")
        if self.bubble_detector is None:
            missing.append("YOLO bubble")
        return missing

    def clear(self) -> None:
        self.bubble_detector = None
        self.layout_detector = None
        try:
            self.manga_detector_manager.unload()
        except Exception:
            pass

    def load_manga_detector(
        self,
        detection_config: DetectionConfig | dict[str, Any] | None,
        *,
        logger: Logger = None,
    ) -> dict[str, Any]:
        return self.manga_detector_manager.load_once(
            DetectionConfig.from_value(detection_config),
            logger=logger,
        )

    def reload_manga_detector(
        self,
        detection_config: DetectionConfig | dict[str, Any] | None,
        *,
        logger: Logger = None,
    ) -> dict[str, Any]:
        return self.manga_detector_manager.reload(
            DetectionConfig.from_value(detection_config),
            logger=logger,
        )

    def unload_manga_detector(self) -> dict[str, Any]:
        return self.manga_detector_manager.unload()

    def manga_detector_status(self) -> dict[str, Any]:
        return self.manga_detector_manager.status()

    def ensure_detection_ready(
        self,
        detection_config: DetectionConfig | dict[str, Any] | None,
        *,
        logger: Logger = None,
        status_callback: StatusCallback = None,
    ) -> None:
        active_config = DetectionConfig.from_value(detection_config)
        if active_config.engine == DETECTION_ENGINE_MANGA_RTDETR:
            status = self.manga_detector_manager.status()
            if not bool(status.get("loaded", False)):
                raise RuntimeError(
                    "Manga RT-DETR detector is not loaded. Open Config and click Load Manga Detector first."
                )
            loaded_config = status.get("detection_config")
            if DetectionConfig.from_value(loaded_config).signature() != active_config.signature():
                raise RuntimeError(
                    "Manga RT-DETR detector config changed. Reload Manga Detector in Config before running Detection."
                )
            return

        self._ensure_classic_detectors_loaded(logger=logger, status_callback=status_callback)
        missing = self.missing_detectors()
        if missing:
            missing_text = ", ".join(missing)
            raise RuntimeError(
                "Classic PP+YOLO detection is unavailable. "
                f"Missing resident detectors: {missing_text}."
            )

    def detect_image(
        self,
        image: Any,
        *,
        detection_config: DetectionConfig | dict[str, Any] | None = None,
        logger: Logger = None,
        diagnostics_path: Any | None = None,
        page_name: str = "",
    ) -> PageDetectionResult:
        active_config = DetectionConfig.from_value(detection_config)
        current_page = page_name
        write_crash_breadcrumb(
            "DetectionEngine.detect_image entered",
            page=current_page,
            has_layout_detector=self.layout_detector is not None,
            has_bubble_detector=self.bubble_detector is not None,
            engine=active_config.engine,
        )
        del diagnostics_path, page_name
        self.ensure_detection_ready(active_config, logger=logger)
        if active_config.engine == DETECTION_ENGINE_MANGA_RTDETR:
            _log(logger, "Running resident OGK Manga RT-DETR inference...")
            write_crash_breadcrumb("before manga_detector_manager.detect_page", page=current_page)
            result = self.manga_detector_manager.detect_page(
                image,
                active_config,
                logger=logger,
            )
            write_crash_breadcrumb("after manga_detector_manager.detect_page", page=current_page)
            return result
        if active_config.engine != DETECTION_ENGINE_CLASSIC:
            raise ValueError(f"Unsupported detection engine: {active_config.engine}")
        _log(logger, "Running resident detection inference...")
        write_crash_breadcrumb("before detect_page_regions_layout_first", page=current_page)
        result = detect_page_regions_layout_first(
            image,
            layout_detector=self.layout_detector,
            bubble_detector=self.bubble_detector,
        )
        result.stats = {"engine": DETECTION_ENGINE_CLASSIC, **dict(result.stats or {})}
        write_crash_breadcrumb("after detect_page_regions_layout_first", page=current_page)
        return result

    def _ensure_classic_detectors_loaded(
        self,
        *,
        logger: Logger = None,
        status_callback: StatusCallback = None,
    ) -> None:
        if self.bubble_detector is None:
            if self.disable_yolo_for_debug:
                _status(
                    status_callback,
                    "MMT_DISABLE_YOLO_BUBBLE is set; YOLO bubble detector will remain unavailable.",
                )
                _log(
                    logger,
                    "MMT_DISABLE_YOLO_BUBBLE is set; Classic PP+YOLO detection will remain unavailable.",
                )
            else:
                _status(status_callback, "Loading YOLO bubble detector...")
                _log(logger, "Loading YOLO bubble detector...")
                self.bubble_detector = get_yolov8_seg_bubble_detector()
                if hasattr(self.bubble_detector, "load"):
                    self.bubble_detector.load()

        if self.layout_detector is None:
            if self.disable_pp_layout_for_debug:
                _status(status_callback, "MMT_DISABLE_PP_LAYOUT is set; PPLayout detector will remain unavailable.")
                _log(logger, "MMT_DISABLE_PP_LAYOUT is set; Classic PP+YOLO detection will remain unavailable.")
            else:
                _status(status_callback, "Loading PPLayout detector...")
                _log(logger, "Loading PPLayout detector...")
                self.layout_detector = get_pp_doclayout_v3_detector()
                if hasattr(self.layout_detector, "load"):
                    self.layout_detector.load()


def _log(logger: Logger, message: str) -> None:
    if logger is not None:
        logger(str(message or ""))


def _status(callback: StatusCallback, message: str) -> None:
    if callback is not None:
        callback(str(message or ""))


def _env_flag(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


__all__ = ["DetectionEngine"]
