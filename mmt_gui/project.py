"""Lightweight project storage for the PyQt6 desktop shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

PROJECT_FILENAME = "project.json"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CACHE_STAGES = ("detection", "ocr", "ocr_crops", "translation", "inpaint", "render", "render_sprites", "masks")
PROJECT_TRASH_DIRNAME = ".trash"


@dataclass(slots=True)
class ProjectPage:
    """Serializable per-page project state."""

    source_path: str
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectPage" | None:
        source_path = payload.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            return None

        stages = payload.get("stages", {})
        if not isinstance(stages, dict):
            stages = {}

        normalized_stages: dict[str, dict[str, Any]] = {}
        for stage_name, stage_payload in stages.items():
            if not isinstance(stage_name, str):
                continue
            if isinstance(stage_payload, dict):
                normalized_stages[stage_name] = {
                    str(key): value for key, value in stage_payload.items()
                }
            else:
                normalized_stages[stage_name] = {}

        return cls(source_path=source_path, stages=normalized_stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "stages": dict(self.stages),
        }


@dataclass(slots=True)
class ProjectData:
    """Serializable GUI project state."""

    name: str
    pages: list[ProjectPage] = field(default_factory=list)
    current_page_index: int = 0
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def source_images(self) -> list[str]:
        return [page.source_path for page in self.pages]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectData":
        settings = payload.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}

        current_page_index = payload.get("current_page_index", 0)
        if not isinstance(current_page_index, int):
            current_page_index = 0

        name = payload.get("name", "")
        if not isinstance(name, str) or not name.strip():
            name = "Untitled Project"

        pages = _load_pages(payload.get("pages"), payload.get("source_images"))

        return cls(
            name=name.strip(),
            pages=pages,
            current_page_index=current_page_index,
            settings=settings,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_images": self.source_images,
            "pages": [page.to_dict() for page in self.pages],
            "current_page_index": self.current_page_index,
            "settings": dict(self.settings),
        }


class MangaProject:
    """Represents a project folder and its JSON-backed UI state."""

    def __init__(self, root_dir: Path, data: ProjectData) -> None:
        self.root_dir = root_dir.resolve()
        self.data = data
        self.ensure_structure()
        self._normalize_state()

    @property
    def project_file(self) -> Path:
        return self.root_dir / PROJECT_FILENAME

    @property
    def source_dir(self) -> Path:
        return self.root_dir / "source"

    @property
    def cache_dir(self) -> Path:
        return self.root_dir / "cache"

    @property
    def page_count(self) -> int:
        return len(self.data.pages)

    @classmethod
    def create(cls, root_dir: Path, name: str | None = None) -> "MangaProject":
        project_name = name.strip() if isinstance(name, str) and name.strip() else root_dir.name
        project = cls(root_dir, ProjectData(name=project_name))
        project.save()
        return project

    @classmethod
    def load(cls, project_file: Path) -> "MangaProject":
        project_path = project_file.resolve()
        payload = json.loads(project_path.read_text(encoding="utf-8"))
        data = ProjectData.from_dict(payload)
        return cls(project_path.parent, data)

    def ensure_structure(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        for stage in CACHE_STAGES:
            (self.cache_dir / stage).mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self._normalize_state()
        payload = self.data.to_dict()
        self.project_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def import_images(self, image_paths: Sequence[Path]) -> list[str]:
        imported_images: list[str] = []
        had_pages = self.page_count > 0

        for image_path in image_paths:
            source_path = Path(image_path)
            if not source_path.exists():
                continue

            if source_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            destination_name = self._build_unique_source_name(source_path.name)
            destination_path = self.source_dir / destination_name
            shutil.copy2(source_path, destination_path)

            relative_path = destination_path.relative_to(self.root_dir).as_posix()
            self.data.pages.append(ProjectPage(source_path=relative_path))
            imported_images.append(relative_path)

        if imported_images and not had_pages:
            self.data.current_page_index = 0

        self._normalize_state()
        return imported_images

    def page_display_names(self) -> list[str]:
        return [Path(page.source_path).name for page in self.data.pages]

    def page_relative_path_for_index(self, index: int) -> str | None:
        page = self.page_for_index(index)
        return page.source_path if page is not None else None

    def image_path_for_index(self, index: int) -> Path | None:
        relative_path = self.page_relative_path_for_index(index)
        if relative_path is None:
            return None

        return self.root_dir / relative_path

    def page_for_index(self, index: int) -> ProjectPage | None:
        if index < 0 or index >= self.page_count:
            return None
        return self.data.pages[index]

    def all_image_paths(self) -> list[Path]:
        return [self.root_dir / page.source_path for page in self.data.pages]

    def relative_source_path(self, image_path: Path | str) -> str | None:
        target_path = Path(image_path).resolve()

        try:
            relative_path = target_path.relative_to(self.root_dir)
        except ValueError:
            return None

        normalized = self._normalize_source_entry(relative_path.as_posix())
        for page in self.data.pages:
            if page.source_path == normalized:
                return normalized
        return None

    def stage_metadata(self, image_relative_path: str, stage_name: str) -> dict[str, Any] | None:
        page = self.page_for_source_path(image_relative_path)
        if page is None:
            return None
        return page.stages.get(stage_name)

    def update_stage_status(
        self,
        image_relative_path: str,
        stage_name: str,
        *,
        status: str,
        cache_path: str | None = None,
        error: str | None = None,
    ) -> None:
        page = self.page_for_source_path(image_relative_path)
        if page is None:
            return

        existing_payload = page.stages.get(stage_name, {})
        stage_payload: dict[str, Any] = dict(existing_payload) if isinstance(existing_payload, dict) else {}
        stage_payload["status"] = status
        if cache_path:
            stage_payload["cache_path"] = str(cache_path)
        else:
            stage_payload.pop("cache_path", None)
        if error:
            stage_payload["error"] = str(error)
        else:
            stage_payload.pop("error", None)
        stage_payload.pop("stale", None)

        page.stages[stage_name] = stage_payload

    def set_stage_stale(self, image_relative_path: str, stage_name: str, stale: bool = True) -> None:
        page = self.page_for_source_path(image_relative_path)
        if page is None:
            return

        existing_payload = page.stages.get(stage_name, {})
        stage_payload: dict[str, Any] = dict(existing_payload) if isinstance(existing_payload, dict) else {}
        if stale:
            stage_payload["stale"] = True
        else:
            stage_payload.pop("stale", None)
        page.stages[stage_name] = stage_payload

    def page_for_source_path(self, image_relative_path: str) -> ProjectPage | None:
        normalized = self._normalize_source_entry(image_relative_path)
        for page in self.data.pages:
            if page.source_path == normalized:
                return page
        return None

    def set_current_page(self, index: int) -> None:
        self.data.current_page_index = index
        self._normalize_state()

    def reorder_pages(self, ordered_source_paths: Sequence[str]) -> None:
        """Reorder project pages without renaming or moving source/cache files."""

        if not isinstance(ordered_source_paths, Sequence):
            raise ValueError("Ordered page list must be a sequence.")

        normalized_order = [self._normalize_source_entry(path) for path in ordered_source_paths]
        if len(normalized_order) != self.page_count:
            raise ValueError("Page reorder list must include every project page exactly once.")

        current_paths = [page.source_path for page in self.data.pages]
        if set(normalized_order) != set(current_paths) or len(set(normalized_order)) != len(current_paths):
            raise ValueError("Page reorder list does not match the current project pages.")

        current_selected_path = None
        if 0 <= self.data.current_page_index < self.page_count:
            current_selected_path = self.data.pages[self.data.current_page_index].source_path

        pages_by_path = {page.source_path: page for page in self.data.pages}
        self.data.pages = [pages_by_path[path] for path in normalized_order]
        if current_selected_path is not None and current_selected_path in normalized_order:
            self.data.current_page_index = normalized_order.index(current_selected_path)
        self._normalize_state()

    def _normalize_state(self) -> None:
        normalized_pages: list[ProjectPage] = []
        seen_source_paths: set[str] = set()

        for page in self.data.pages:
            normalized_source = self._normalize_source_entry(page.source_path)
            if normalized_source in seen_source_paths:
                continue
            seen_source_paths.add(normalized_source)
            normalized_pages.append(
                ProjectPage(
                    source_path=normalized_source,
                    stages=_normalize_stage_payload(page.stages),
                )
            )

        self.data.pages = normalized_pages

        if not self.data.pages:
            self.data.current_page_index = 0
            return

        self.data.current_page_index = max(0, min(self.data.current_page_index, self.page_count - 1))

    def _normalize_source_entry(self, source_image: str) -> str:
        relative_path = Path(source_image)

        if relative_path.is_absolute():
            try:
                relative_path = relative_path.relative_to(self.root_dir)
            except ValueError:
                relative_path = Path("source") / relative_path.name
        elif not relative_path.parts or relative_path.parts[0] != "source":
            relative_path = Path("source") / relative_path.name

        return relative_path.as_posix()

    def _build_unique_source_name(self, original_name: str) -> str:
        original_path = Path(original_name)
        stem = original_path.stem
        suffix = original_path.suffix
        candidate_name = original_path.name
        counter = 1

        while (self.source_dir / candidate_name).exists():
            candidate_name = f"{stem}_{counter}{suffix}"
            counter += 1

        return candidate_name


def _load_pages(
    raw_pages: Any,
    raw_source_images: Any,
) -> list[ProjectPage]:
    pages: list[ProjectPage] = []

    if isinstance(raw_pages, list):
        for raw_page in raw_pages:
            if isinstance(raw_page, dict):
                page = ProjectPage.from_dict(raw_page)
                if page is not None:
                    pages.append(page)

    if pages:
        return pages

    source_images: list[str] = []
    if isinstance(raw_source_images, list):
        source_images = [
            str(item)
            for item in raw_source_images
            if isinstance(item, str) and item.strip()
        ]

    return [ProjectPage(source_path=source_path) for source_path in source_images]


def _normalize_stage_payload(stages: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}

    for stage_name, stage_payload in stages.items():
        if not isinstance(stage_name, str):
            continue

        if isinstance(stage_payload, dict):
            normalized[stage_name] = {
                str(key): value for key, value in stage_payload.items()
            }
        else:
            normalized[stage_name] = {}

    return normalized


def collect_page_cache_paths(project: MangaProject, image_relative_path: str) -> list[Path]:
    """Return all known per-page cache files and folders for one project page."""

    normalized_source_path, _page, _index = _resolve_project_page(project, image_relative_path)
    page_stem = Path(normalized_source_path).stem
    cache_dir = project.cache_dir
    return [
        cache_dir / "detection" / f"{page_stem}.json",
        cache_dir / "ocr" / f"{page_stem}.json",
        cache_dir / "translation" / f"{page_stem}.json",
        cache_dir / "inpaint" / f"{page_stem}.json",
        cache_dir / "inpaint" / f"{page_stem}.png",
        cache_dir / "render" / f"{page_stem}.json",
        cache_dir / "render" / f"{page_stem}.png",
        cache_dir / "masks" / page_stem,
        cache_dir / "ocr_crops" / page_stem,
        cache_dir / "render_sprites" / page_stem,
    ]


def move_project_file_to_trash(project: MangaProject, path: Path, category: str) -> Path:
    """Move one project-local file or folder into the project trash."""

    return _move_project_path_to_trash(
        project,
        path,
        category,
        prefix=_trash_timestamp(),
    )


def remove_page_from_project(
    project: MangaProject,
    image_relative_path: str,
    *,
    move_to_trash: bool = True,
    allow_missing_source: bool = False,
) -> dict[str, Any]:
    """Remove one page from a project and optionally move its files into project-local trash."""

    return remove_pages_from_project(
        project,
        [image_relative_path],
        move_to_trash=move_to_trash,
        allow_missing_source=allow_missing_source,
    )


def remove_pages_from_project(
    project: MangaProject,
    image_relative_paths: Sequence[str],
    *,
    move_to_trash: bool = True,
    allow_missing_source: bool = False,
) -> dict[str, Any]:
    """Remove one or more pages from a project safely."""

    result: dict[str, Any] = {
        "removed_pages": [],
        "trash_paths": [],
        "skipped_paths": [],
        "errors": [],
        "selected_next_page": None,
    }

    if not isinstance(image_relative_paths, Sequence):
        result["errors"].append("No page paths were provided for removal.")
        return result

    requested_paths: list[str] = []
    seen_paths: set[str] = set()
    for raw_path in image_relative_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            result["errors"].append("Encountered an invalid page path while removing pages.")
            continue
        requested_paths.append(raw_path.strip())

    removal_specs: list[dict[str, Any]] = []
    for requested_path in requested_paths:
        try:
            normalized_source_path, _page, page_index = _resolve_project_page(project, requested_path)
        except Exception as exc:
            result["errors"].append(str(exc))
            continue

        if normalized_source_path in seen_paths:
            continue
        seen_paths.add(normalized_source_path)

        source_path = project.root_dir / normalized_source_path
        source_exists = source_path.exists()
        if move_to_trash and not source_exists and not allow_missing_source:
            result["errors"].append(
                f"Source image is missing for {normalized_source_path}. Remove the page entry only if you intend a metadata-only cleanup."
            )
            continue

        removal_specs.append(
            {
                "normalized_source_path": normalized_source_path,
                "index": page_index,
                "source_path": source_path,
                "cache_paths": collect_page_cache_paths(project, normalized_source_path),
                "trash_prefix": _trash_timestamp(),
            }
        )

    if not removal_specs:
        return result

    original_pages = list(project.data.pages)
    original_current_page_index = int(project.data.current_page_index)
    moved_entries: list[tuple[Path, Path]] = []
    removed_indices: list[int] = []
    removed_pages: list[str] = []

    for spec in removal_specs:
        normalized_source_path = str(spec["normalized_source_path"])
        source_path = Path(spec["source_path"])
        cache_paths = [Path(path) for path in spec["cache_paths"]]
        trash_prefix = str(spec["trash_prefix"])

        if move_to_trash:
            if source_path.exists():
                try:
                    source_trash_path = _move_project_path_to_trash(project, source_path, "pages", prefix=trash_prefix)
                except Exception as exc:
                    result["errors"].append(f"Failed to move source image for {normalized_source_path}: {exc}")
                    continue
                moved_entries.append((source_trash_path, source_path))
                result["trash_paths"].append(source_trash_path)
            else:
                result["skipped_paths"].append(source_path)

            for cache_path in cache_paths:
                if not cache_path.exists():
                    result["skipped_paths"].append(cache_path)
                    continue
                try:
                    cache_trash_path = _move_project_path_to_trash(
                        project,
                        cache_path,
                        _trash_category_for_project_path(project, cache_path),
                        prefix=trash_prefix,
                    )
                except Exception as exc:
                    result["errors"].append(f"Failed to move cache path for {normalized_source_path}: {cache_path} ({exc})")
                    continue
                moved_entries.append((cache_trash_path, cache_path))
                result["trash_paths"].append(cache_trash_path)

        removed_indices.append(int(spec["index"]))
        removed_pages.append(normalized_source_path)

    if not removed_pages:
        return result

    removed_index_set = set(removed_indices)
    project.data.pages = [
        page
        for index, page in enumerate(original_pages)
        if index not in removed_index_set
    ]
    project.data.current_page_index = _next_page_index_after_removal(
        original_current_page_index,
        removed_indices,
        len(project.data.pages),
    )
    project._normalize_state()

    try:
        project.save()
    except Exception as exc:
        project.data.pages = original_pages
        project.data.current_page_index = original_current_page_index
        project._normalize_state()
        rollback_errors = _rollback_moved_entries(moved_entries)
        result["errors"].append(f"Failed to save project after removing pages: {exc}")
        result["errors"].extend(rollback_errors)
        result["removed_pages"] = []
        result["trash_paths"] = []
        result["selected_next_page"] = None
        return result

    result["removed_pages"] = removed_pages
    next_page = project.page_relative_path_for_index(project.data.current_page_index)
    result["selected_next_page"] = next_page
    return result


def _resolve_project_page(project: MangaProject, image_relative_path: str) -> tuple[str, ProjectPage, int]:
    normalized_source_path = project._normalize_source_entry(image_relative_path)
    for index, page in enumerate(project.data.pages):
        if page.source_path == normalized_source_path:
            return normalized_source_path, page, index
    raise ValueError(f"Page not found in project: {image_relative_path}")


def _move_project_path_to_trash(
    project: MangaProject,
    path: Path,
    category: str,
    *,
    prefix: str,
) -> Path:
    source_path = _project_local_path(project, path)
    if not source_path.exists():
        raise FileNotFoundError(f"Project path does not exist: {source_path}")

    category_path = _validate_trash_category(category)
    trash_dir = project.root_dir / PROJECT_TRASH_DIRNAME / category_path
    try:
        trash_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to create project trash folder: {trash_dir}") from exc

    target_path = _unique_trash_target(trash_dir, source_path.name, prefix=prefix)
    shutil.move(str(source_path), str(target_path))
    return target_path


def _trash_category_for_project_path(project: MangaProject, path: Path) -> str:
    normalized_path = _project_local_path(project, path)
    relative_path = normalized_path.relative_to(project.root_dir)
    parts = relative_path.parts
    if len(parts) < 2 or parts[0] != "cache":
        raise ValueError(f"Only cache paths can be moved into cache trash categories: {normalized_path}")
    return Path(*parts[:2]).as_posix()


def _project_local_path(project: MangaProject, path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project.root_dir / candidate
    normalized = candidate.resolve(strict=False)
    project_root = project.root_dir.resolve(strict=False)
    try:
        normalized.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to move a path outside the project folder: {candidate}") from exc
    return normalized


def _validate_trash_category(category: str) -> Path:
    category_path = Path(str(category or "").strip().replace("\\", "/"))
    if not category_path.parts or category_path.is_absolute():
        raise ValueError(f"Invalid project trash category: {category}")
    if any(part in {"", ".", ".."} for part in category_path.parts):
        raise ValueError(f"Invalid project trash category: {category}")
    return category_path


def _unique_trash_target(trash_dir: Path, original_name: str, *, prefix: str) -> Path:
    safe_prefix = str(prefix or _trash_timestamp()).strip("_") or _trash_timestamp()
    candidate_name = f"{safe_prefix}_{original_name}"
    candidate_path = trash_dir / candidate_name
    counter = 1
    while candidate_path.exists():
        candidate_path = trash_dir / f"{safe_prefix}_{counter}_{original_name}"
        counter += 1
    return candidate_path


def _next_page_index_after_removal(
    current_page_index: int,
    removed_indices: Sequence[int],
    remaining_page_count: int,
) -> int:
    if remaining_page_count <= 0:
        return 0

    removed_set = {int(index) for index in removed_indices}
    removed_before_current = sum(1 for index in removed_set if index < int(current_page_index))
    if int(current_page_index) in removed_set:
        return max(0, min(int(current_page_index) - removed_before_current, remaining_page_count - 1))
    return max(0, min(int(current_page_index) - removed_before_current, remaining_page_count - 1))


def _rollback_moved_entries(moved_entries: Sequence[tuple[Path, Path]]) -> list[str]:
    errors: list[str] = []
    for trash_path, original_path in reversed(list(moved_entries)):
        if not trash_path.exists():
            continue
        try:
            original_path.parent.mkdir(parents=True, exist_ok=True)
            if original_path.exists():
                errors.append(
                    f"Rollback skipped because the original path already exists: {original_path}"
                )
                continue
            shutil.move(str(trash_path), str(original_path))
        except Exception as exc:
            errors.append(f"Failed to restore {original_path} from project trash: {exc}")
    return errors


def _trash_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
