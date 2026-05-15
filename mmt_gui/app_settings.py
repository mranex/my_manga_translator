"""Persistent application settings for the desktop workbench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSettings

APP_SETTINGS_ORGANIZATION = "MMT"
APP_SETTINGS_APPLICATION = "MangaTranslatorDesktop"
MAX_RECENT_PROJECTS = 10


class AppSettings:
    """Thin wrapper around QSettings with safe typed helpers."""

    def __init__(self) -> None:
        self._settings = QSettings(APP_SETTINGS_ORGANIZATION, APP_SETTINGS_APPLICATION)

    def sync(self) -> None:
        self._settings.sync()

    def string_value(self, key: str, default: str = "") -> str:
        value = self._settings.value(key, default)
        if value is None:
            return str(default)
        return str(value)

    def bool_value(self, key: str, default: bool = False) -> bool:
        value = self._settings.value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def int_value(self, key: str, default: int = 0) -> int:
        value = self._settings.value(key, default)
        try:
            return int(value)
        except Exception:
            return int(default)

    def bytes_value(self, key: str) -> bytes | None:
        value = self._settings.value(key)
        if value is None:
            return None
        try:
            return bytes(value)
        except Exception:
            return None

    def list_value(self, key: str, default: list[str] | None = None) -> list[str]:
        value = self._settings.value(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value]
        return list(default or [])

    def json_value(self, key: str, default: dict[str, Any] | list[Any] | None = None) -> Any:
        value = self._settings.value(key)
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str) or not value.strip():
            if isinstance(default, dict):
                return dict(default)
            if isinstance(default, list):
                return list(default)
            return default
        try:
            parsed = json.loads(value)
        except Exception:
            if isinstance(default, dict):
                return dict(default)
            if isinstance(default, list):
                return list(default)
            return default
        return parsed

    def set_value(self, key: str, value: Any) -> None:
        self._settings.setValue(key, value)

    def set_json_value(self, key: str, value: Any) -> None:
        self._settings.setValue(key, json.dumps(value, ensure_ascii=False))

    def remove(self, key: str) -> None:
        self._settings.remove(key)

    def last_project_path(self) -> str:
        return self.string_value("workspace/last_project_json_path", "")

    def set_last_project_path(self, project_path: str | Path | None) -> None:
        normalized = str(project_path or "").strip()
        if normalized:
            self.set_value("workspace/last_project_json_path", normalized)
        else:
            self.remove("workspace/last_project_json_path")

    def recent_projects(self) -> list[str]:
        raw_projects = self.list_value("workspace/recent_projects", [])
        unique_projects: list[str] = []
        seen: set[str] = set()
        for project_path in raw_projects:
            normalized = str(project_path).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_projects.append(normalized)
        return unique_projects[:MAX_RECENT_PROJECTS]

    def set_recent_projects(self, project_paths: list[str]) -> None:
        cleaned = self._normalize_recent_projects(project_paths)
        self.set_value("workspace/recent_projects", cleaned)

    def push_recent_project(self, project_path: str | Path) -> list[str]:
        normalized = str(project_path).strip()
        recent_projects = [normalized] + [path for path in self.recent_projects() if path != normalized]
        cleaned = self._normalize_recent_projects(recent_projects)
        self.set_value("workspace/recent_projects", cleaned)
        return cleaned

    def clear_recent_projects(self) -> None:
        self.remove("workspace/recent_projects")

    def panel_settings(self, panel_key: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = self.json_value(f"panels/{panel_key}", default or {})
        if isinstance(payload, dict):
            return dict(payload)
        return dict(default or {})

    def set_panel_settings(self, panel_key: str, payload: dict[str, Any]) -> None:
        self.set_json_value(f"panels/{panel_key}", payload)

    @staticmethod
    def _normalize_recent_projects(project_paths: list[str]) -> list[str]:
        normalized_projects: list[str] = []
        seen: set[str] = set()
        for project_path in project_paths:
            normalized = str(project_path).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_projects.append(normalized)
        return normalized_projects[:MAX_RECENT_PROJECTS]


__all__ = [
    "APP_SETTINGS_APPLICATION",
    "APP_SETTINGS_ORGANIZATION",
    "AppSettings",
    "MAX_RECENT_PROJECTS",
]
