"""External local OCR server helpers for llama.cpp-backed OCR providers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any
from urllib.parse import urlparse

from .ocr_models import (
    OCR_PROVIDER_DEEPSEEK_OCR_LLAMA,
    OCR_PROVIDER_LABELS,
    OCR_PROVIDER_PADDLE_VL_LLAMA,
)


@dataclass(slots=True)
class LlamaServerStatus:
    """Structured status returned by the external OCR server helper."""

    state: str
    message: str
    is_alive: bool
    managed: bool = False


class LlamaServerManager:
    """Builds, checks, and launches external OCR server scripts without owning the process."""

    def __init__(
        self,
        *,
        workspace_root: str | Path = "",
        server_url: str = "http://127.0.0.1:8080",
        host: str = "127.0.0.1",
        port: int = 8080,
        model_path: str | Path = "",
        mmproj_path: str | Path = "",
        llama_cpp_dir: str | Path = "",
        gpu_layers: int = 99,
        ctx_size: int = 8192,
        temperature: float = 0.0,
        extra_args: str = "",
        provider_key: str = "",
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve() if str(workspace_root).strip() else None
        self.server_url = ""
        self.host = "127.0.0.1"
        self.port = 8080
        self.model_path = ""
        self.mmproj_path = ""
        self.llama_cpp_dir = ""
        self.gpu_layers = 99
        self.ctx_size = 8192
        self.temperature = 0.0
        self.extra_args = ""
        self.provider_key = ""
        self.update_config(
            server_url=server_url,
            host=host,
            port=port,
            model_path=model_path,
            mmproj_path=mmproj_path,
            llama_cpp_dir=llama_cpp_dir,
            gpu_layers=gpu_layers,
            ctx_size=ctx_size,
            temperature=temperature,
            extra_args=extra_args,
            provider_key=provider_key,
        )

    def update_config(
        self,
        *,
        server_url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        model_path: str | Path | None = None,
        mmproj_path: str | Path | None = None,
        llama_cpp_dir: str | Path | None = None,
        gpu_layers: int | None = None,
        ctx_size: int | None = None,
        temperature: float | None = None,
        extra_args: str | None = None,
        provider_key: str | None = None,
    ) -> None:
        if host is not None or port is not None:
            resolved_host = str(host or self.host or "127.0.0.1").strip() or "127.0.0.1"
            resolved_port = _coerce_port(port if port is not None else self.port)
            self.host = resolved_host
            self.port = resolved_port
            self.server_url = f"http://{self.host}:{self.port}"
        elif server_url is not None:
            normalized_url, parsed_host, parsed_port = self._normalize_server_url(server_url)
            self.server_url = normalized_url
            self.host = parsed_host
            self.port = parsed_port
        else:
            self.server_url = f"http://{self.host}:{self.port}"

        if model_path is not None:
            self.model_path = str(model_path or "").strip()
        if mmproj_path is not None:
            self.mmproj_path = str(mmproj_path or "").strip()
        if llama_cpp_dir is not None:
            self.llama_cpp_dir = str(llama_cpp_dir or "").strip()
        if gpu_layers is not None:
            self.gpu_layers = int(gpu_layers)
        if ctx_size is not None:
            self.ctx_size = max(1, int(ctx_size))
        if temperature is not None:
            self.temperature = float(temperature)
        if extra_args is not None:
            self.extra_args = str(extra_args or "").strip()
        if provider_key is not None:
            self.provider_key = str(provider_key or "").strip()

    def build_command(
        self,
        *,
        disable_prompt_cache: bool | None = None,
        validate_paths: bool = True,
    ) -> list[str]:
        model_file = self._resolve_path_argument(self.model_path, "model.gguf", validate=validate_paths)
        mmproj_file = self._resolve_path_argument(self.mmproj_path, "mmproj.gguf", validate=validate_paths)
        binary_path = self.resolve_binary_path(allow_missing=not validate_paths)

        command = [
            str(binary_path),
            "-m",
            str(model_file),
            "--mmproj",
            str(mmproj_file),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-c",
            str(self.ctx_size),
            "-ngl",
            str(self.gpu_layers),
            "--temp",
            _format_float(self.temperature),
        ]
        if disable_prompt_cache is None:
            disable_prompt_cache = self._should_disable_prompt_cache()
        if disable_prompt_cache:
            command.extend(
                [
                    "--no-cache-prompt",
                    "--cache-ram",
                    "0",
                    "--no-cache-idle-slots",
                ]
            )
        command.extend(self._parse_extra_args(self.extra_args))
        return command

    def build_bat_content(self) -> str:
        command = self.build_command(validate_paths=False)
        working_directory = self._resolve_working_directory_for_bat(command[0])
        bat_lines = ["@echo off"]
        if working_directory is not None:
            bat_lines.append(f'cd /d "{working_directory}"')

        command_lines: list[str] = []
        for index, token in enumerate(command):
            rendered = subprocess.list2cmdline([token])
            suffix = " ^" if index < len(command) - 1 else ""
            prefix = "call " if index == 0 else "  "
            command_lines.append(f"{prefix}{rendered}{suffix}")
        bat_lines.extend(command_lines)
        bat_lines.extend(
            [
                "echo.",
                "echo Server stopped. Press any key to close this window.",
                "pause >nul",
            ]
        )
        return "\n".join(bat_lines) + "\n"

    def provider_server_dir(self) -> Path:
        if self.workspace_root is None:
            raise RuntimeError("A workspace root is required to manage run_server.bat files.")
        return (self.workspace_root / "servers" / self._provider_folder_name()).resolve()

    def run_server_bat_path(self) -> Path:
        return self.provider_server_dir() / "run_server.bat"

    def write_run_server_bat(self) -> Path:
        bat_path = self.run_server_bat_path()
        bat_path.parent.mkdir(parents=True, exist_ok=True)
        bat_path.write_text(self.build_bat_content(), encoding="utf-8")
        return bat_path

    def check_run_server_bat(self) -> Path | None:
        bat_path = self.run_server_bat_path()
        return bat_path if bat_path.exists() else None

    def open_server_folder(self) -> Path:
        folder_path = self.provider_server_dir()
        folder_path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(folder_path))
        else:
            subprocess.Popen(["xdg-open", str(folder_path)])
        return folder_path

    def start_run_server_bat_external(self) -> Path:
        bat_path = self.run_server_bat_path()
        if not bat_path.exists():
            raise FileNotFoundError("run_server.bat does not exist. Use Create run_server.bat first.")
        if os.name == "nt":
            os.startfile(str(bat_path))
        else:
            subprocess.Popen(["sh", str(bat_path)])
        return bat_path

    def check_health(self, *, timeout: float = 3.0) -> LlamaServerStatus:
        alive, message = self._probe_server(timeout=timeout)
        return LlamaServerStatus(
            state="Ready" if alive else "Stopped",
            message=message,
            is_alive=alive,
            managed=False,
        )

    def check_server(self, *, timeout: float = 3.0) -> LlamaServerStatus:
        return self.check_health(timeout=timeout)

    def start_server(
        self,
        *,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
        logger: Callable[[str], None] | None = None,
    ) -> LlamaServerStatus:
        del timeout, poll_interval
        self.start_run_server_bat_external()
        message = "Started run_server.bat externally. Use Check health server to confirm readiness."
        self._log(logger, message)
        return LlamaServerStatus(
            state="Starting",
            message=message,
            is_alive=False,
            managed=False,
        )

    def stop_server(
        self,
        *,
        timeout: float = 10.0,
        logger: Callable[[str], None] | None = None,
    ) -> LlamaServerStatus:
        del timeout
        message = "Local OCR server is external. Stop it from its own terminal/window."
        self._log(logger, message)
        return LlamaServerStatus(
            state="Stopped",
            message=message,
            is_alive=False,
            managed=False,
        )

    def create_run_server_bat_status(self) -> LlamaServerStatus:
        bat_path = self.run_server_bat_path()
        existed_before = bat_path.exists()
        bat_path = self.write_run_server_bat()
        verb = "Updated" if existed_before else "Created"
        return LlamaServerStatus(
            state="Unknown",
            message=f"{verb} run_server.bat: {bat_path}",
            is_alive=False,
            managed=False,
        )

    def check_run_server_bat_status(self) -> LlamaServerStatus:
        bat_path = self.check_run_server_bat()
        if bat_path is not None:
            return LlamaServerStatus(
                state="Unknown",
                message=f"Found run_server.bat: {bat_path}",
                is_alive=False,
                managed=False,
            )
        return LlamaServerStatus(
            state="Unknown",
            message=f"run_server.bat does not exist for {self.provider_display_name()}.",
            is_alive=False,
            managed=False,
        )

    def open_server_folder_status(self) -> LlamaServerStatus:
        folder_path = self.open_server_folder()
        return LlamaServerStatus(
            state="Unknown",
            message=f"Opened server folder: {folder_path}",
            is_alive=False,
            managed=False,
        )

    def resolve_binary_path(self, *, allow_missing: bool = False) -> Path:
        search_names = ("llama-server.exe", "llama-server")
        configured_path = Path(self.llama_cpp_dir).expanduser() if self.llama_cpp_dir else None

        if configured_path is not None and configured_path.exists():
            if configured_path.is_file():
                return configured_path.resolve()
            for executable_name in search_names:
                direct_candidate = configured_path / executable_name
                if direct_candidate.exists():
                    return direct_candidate.resolve()
                nested_candidate = configured_path / "build" / "bin" / executable_name
                if nested_candidate.exists():
                    return nested_candidate.resolve()
                release_candidate = configured_path / "build" / "bin" / "Release" / executable_name
                if release_candidate.exists():
                    return release_candidate.resolve()

        for executable_name in search_names:
            executable_path = shutil.which(executable_name)
            if executable_path:
                return Path(executable_path).resolve()

        if allow_missing:
            if configured_path is not None and str(configured_path).strip():
                if configured_path.suffix.lower() == ".exe" or configured_path.name.lower().startswith("llama-server"):
                    return configured_path
                executable_name = "llama-server.exe" if os.name == "nt" else "llama-server"
                return configured_path / executable_name
            return Path("llama-server.exe" if os.name == "nt" else "llama-server")

        raise FileNotFoundError(
            "Could not find the llama-server binary. Check the llama.cpp folder or executable path."
        )

    def provider_display_name(self) -> str:
        if self.provider_key == OCR_PROVIDER_DEEPSEEK_OCR_LLAMA:
            return "DeepSeek OCR"
        if self.provider_key == OCR_PROVIDER_PADDLE_VL_LLAMA:
            return "PaddleOCR-VL"
        return OCR_PROVIDER_LABELS.get(self.provider_key, "Local OCR")

    def _normalize_server_url(self, raw_url: str) -> tuple[str, str, int]:
        normalized_input = str(raw_url or "").strip() or f"http://{self.host}:{self.port}"
        if "://" not in normalized_input:
            normalized_input = f"http://{normalized_input}"

        parsed = urlparse(normalized_input)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or 8080)
        scheme = parsed.scheme or "http"
        return f"{scheme}://{host}:{port}", host, port

    def _probe_server(self, *, timeout: float) -> tuple[bool, str]:
        requests = self._requests_module()
        request_error = self._request_exception_class(requests)
        base_url = self.server_url.rstrip("/")
        endpoints = (
            f"{base_url}/health",
            f"{base_url}/v1/models",
        )

        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=float(timeout))
            except request_error:
                continue

            if 200 <= int(response.status_code) < 400:
                return True, f"Local OCR server is reachable at {self.server_url}."

        return False, f"Local OCR server is not reachable at {self.server_url}. Start run_server.bat first."

    def _resolve_working_directory_for_bat(self, executable_path: str) -> Path | None:
        configured_path = Path(self.llama_cpp_dir).expanduser() if self.llama_cpp_dir else None
        if configured_path is not None:
            if configured_path.exists() and configured_path.is_dir():
                return configured_path.resolve()
            if configured_path.exists() and configured_path.is_file():
                return configured_path.resolve().parent
            if configured_path.suffix.lower() == ".exe":
                return configured_path.parent
            if str(configured_path).strip():
                return configured_path
        executable = Path(executable_path).expanduser()
        if executable.exists():
            return executable.resolve().parent
        return None

    def _resolve_path_argument(self, raw_path: str | Path, label: str, *, validate: bool) -> Path:
        candidate = Path(raw_path).expanduser()
        if not str(candidate).strip():
            raise FileNotFoundError(f"Missing required {label} path.")
        if validate and not candidate.exists():
            raise FileNotFoundError(f"Missing required {label} file: {candidate}")
        return candidate.resolve() if validate else candidate

    def _provider_folder_name(self) -> str:
        if self.provider_key == OCR_PROVIDER_DEEPSEEK_OCR_LLAMA:
            return "deepseek_ocr"
        if self.provider_key == OCR_PROVIDER_PADDLE_VL_LLAMA:
            return "paddleocr_vl"
        return "local_ocr"

    def _should_disable_prompt_cache(self) -> bool:
        return self.provider_key == OCR_PROVIDER_DEEPSEEK_OCR_LLAMA

    def _parse_extra_args(self, raw_args: str) -> list[str]:
        normalized = str(raw_args or "").strip()
        if not normalized:
            return []
        try:
            return shlex.split(normalized, posix=False)
        except ValueError as exc:
            raise ValueError(f"Invalid extra args: {exc}") from exc

    def _log(self, logger: Callable[[str], None] | None, message: str) -> None:
        if logger is not None:
            logger(message)

    def _requests_module(self) -> Any:
        try:
            import requests
        except Exception as exc:
            raise RuntimeError(
                "The 'requests' package is required to communicate with the local OCR server."
            ) from exc
        return requests

    def _request_exception_class(self, requests_module: Any) -> type[Exception]:
        request_error = getattr(requests_module, "RequestException", None)
        if isinstance(request_error, type) and issubclass(request_error, Exception):
            return request_error
        return Exception


def _coerce_port(value: Any) -> int:
    try:
        parsed = int(value)
    except Exception:
        return 8080
    return parsed if parsed > 0 else 8080


def _format_float(value: float) -> str:
    rendered = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return rendered or "0"


__all__ = ["LlamaServerManager", "LlamaServerStatus"]
