"""Persistent llama.cpp server management for OCR providers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

from .ocr_models import OCR_PROVIDER_DEEPSEEK_OCR_LLAMA


@dataclass(slots=True)
class LlamaServerStatus:
    """Structured status returned by the server manager."""

    state: str
    message: str
    is_alive: bool
    managed: bool


class LlamaServerManager:
    """Starts, probes, and stops a persistent llama.cpp server."""

    def __init__(
        self,
        *,
        server_url: str = "http://127.0.0.1:8080",
        host: str = "127.0.0.1",
        port: int = 8080,
        model_path: str | Path = "",
        mmproj_path: str | Path = "",
        llama_cpp_dir: str | Path = "",
        gpu_layers: int = 99,
        ctx_size: int = 8192,
        provider_key: str = "",
    ) -> None:
        self.server_url = ""
        self.host = host
        self.port = int(port)
        self.model_path = str(model_path)
        self.mmproj_path = str(mmproj_path)
        self.llama_cpp_dir = str(llama_cpp_dir)
        self.gpu_layers = int(gpu_layers)
        self.ctx_size = int(ctx_size)
        self.provider_key = str(provider_key or "").strip()
        self._process: subprocess.Popen[Any] | None = None
        self.update_config(
            server_url=server_url,
            host=host,
            port=port,
            model_path=model_path,
            mmproj_path=mmproj_path,
            llama_cpp_dir=llama_cpp_dir,
            gpu_layers=gpu_layers,
            ctx_size=ctx_size,
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
        provider_key: str | None = None,
    ) -> None:
        if server_url is not None:
            normalized_url, parsed_host, parsed_port = self._normalize_server_url(server_url)
            self.server_url = normalized_url
            self.host = parsed_host
            self.port = parsed_port
        else:
            if host is not None:
                self.host = str(host)
            if port is not None:
                self.port = int(port)
            self.server_url = f"http://{self.host}:{self.port}"

        if model_path is not None:
            self.model_path = str(model_path)
        if mmproj_path is not None:
            self.mmproj_path = str(mmproj_path)
        if llama_cpp_dir is not None:
            self.llama_cpp_dir = str(llama_cpp_dir)
        if gpu_layers is not None:
            self.gpu_layers = int(gpu_layers)
        if ctx_size is not None:
            self.ctx_size = int(ctx_size)
        if provider_key is not None:
            self.provider_key = str(provider_key or "").strip()

    def resolve_binary_path(self) -> Path:
        search_names = ("llama-server.exe", "llama-server")
        candidate_dirs: list[Path] = []

        if self.llama_cpp_dir:
            base_dir = Path(self.llama_cpp_dir).expanduser()
            candidate_dirs.extend(
                [
                    base_dir,
                    base_dir / "build" / "bin",
                    base_dir / "build" / "bin" / "Release",
                ]
            )

        for candidate_dir in candidate_dirs:
            for executable_name in search_names:
                candidate_path = candidate_dir / executable_name
                if candidate_path.exists():
                    return candidate_path.resolve()

        for executable_name in search_names:
            executable_path = shutil.which(executable_name)
            if executable_path:
                return Path(executable_path).resolve()

        raise FileNotFoundError(
            "Could not find the llama-server binary. Check the llama.cpp directory or PATH."
        )

    def build_command(self, *, disable_prompt_cache: bool | None = None) -> list[str]:
        model_file = self._validate_existing_file(self.model_path, "model.gguf")
        mmproj_file = self._validate_existing_file(self.mmproj_path, "mmproj.gguf")
        binary_path = self.resolve_binary_path()

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
            "0",
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
        return command

    def check_server(self, *, timeout: float = 3.0) -> LlamaServerStatus:
        alive, message = self._probe_server(timeout=timeout)
        if alive:
            return LlamaServerStatus(
                state="Ready",
                message=message,
                is_alive=True,
                managed=self._is_managed_process_running(),
            )

        return LlamaServerStatus(
            state="Stopped",
            message=message,
            is_alive=False,
            managed=False,
        )

    def start_server(
        self,
        *,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
        logger: Callable[[str], None] | None = None,
    ) -> LlamaServerStatus:
        existing_status = self.check_server(timeout=3.0)
        if existing_status.is_alive:
            managed_label = "managed by GUI" if existing_status.managed else "external"
            return LlamaServerStatus(
                state="Ready",
                message=f"llama.cpp server is already running ({managed_label}) at {self.server_url}.",
                is_alive=True,
                managed=existing_status.managed,
            )

        use_cache_safe_flags = self._should_disable_prompt_cache()
        command_variants: list[tuple[list[str], bool]] = [
            (self.build_command(disable_prompt_cache=use_cache_safe_flags), use_cache_safe_flags)
        ]
        if use_cache_safe_flags:
            command_variants.append((self.build_command(disable_prompt_cache=False), False))

        fallback_triggered = False

        for attempt_index, (command, using_cache_safe_flags) in enumerate(command_variants):
            self._terminate_managed_process_if_stale()
            if using_cache_safe_flags:
                self._log(
                    logger,
                    "Starting DeepSeek OCR llama-server with prompt cache disabled/reduced.",
                )
            self._log(logger, f"Starting llama.cpp server: {' '.join(command)}")
            self._process = self._spawn_process(command)

            startup = self._wait_for_startup(timeout=float(timeout), poll_interval=float(poll_interval))
            if startup["ready"]:
                if fallback_triggered:
                    self._log(
                        logger,
                        "llama.cpp did not accept DeepSeek OCR cache-control flags; started without them. "
                        "Slot erase will still be attempted after each page.",
                    )
                return LlamaServerStatus(
                    state="Ready",
                    message=str(startup["message"]),
                    is_alive=True,
                    managed=True,
                )

            if (
                using_cache_safe_flags
                and attempt_index + 1 < len(command_variants)
                and self._is_cache_flag_rejection(str(startup.get("stderr", "") or ""))
            ):
                fallback_triggered = True
                self._log(
                    logger,
                    "llama.cpp did not accept DeepSeek OCR cache-control flags; retrying without them. "
                    "Slot erase will still be attempted after each page.",
                )
                continue

            return LlamaServerStatus(
                state="Error",
                message=str(startup["message"]),
                is_alive=False,
                managed=False,
            )

        return LlamaServerStatus(
            state="Error",
            message=f"Timed out waiting for llama.cpp server to become ready at {self.server_url}.",
            is_alive=False,
            managed=False,
        )

    def stop_server(
        self,
        *,
        timeout: float = 10.0,
        logger: Callable[[str], None] | None = None,
    ) -> LlamaServerStatus:
        if not self._is_managed_process_running():
            external_status = self.check_server(timeout=2.0)
            if external_status.is_alive:
                return LlamaServerStatus(
                    state="Ready",
                    message="A llama.cpp server is running externally; the GUI will not stop it.",
                    is_alive=True,
                    managed=False,
                )
            self._process = None
            return LlamaServerStatus(
                state="Stopped",
                message="llama.cpp server is not running.",
                is_alive=False,
                managed=False,
            )

        assert self._process is not None
        self._log(logger, "Stopping GUI-managed llama.cpp server.")
        self._process.terminate()
        try:
            self._process.wait(timeout=float(timeout))
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2.0)

        self._process = None
        return LlamaServerStatus(
            state="Stopped",
            message="Stopped the GUI-managed llama.cpp server.",
            is_alive=False,
            managed=False,
        )

    def _probe_server(self, *, timeout: float) -> tuple[bool, str]:
        requests = self._requests_module()
        request_error = self._request_exception_class(requests)
        base_url = self.server_url.rstrip("/")
        endpoints = (
            f"{base_url}/health",
            f"{base_url}/v1/models",
        )

        last_error = "Server check failed."
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=float(timeout))
            except request_error as exc:
                last_error = f"Unable to reach {endpoint}: {exc}"
                continue

            if 200 <= response.status_code < 400:
                return True, f"llama.cpp server is ready at {self.server_url}."

            last_error = f"{endpoint} returned HTTP {response.status_code}."

        return False, last_error

    def _normalize_server_url(self, raw_url: str) -> tuple[str, str, int]:
        normalized_input = raw_url.strip() or "http://127.0.0.1:8080"
        if "://" not in normalized_input:
            normalized_input = f"http://{normalized_input}"

        parsed = urlparse(normalized_input)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or 8080)
        scheme = parsed.scheme or "http"
        return f"{scheme}://{host}:{port}", host, port

    def _resolve_working_directory(self) -> Path | None:
        if not self.llama_cpp_dir:
            return None
        path = Path(self.llama_cpp_dir).expanduser()
        return path.resolve() if path.exists() else None

    def _validate_existing_file(self, raw_path: str | Path, label: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not str(candidate).strip():
            raise FileNotFoundError(f"Missing required {label} path.")
        if not candidate.exists():
            raise FileNotFoundError(f"Missing required {label} file: {candidate}")
        return candidate.resolve()

    def _is_managed_process_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _terminate_managed_process_if_stale(self) -> None:
        if self._process is not None and self._process.poll() is not None:
            self._process = None

    def _log(self, logger: Callable[[str], None] | None, message: str) -> None:
        if logger is not None:
            logger(message)

    def _should_disable_prompt_cache(self) -> bool:
        return self.provider_key == OCR_PROVIDER_DEEPSEEK_OCR_LLAMA

    def _spawn_process(self, command: list[str]) -> subprocess.Popen[Any]:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        working_directory = self._resolve_working_directory()
        return subprocess.Popen(
            command,
            cwd=str(working_directory) if working_directory is not None else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
        )

    def _wait_for_startup(self, *, timeout: float, poll_interval: float) -> dict[str, Any]:
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                exit_code = self._process.returncode
                stderr_text = self._read_process_stderr(self._process)
                self._process = None
                excerpt = self._stderr_excerpt(stderr_text)
                message = f"llama.cpp server exited during startup with code {exit_code}."
                if excerpt:
                    message = f"{message} {excerpt}"
                return {
                    "ready": False,
                    "message": message,
                    "stderr": stderr_text,
                }

            alive, message = self._probe_server(timeout=2.0)
            if alive:
                return {
                    "ready": True,
                    "message": message,
                    "stderr": "",
                }

            time.sleep(float(poll_interval))

        self.stop_server(timeout=5.0)
        return {
            "ready": False,
            "message": f"Timed out waiting for llama.cpp server to become ready at {self.server_url}.",
            "stderr": "",
        }

    def _read_process_stderr(self, process: subprocess.Popen[Any]) -> str:
        stderr_stream = getattr(process, "stderr", None)
        if stderr_stream is None:
            return ""
        try:
            output = stderr_stream.read()
        except Exception:
            return ""
        return output if isinstance(output, str) else ""

    def _stderr_excerpt(self, stderr_text: str) -> str:
        normalized = " ".join(str(stderr_text or "").split())
        if not normalized:
            return ""
        return normalized[:400]

    def _is_cache_flag_rejection(self, stderr_text: str) -> bool:
        normalized = str(stderr_text or "").lower()
        if not normalized:
            return False
        mentions_cache_flag = any(
            flag in normalized
            for flag in (
                "--no-cache-prompt",
                "--cache-ram",
                "--no-cache-idle-slots",
            )
        )
        mentions_argument_error = any(
            phrase in normalized
            for phrase in (
                "unknown argument",
                "unknown option",
                "unrecognized option",
                "unrecognized argument",
                "invalid argument",
                "unexpected argument",
            )
        )
        return mentions_cache_flag and mentions_argument_error

    def _requests_module(self) -> Any:
        try:
            import requests
        except Exception as exc:
            raise RuntimeError(
                "The 'requests' package is required to communicate with the llama.cpp server."
            ) from exc
        return requests

    def _request_exception_class(self, requests_module: Any) -> type[Exception]:
        request_error = getattr(requests_module, "RequestException", None)
        if isinstance(request_error, type) and issubclass(request_error, Exception):
            return request_error
        return Exception


__all__ = ["LlamaServerManager", "LlamaServerStatus"]
