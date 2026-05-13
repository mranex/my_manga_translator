from __future__ import annotations

import base64
import io
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8088
DEFAULT_TIMEOUT = 120
DEFAULT_CTX = 4096
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PROMPT = (
    "Recognize all text in this image crop. "
    "Return only the exact text, no explanation."
)


class PaddleOCRVLError(RuntimeError):
    pass


def clean_paddleocr_vl_output(text: str) -> str:
    if text is None:
        return ""

    cleaned = str(text).strip()
    if not cleaned:
        return ""

    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        for key in ("text", "ocr", "content", "result"):
            value = parsed.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                break
    elif isinstance(parsed, list):
        text_parts = [value for value in parsed if isinstance(value, str)]
        if text_parts:
            cleaned = "\n".join(text_parts).strip()

    prefix_patterns = (
        "OCR:",
        "Text:",
        "Recognized text:",
        "Recognized Text:",
        "Output:",
    )
    for prefix in prefix_patterns:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
            break

    lines = [line.strip() for line in cleaned.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [" ".join(line.split()) for line in lines if line.strip()]
    return "\n".join(lines).strip()


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "model" / "paddleocr_vl"


def _resolve_model_paths(
    model_path: str | None = None,
    mmproj_path: str | None = None,
) -> tuple[Path, Path]:
    resolved_model = Path(
        model_path
        or os.environ.get("PADDLEOCR_VL_MODEL_PATH", "")
        or (_default_model_dir() / "model.gguf")
    )
    resolved_mmproj = Path(
        mmproj_path
        or os.environ.get("PADDLEOCR_VL_MMPROJ_PATH", "")
        or (_default_model_dir() / "mmproj.gguf")
    )

    if resolved_model.exists() and resolved_mmproj.exists():
        return resolved_model, resolved_mmproj

    raise PaddleOCRVLError(
        "PaddleOCR-VL GGUF model/mmproj not found. "
        "Set PADDLEOCR_VL_MODEL_PATH and PADDLEOCR_VL_MMPROJ_PATH or place files under "
        "model/paddleocr_vl/."
    )


def _candidate_llama_dirs(llama_cpp_dir: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    for raw in (
        llama_cpp_dir,
        os.environ.get("LLAMA_CPP_DIR"),
        "tools/llama.cpp",
        "vendor/llama.cpp",
        "llama.cpp",
    ):
        if not raw:
            continue
        candidates.append(Path(raw))
    return candidates


def _find_binary(name: str, directories: Sequence[Path]) -> str | None:
    for directory in directories:
        candidate = directory / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def _find_llama_binaries(llama_cpp_dir: str | None = None) -> dict[str, str | None]:
    directories = _candidate_llama_dirs(llama_cpp_dir)
    is_windows = os.name == "nt"

    server_names = ["llama-server.exe", "llama-server"] if is_windows else ["llama-server"]
    mtmd_names = ["llama-mtmd-cli.exe", "llama-mtmd-cli"] if is_windows else ["llama-mtmd-cli"]
    cli_names = ["llama-cli.exe", "llama-cli"] if is_windows else ["llama-cli"]

    return {
        "server": next((path for name in server_names if (path := _find_binary(name, directories)) is not None), None),
        "mtmd_cli": next((path for name in mtmd_names if (path := _find_binary(name, directories)) is not None), None),
        "cli": next((path for name in cli_names if (path := _find_binary(name, directories)) is not None), None),
    }


def _build_llama_server_command(
    binary_path: str,
    *,
    model_path: Path,
    mmproj_path: Path,
    host: str,
    port: int,
    num_ctx: int,
    gpu_layers: int | None,
) -> list[str]:
    template = os.environ.get("PADDLEOCR_VL_LLAMA_SERVER_CMD")
    if template:
        formatted = template.format(
            model=str(model_path),
            mmproj=str(mmproj_path),
            host=host,
            port=port,
            ctx=num_ctx,
            ngl=0 if gpu_layers is None else gpu_layers,
        )
        return shlex.split(formatted, posix=(os.name != "nt"))

    command = [
        str(binary_path),
        "-m",
        str(model_path),
        "--mmproj",
        str(mmproj_path),
        "--host",
        host,
        "--port",
        str(int(port)),
        "-c",
        str(int(num_ctx)),
    ]
    if gpu_layers is not None:
        command.extend(["-ngl", str(int(gpu_layers))])
    return command


def _chat_completion_content(response_data: dict[str, Any]) -> str:
    choices = response_data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise PaddleOCRVLError("Invalid llama.cpp response: missing choices")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise PaddleOCRVLError("Invalid llama.cpp response: missing message")

    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        content = "\n".join(text_parts)
    if not isinstance(content, str):
        raise PaddleOCRVLError("Invalid llama.cpp response: content is not text")
    return clean_paddleocr_vl_output(content)


class PaddleOCRVLOCR:
    def __init__(
        self,
        server_url: str | None = None,
        llama_cpp_dir: str | None = None,
        model_path: str | None = None,
        mmproj_path: str | None = None,
        auto_start_server: bool = True,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: int = DEFAULT_TIMEOUT,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        num_ctx: int = DEFAULT_CTX,
        gpu_layers: int | None = None,
    ):
        self.server_url = (
            (server_url or os.environ.get("PADDLEOCR_VL_SERVER_URL") or f"http://{host}:{port}")
            .rstrip("/")
        )
        self.llama_cpp_dir = llama_cpp_dir
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.auto_start_server = bool(auto_start_server)
        self.host = host
        self.port = int(port)
        self.timeout = int(timeout)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.num_ctx = int(num_ctx)
        self.gpu_layers = gpu_layers
        self._server_process: subprocess.Popen[str] | None = None
        self._server_started_here = False
        self._server_command: list[str] | None = None
        self._cli_command: list[str] | None = None

    def __call__(self, image) -> str:
        return self.recognize(image)

    def _requests(self):
        try:
            import requests
        except Exception as exc:
            raise PaddleOCRVLError(
                "requests is required for PaddleOCR-VL llama.cpp HTTP client. "
                "Add requests to the environment."
            ) from exc
        return requests

    def _image_to_base64_png(self, image) -> str:
        pil_image = self._normalize_image(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _normalize_image(self, image) -> Image.Image:
        if Image is None:
            raise PaddleOCRVLError(
                "Pillow is required for PaddleOCR-VL OCR image handling."
            )
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if np is not None and isinstance(image, np.ndarray):
            array = image
            if array.ndim == 2:
                return Image.fromarray(array).convert("RGB")
            if array.ndim == 3 and array.shape[2] >= 3:
                return Image.fromarray(array[..., :3][:, :, ::-1]).convert("RGB")
            return Image.fromarray(array).convert("RGB")

        raise PaddleOCRVLError(
            "PaddleOCR-VL expects a PIL.Image or numpy array crop."
        )

    def is_server_alive(self) -> bool:
        requests = self._requests()
        endpoints = [
            (f"{self.server_url}/health", "get"),
            (f"{self.server_url}/v1/models", "get"),
        ]
        for url, method in endpoints:
            try:
                response = getattr(requests, method)(url, timeout=3)
            except Exception:
                continue
            if response.status_code < 500:
                return True
        return False

    def _resolve_model_paths(self) -> tuple[Path, Path]:
        return _resolve_model_paths(self.model_path, self.mmproj_path)

    def _build_server_command(self) -> list[str]:
        binaries = _find_llama_binaries(self.llama_cpp_dir)
        server_binary = binaries["server"]
        if not server_binary:
            raise PaddleOCRVLError(
                "llama-server binary not found. Set LLAMA_CPP_DIR or add llama-server to PATH."
            )
        model_path, mmproj_path = self._resolve_model_paths()
        return _build_llama_server_command(
            server_binary,
            model_path=model_path,
            mmproj_path=mmproj_path,
            host=self.host,
            port=self.port,
            num_ctx=self.num_ctx,
            gpu_layers=self.gpu_layers,
        )

    def _build_cli_command(self, image_path: str) -> list[str]:
        binaries = _find_llama_binaries(self.llama_cpp_dir)
        cli_binary = binaries["mtmd_cli"] or binaries["cli"]
        if not cli_binary:
            raise PaddleOCRVLError(
                "No llama.cpp OCR binary found. Set LLAMA_CPP_DIR or add llama-mtmd-cli / llama-cli to PATH."
            )
        model_path, mmproj_path = self._resolve_model_paths()
        return [
            str(cli_binary),
            "-m",
            str(model_path),
            "--mmproj",
            str(mmproj_path),
            "--image",
            str(image_path),
            "-p",
            DEFAULT_PROMPT,
            "-c",
            str(self.num_ctx),
        ] + (["-ngl", str(int(self.gpu_layers))] if self.gpu_layers is not None else [])

    def _has_cli_fallback(self) -> bool:
        binaries = _find_llama_binaries(self.llama_cpp_dir)
        return bool(binaries["mtmd_cli"] or binaries["cli"])

    def ensure_llama_server_running(self) -> None:
        if self.is_server_alive():
            return

        if not self.auto_start_server:
            raise PaddleOCRVLError(
                "PaddleOCR-VL server is not reachable. Set PADDLEOCR_VL_SERVER_URL to a running llama.cpp server "
                "or enable auto_start_server with LLAMA_CPP_DIR / model paths configured."
            )

        self._server_command = self._build_server_command()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self._server_process = subprocess.Popen(
            self._server_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        self._server_started_here = True
        print(f"Using PaddleOCR-VL via llama.cpp")
        print(f"PaddleOCR-VL server: {self.server_url}")
        print(f"Auto-started llama-server on {self.host}:{self.port}")

        deadline = time.time() + 60.0
        while time.time() < deadline:
            if self.is_server_alive():
                return
            if self._server_process.poll() is not None:
                output = ""
                try:
                    output = self._server_process.communicate(timeout=1)[0]
                except Exception:
                    pass
                excerpt = output.strip()[:600]
                raise PaddleOCRVLError(
                    "Failed to start PaddleOCR-VL llama.cpp server. "
                    f"Command: {' '.join(self._server_command)}\n"
                    f"Output: {excerpt or '(no output)'}"
                )
            time.sleep(1.0)

        raise PaddleOCRVLError(
            "Timed out waiting for PaddleOCR-VL llama.cpp server to become ready."
        )

    def _chat_completion(self, image) -> str:
        self.ensure_llama_server_running()
        requests = self._requests()
        image_b64 = self._image_to_base64_png(image)
        payload = {
            "model": "paddleocr-vl",
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DEFAULT_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        }

        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return self._legacy_completion(image_b64)
        if response.status_code >= 400:
            raise PaddleOCRVLError(
                "PaddleOCR-VL server request failed: "
                f"{response.status_code} {response.text[:400]}"
            )
        return _chat_completion_content(response.json())

    def _legacy_completion(self, image_b64: str) -> str:
        requests = self._requests()
        payload = {
            "prompt": DEFAULT_PROMPT,
            "image_data": image_b64,
            "n_predict": self.max_new_tokens,
            "temperature": self.temperature,
        }
        response = requests.post(
            f"{self.server_url}/completion",
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise PaddleOCRVLError(
                "PaddleOCR-VL server did not expose a compatible endpoint. "
                f"HTTP {response.status_code}: {response.text[:400]}"
            )
        response_json = response.json()
        content = response_json.get("content") or response_json.get("text") or ""
        if not isinstance(content, str):
            raise PaddleOCRVLError("Invalid legacy llama.cpp OCR response payload")
        return clean_paddleocr_vl_output(content)

    def _recognize_via_cli(self, image) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            pil_image = self._normalize_image(image)
            pil_image.save(temp_path, format="PNG")
            self._cli_command = self._build_cli_command(str(temp_path))
            completed = subprocess.run(
                self._cli_command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise PaddleOCRVLError(
                    "PaddleOCR-VL CLI OCR failed: "
                    f"{completed.stderr[:400] or completed.stdout[:400]}"
                )
            return clean_paddleocr_vl_output(completed.stdout)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def recognize(self, image) -> str:
        pil_image = self._normalize_image(image)
        width, height = pil_image.size
        if width <= 1 or height <= 1:
            return ""

        try:
            return self._chat_completion(pil_image)
        except PaddleOCRVLError:
            if self.auto_start_server and self._has_cli_fallback():
                return self._recognize_via_cli(pil_image)
            raise
        except Exception:
            if not self.auto_start_server:
                raise
            return self._recognize_via_cli(pil_image)

    def process_batch(self, images, max_workers: int = 1) -> list[str]:
        if images is None:
            return []

        env_workers = os.environ.get("PADDLEOCR_VL_MAX_WORKERS")
        if env_workers:
            try:
                max_workers = max(1, int(env_workers))
            except ValueError:
                max_workers = 1

        print(f"OCR processing {len(images)} text items with PaddleOCR-VL...")
        print(f"Using PaddleOCR-VL via llama.cpp")
        print(f"PaddleOCR-VL server: {self.server_url}")

        results: list[str] = []
        for index, image in enumerate(images, start=1):
            try:
                text = self.recognize(image)
            except Exception as exc:
                print(f"  [Text Item {index}] ERROR: {exc}")
                results.append("")
                continue

            if text:
                preview = text[:50].replace("\n", " ") + ("..." if len(text) > 50 else "")
                print(f"  [Text Item {index}] OK: {preview}")
            else:
                print(f"  [Text Item {index}] EMPTY (no text detected)")
            results.append(text)

        return results

    def close(self) -> None:
        if self._server_process is not None and self._server_started_here:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except Exception:
                self._server_process.kill()
            finally:
                self._server_process = None
                self._server_started_here = False


__all__ = [
    "PaddleOCRVLError",
    "PaddleOCRVLOCR",
    "clean_paddleocr_vl_output",
]
