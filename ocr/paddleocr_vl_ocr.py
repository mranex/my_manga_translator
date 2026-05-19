from __future__ import annotations

import base64
import io
import json
import os
import re
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
DEFAULT_TASK_PREFIX = "OCR:"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_REPEAT_LAST_N = -1
DEFAULT_MAX_RETRIES = 3
DEFAULT_MIN_SHORT_SIDE = 10
DEFAULT_MAX_LONG_SIDE = 1000
DEFAULT_IMAGE_PAD = 12
DEFAULT_MAX_UPSCALE_FACTOR = 4.0
OCR_REPEAT_MAX_UNIT_CHARS = 12
OCR_REPEAT_MIN_REPETITIONS = 4
OCR_REPEAT_MIN_TOTAL_CHARS = 12
OCR_LIST_MARKER_CHARS = r"\-\–\—\*\•\・"
OCR_CIRCLED_NUMBER_CHARS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
LINE_LIST_MARKER_RE = re.compile(
    rf"^\s*(?:(?:\((?:\d{{1,2}})\)|(?:\d{{1,2}}(?:[.):]|\s+)|[{OCR_CIRCLED_NUMBER_CHARS}]))|[{OCR_LIST_MARKER_CHARS}])\s*"
)
LINE_NUMBERED_MARKER_RE = re.compile(
    rf"^\s*(?:\((?:\d{{1,2}})\)|(?:\d{{1,2}}(?:[.):]|\s+)|[{OCR_CIRCLED_NUMBER_CHARS}]))\s*"
)
INLINE_NUMBERED_MARKER_RE = re.compile(
    rf"(?:(?<=^)|(?<=\s))(?:\((?:\d{{1,2}})\)|(?:\d{{1,2}}[.):]|[{OCR_CIRCLED_NUMBER_CHARS}]))(?=\s*\S)"
)
INLINE_BULLET_MARKER_RE = re.compile(
    rf"(?:(?<=^)|(?<=\s))(?:[{OCR_LIST_MARKER_CHARS}])(?=\s*\S)"
)


class PaddleOCRVLError(RuntimeError):
    pass


def normalize_ocr_language(language: str | None) -> str | None:
    if language is None:
        return None

    cleaned = str(language).strip()
    if not cleaned:
        return None

    normalized = cleaned.lower()
    if normalized in {"auto", "unknown", "none", "null"}:
        return None

    alias_map = {
        "en": "English",
        "eng": "English",
        "english": "English",
        "ja": "Japanese",
        "jpn": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "kor": "Korean",
        "korean": "Korean",
        "zh": "Chinese",
        "chi": "Chinese",
        "zho": "Chinese",
        "chinese": "Chinese",
        "simplified chinese": "Chinese",
        "traditional chinese": "Chinese",
        "vi": "Vietnamese",
        "vie": "Vietnamese",
        "vietnamese": "Vietnamese",
        "fr": "French",
        "french": "French",
        "de": "German",
        "german": "German",
        "es": "Spanish",
        "spanish": "Spanish",
        "th": "Thai",
        "thai": "Thai",
        "ru": "Russian",
        "russian": "Russian",
    }
    if normalized in alias_map:
        return alias_map[normalized]

    tokens = re.split(r"[\s_-]+", cleaned)
    return " ".join(token[:1].upper() + token[1:] for token in tokens if token)


def build_paddleocr_vl_prompt(source_language: str | None = None) -> str:
    language = normalize_ocr_language(source_language)
    if language:
        return f"{DEFAULT_TASK_PREFIX}"
    return f"{DEFAULT_TASK_PREFIX}"


def _looks_like_multiline_list(lines: list[str]) -> bool:
    non_empty_lines = [line for line in lines if line.strip()]
    if len(non_empty_lines) < 2:
        return False

    marker_count = sum(1 for line in non_empty_lines if LINE_LIST_MARKER_RE.match(line))
    return marker_count >= 2 and marker_count >= max(2, len(non_empty_lines) - 1)


def _strip_line_marker(line: str) -> str:
    return LINE_LIST_MARKER_RE.sub("", line, count=1).strip()


def _strip_inline_numbered_markers(text: str) -> str:
    cleaned = str(text or "")

    numbered_markers = list(INLINE_NUMBERED_MARKER_RE.finditer(cleaned))
    if len(numbered_markers) >= 1:
        cleaned = INLINE_NUMBERED_MARKER_RE.sub("", cleaned)

    bullet_markers = list(INLINE_BULLET_MARKER_RE.finditer(cleaned))
    if len(bullet_markers) >= 1:
        cleaned = INLINE_BULLET_MARKER_RE.sub("", cleaned)

    return cleaned


def _join_ocr_lines(lines: list[str]) -> str:
    joined = " ".join(line.strip() for line in lines if line and line.strip())
    joined = re.sub(r"\s+", " ", joined)
    joined = re.sub(r"\s+([,.;:!?])", r"\1", joined)
    return joined.strip()


def strip_ocr_list_markers(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    lines = cleaned.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if _looks_like_multiline_list(lines):
        stripped_lines = [_strip_line_marker(line) for line in lines]
        return _join_ocr_lines(stripped_lines)

    inline_cleaned = _strip_inline_numbered_markers(cleaned)
    if inline_cleaned != cleaned:
        return _join_ocr_lines(inline_cleaned.replace("\r\n", "\n").replace("\r", "\n").split("\n"))

    return cleaned


def clean_paddleocr_vl_output(text: str) -> str:
    if text is None:
        return ""

    cleaned = str(text).strip()
    if not cleaned:
        return ""

    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
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

    cleaned = strip_ocr_list_markers(cleaned)

    lines = [line.strip() for line in cleaned.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [" ".join(line.split()) for line in lines if line.strip()]
    return "\n".join(lines).strip()


def repeated_ocr_suffix_start(text: str) -> int | None:
    cleaned = str(text or "")
    non_space_chars = []
    non_space_indices = []
    for index, char in enumerate(cleaned):
        if char.isspace():
            continue
        non_space_chars.append(char)
        non_space_indices.append(index)

    condensed = "".join(non_space_chars)
    if len(condensed) < OCR_REPEAT_MIN_TOTAL_CHARS:
        return None

    best_start = None
    max_unit = min(OCR_REPEAT_MAX_UNIT_CHARS, len(condensed) // OCR_REPEAT_MIN_REPETITIONS)
    for unit_length in range(1, max_unit + 1):
        unit = condensed[-unit_length:]
        repetitions = 1
        start_index = len(condensed) - unit_length

        while start_index - unit_length >= 0:
            candidate = condensed[start_index - unit_length:start_index]
            if candidate != unit:
                break
            repetitions += 1
            start_index -= unit_length

        total_chars = repetitions * unit_length
        if repetitions < OCR_REPEAT_MIN_REPETITIONS or total_chars < OCR_REPEAT_MIN_TOTAL_CHARS:
            continue

        original_start = non_space_indices[start_index]
        if best_start is None or original_start < best_start:
            best_start = original_start

    return best_start


def trim_repeated_ocr_output(text: str) -> tuple[str, bool]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "", False

    repeat_start = repeated_ocr_suffix_start(cleaned)
    if repeat_start is None:
        return cleaned, False

    trimmed = cleaned[:repeat_start].strip()
    return trimmed, True


def is_degenerate_ocr_output(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False

    compact = "".join(char for char in cleaned if not char.isspace())
    if len(compact) < OCR_REPEAT_MIN_TOTAL_CHARS:
        return False

    unique_chars = set(compact)
    if len(unique_chars) <= 2:
        return True

    trimmed, had_repeat = trim_repeated_ocr_output(cleaned)
    if had_repeat and not trimmed:
        return True

    if had_repeat and len(compact) >= 18:
        return True

    diversity = len(unique_chars) / max(1, len(compact))
    return len(compact) >= 24 and diversity < 0.18


def _candidate_score(text: str) -> tuple[int, int]:
    compact = "".join(char for char in str(text or "") if not char.isspace())
    return (len(compact), len(set(compact)))


def _prefer_candidate(current: str, candidate: str) -> str:
    if not candidate:
        return current
    if not current:
        return candidate
    return candidate if _candidate_score(candidate) > _candidate_score(current) else current


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
        "--temp",
        "0",
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
        min_short_side: int | None = None,
        max_long_side: int | None = None,
        image_pad: int | None = None,
        max_upscale_factor: float | None = None,
        prompt: str | None = None,
        source_language: str | None = None,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        repeat_last_n: int = DEFAULT_REPEAT_LAST_N,
        max_retries: int = DEFAULT_MAX_RETRIES,
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
        self.repetition_penalty = float(repetition_penalty)
        self.repeat_last_n = int(repeat_last_n)
        self.max_retries = max(1, int(max_retries))
        self._server_process: subprocess.Popen[str] | None = None
        self._server_started_here = False
        self._server_command: list[str] | None = None
        self._cli_command: list[str] | None = None

        self.configured_source_language = source_language
        self.source_language = normalize_ocr_language(
            source_language if source_language is not None else os.environ.get("PADDLEOCR_VL_SOURCE_LANGUAGE")
        )
        self.prompt = (
            prompt
            or os.environ.get("PADDLEOCR_VL_PROMPT")
            or build_paddleocr_vl_prompt(self.source_language)
        )

        raw_min_short_side = (
            min_short_side
            if min_short_side is not None
            else os.environ.get("PADDLEOCR_VL_MIN_SHORT_SIDE", DEFAULT_MIN_SHORT_SIDE)
        )
        self.min_short_side = int(raw_min_short_side)

        raw_max_long_side = (
            max_long_side
            if max_long_side is not None
            else os.environ.get("PADDLEOCR_VL_MAX_LONG_SIDE", DEFAULT_MAX_LONG_SIDE)
        )
        self.max_long_side = int(raw_max_long_side)

        raw_image_pad = (
            image_pad
            if image_pad is not None
            else os.environ.get("PADDLEOCR_VL_PAD", DEFAULT_IMAGE_PAD)
        )
        self.image_pad = int(raw_image_pad)

        raw_max_upscale_factor = (
            max_upscale_factor
            if max_upscale_factor is not None
            else os.environ.get("PADDLEOCR_VL_MAX_UPSCALE_FACTOR", DEFAULT_MAX_UPSCALE_FACTOR)
        )
        self.max_upscale_factor = float(raw_max_upscale_factor)
    
    def _preprocess_ocr_image(self, pil_image: Image.Image) -> Image.Image:
        image = pil_image.convert("RGB")

        if self.image_pad > 0:
            padded = Image.new(
                "RGB",
                (image.width + self.image_pad * 2, image.height + self.image_pad * 2),
                "white",
            )
            padded.paste(image, (self.image_pad, self.image_pad))
            image = padded

        width, height = image.size
        if width <= 0 or height <= 0:
            return image

        short_side = min(width, height)
        long_side = max(width, height)

        scale = 1.0

        if self.min_short_side > 0 and short_side < self.min_short_side:
            desired_scale = self.min_short_side / short_side
            scale = max(scale, min(desired_scale, self.max_upscale_factor))

        if self.max_long_side > 0 and long_side * scale > self.max_long_side:
            scale = min(scale, self.max_long_side / long_side)

        if abs(scale - 1.0) > 0.01 and (scale < 1.0 or scale > 1.01):
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image
    
    
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
        pil_image = self._preprocess_ocr_image(pil_image)
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
            # Numpy crops are assumed to come from the app's OpenCV pipeline,
            # so 3-channel arrays are treated as BGR and 4-channel arrays as BGRA.
            array = image
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            if array.ndim == 2:
                return Image.fromarray(array).convert("RGB")
            if array.ndim == 3 and array.shape[2] == 3:
                return Image.fromarray(array[:, :, ::-1]).convert("RGB")
            if array.ndim == 3 and array.shape[2] == 4:
                return Image.fromarray(array[:, :, [2, 1, 0, 3]]).convert("RGB")
            raise PaddleOCRVLError(
                f"PaddleOCR-VL received unsupported numpy crop shape: {array.shape}"
            )

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
            self.prompt,
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
                "Local OCR server is not reachable. Start run_server.bat and check health first."
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

    def _chat_completion(self, image, *, max_new_tokens: int | None = None) -> str:
        self.ensure_llama_server_running()
        requests = self._requests()
        image_b64 = self._image_to_base64_png(image)
        token_limit = int(max_new_tokens if max_new_tokens is not None else self.max_new_tokens)
        payload = {
            "model": "paddleocr-vl",
            "temperature": self.temperature,
            "max_tokens": token_limit,
            "repeat_penalty": self.repetition_penalty,
            "repeat_last_n": self.repeat_last_n,
            "repetition_penalty": self.repetition_penalty,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": self.prompt},
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
            return self._legacy_completion(image_b64, max_new_tokens=token_limit)
        if response.status_code >= 400:
            raise PaddleOCRVLError(
                "PaddleOCR-VL server request failed: "
                f"{response.status_code} {response.text[:400]}"
            )
        return _chat_completion_content(response.json())

    def _legacy_completion(self, image_b64: str, *, max_new_tokens: int | None = None) -> str:
        requests = self._requests()
        token_limit = int(max_new_tokens if max_new_tokens is not None else self.max_new_tokens)
        payload = {
            "prompt": self.prompt,
            "image_data": image_b64,
            "n_predict": token_limit,
            "temperature": self.temperature,
            "repeat_penalty": self.repetition_penalty,
            "repeat_last_n": self.repeat_last_n,
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

    def _recognize_via_cli(self, image, *, max_new_tokens: int | None = None) -> str:
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

    def _recognize_once(self, pil_image: Image.Image, *, max_new_tokens: int | None = None) -> str:
        try:
            return self._chat_completion(pil_image, max_new_tokens=max_new_tokens)
        except PaddleOCRVLError:
            if self.auto_start_server and self._has_cli_fallback():
                return self._recognize_via_cli(pil_image, max_new_tokens=max_new_tokens)
            raise
        except Exception:
            if not self.auto_start_server:
                raise
            return self._recognize_via_cli(pil_image, max_new_tokens=max_new_tokens)

    def _recognize_with_retries(self, image, *, logger=None, item_label: str | None = None) -> tuple[str, str]:
        pil_image = self._normalize_image(image)
        width, height = pil_image.size
        if width <= 1 or height <= 1:
            return "", "EMPTY"

        best_trimmed_candidate = ""
        best_fallback_candidate = ""

        for attempt in range(1, self.max_retries + 1):
            request_max_tokens = (
                self.max_new_tokens
                if attempt == 1
                else min(self.max_new_tokens, 128)
            )
            raw_text = self._recognize_once(pil_image, max_new_tokens=request_max_tokens)
            cleaned = clean_paddleocr_vl_output(raw_text)
            trimmed, had_repeat = trim_repeated_ocr_output(cleaned)
            degenerate = is_degenerate_ocr_output(cleaned)

            if cleaned:
                best_fallback_candidate = _prefer_candidate(best_fallback_candidate, cleaned)
            if had_repeat and trimmed:
                best_trimmed_candidate = _prefer_candidate(best_trimmed_candidate, trimmed)

            if cleaned and not had_repeat and not degenerate:
                return cleaned, "OK"

            if logger and item_label and attempt < self.max_retries:
                if not cleaned:
                    logger(f"  [{item_label}] RETRY {attempt}/{self.max_retries}: empty OCR output")
                elif had_repeat:
                    logger(f"  [{item_label}] RETRY {attempt}/{self.max_retries}: repeated OCR output")
                elif degenerate:
                    logger(f"  [{item_label}] RETRY {attempt}/{self.max_retries}: degenerate OCR output")

        if best_trimmed_candidate:
            return best_trimmed_candidate, "LOOP_TRIMMED"
        if best_fallback_candidate:
            return best_fallback_candidate, "BEST_EFFORT"
        return "", "EMPTY"

    def recognize(self, image) -> str:
        text, _status = self._recognize_with_retries(image)
        return text

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
                text, status = self._recognize_with_retries(
                    image,
                    logger=print,
                    item_label=f"Text Item {index}",
                )
            except Exception as exc:
                print(f"  [Text Item {index}] ERROR: {exc}")
                results.append("")
                continue

            if text and status == "OK":
                preview = text[:50].replace("\n", " ") + ("..." if len(text) > 50 else "")
                print(f"  [Text Item {index}] OK: {preview}")
            elif text and status == "LOOP_TRIMMED":
                preview = text[:50].replace("\n", " ") + ("..." if len(text) > 50 else "")
                print(f"  [Text Item {index}] LOOP_TRIMMED: {preview}")
            elif text:
                preview = text[:50].replace("\n", " ") + ("..." if len(text) > 50 else "")
                print(f"  [Text Item {index}] BEST_EFFORT: {preview}")
            else:
                print(f"  [Text Item {index}] EMPTY (no text detected)")
            results.append(text)

        return results

    def reset_session(self) -> None:
        """
        Called between pages to allow OCR backends to clear transient state.

        For llama.cpp server mode this is intentionally a no-op unless a future
        endpoint or opt-in restart behavior is configured.
        """
        if os.environ.get("PADDLEOCR_VL_RESTART_SERVER_PER_PAGE", "0") == "1":
            self.close()

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
    "DEFAULT_MAX_NEW_TOKENS",
    "build_paddleocr_vl_prompt",
    "clean_paddleocr_vl_output",
    "is_degenerate_ocr_output",
    "normalize_ocr_language",
    "repeated_ocr_suffix_start",
    "strip_ocr_list_markers",
    "trim_repeated_ocr_output",
]
