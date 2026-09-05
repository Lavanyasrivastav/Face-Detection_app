"""Downloads and caches the MediaPipe FaceLandmarker model bundle.

Modern MediaPipe (>=0.10) ships the face mesh model as a separate
``.task`` file rather than bundling it in the pip package, so it has to
be fetched once and cached locally. This module handles that: check
cache -> download if missing -> verify -> return a local path the
Tasks API can load.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

from .exceptions import ModelInitializationError
from .logging_config import get_logger

logger = get_logger(__name__)

# Official Google-hosted model bundle (face detection + face mesh +
# blendshapes), per the MediaPipe Face Landmarker docs.
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Community mirror, useful when storage.googleapis.com is blocked/unreachable
# from a given network (documented as a known issue in some regions).
FALLBACK_MODEL_URL = (
    "https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/"
    "download/v1/face_landmarker.task"
)

DEFAULT_CACHE_DIR = Path(
    os.environ.get("FACEMESH_MODEL_DIR", Path.home() / ".cache" / "facemesh")
)
MODEL_FILENAME = "face_landmarker.task"

# The bundle is a few MB; a completed download should always be well
# above this floor. Guards against silently caching a truncated/HTML
# error-page response as if it were the real model.
_MIN_VALID_MODEL_BYTES = 1_000_000


def get_model_path(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_url: str = DEFAULT_MODEL_URL,
    force_download: bool = False,
) -> Path:
    """Return a local path to the face_landmarker.task model, downloading
    it into ``cache_dir`` on first use.

    Args:
        cache_dir: Directory to cache the model in. Override with the
            FACEMESH_MODEL_DIR environment variable, or pass explicitly
            (e.g. to bundle the model inside a Docker image at build time).
        model_url: URL to download from if not cached.
        force_download: Re-download even if a cached copy exists.

    Returns:
        Path to a valid, on-disk .task model file.

    Raises:
        ModelInitializationError: If the model can't be found locally and
            can't be downloaded (e.g. no network, both primary and
            fallback URLs fail).
    """
    cache_dir = Path(cache_dir)
    model_path = cache_dir / MODEL_FILENAME

    if not force_download and _is_valid_model_file(model_path):
        logger.debug("Using cached model at %s", model_path)
        return model_path

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Try the requested URL first, then the known-good fallback mirror
    # (deduplicated, in case the caller already passed the fallback).
    urls_to_try = [model_url]
    if FALLBACK_MODEL_URL not in urls_to_try:
        urls_to_try.append(FALLBACK_MODEL_URL)

    for url in urls_to_try:
        try:
            _download(url, model_path)
            if _is_valid_model_file(model_path):
                return model_path
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Download from %s failed: %s", url, exc)
            continue

    raise ModelInitializationError(
        "Could not obtain the FaceLandmarker model bundle. Tried:\n"
        f"  - {model_url}\n"
        f"  - {FALLBACK_MODEL_URL}\n"
        "Check your network connection, or manually download the .task "
        f"file and place it at: {model_path}"
    )


def _is_valid_model_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= _MIN_VALID_MODEL_BYTES


def _download(url: str, dest: Path) -> None:
    logger.info("Downloading FaceLandmarker model from %s", url)
    tmp_path = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
            if total and downloaded < total:
                raise OSError(
                    f"Incomplete download: got {downloaded} of {total} bytes"
                )
        tmp_path.replace(dest)
        logger.info("Model cached at %s (%d bytes)", dest, dest.stat().st_size)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
