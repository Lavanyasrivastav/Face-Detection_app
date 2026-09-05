"""Core face mesh detection logic.

Built on MediaPipe's Tasks API (``mediapipe.tasks.python.vision.FaceLandmarker``),
which is the current, actively maintained interface -- the older
``mediapipe.solutions.face_mesh`` API used in earlier MediaPipe releases has
been removed from recent MediaPipe packages. This module isolates that
dependency behind a small, typed API so the rest of the application (CLI,
image pipeline, video pipeline) never touches MediaPipe/OpenCV objects
directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .exceptions import ModelInitializationError
from .logging_config import get_logger
from .model_manager import get_model_path

logger = get_logger(__name__)

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision as mp_vision
except ImportError as exc:  # pragma: no cover - exercised only when deps missing
    raise ImportError(
        "Missing required dependencies. Install them with:\n"
        "    pip install -r requirements.txt"
    ) from exc


Point3D = Tuple[float, float, float]

# Drawing colors (BGR, OpenCV convention).
_COLOR_TESSELATION = (128, 128, 128)  # light gray mesh
_COLOR_CONTOURS = (0, 200, 0)  # green face outline/features
_COLOR_IRISES = (0, 165, 255)  # orange irises


class RunningMode(str, Enum):
    """Mirrors MediaPipe's VisionTaskRunningMode, exposed as plain strings
    so callers don't need to import MediaPipe enums directly."""

    IMAGE = "image"
    VIDEO = "video"


@dataclass
class DetectionResult:
    """Result of running face mesh detection on a single frame.

    Attributes:
        landmarks: One list of (x, y, z) tuples per detected face.
            x, y are normalized to [0, 1] relative to image width/height;
            z is relative depth (smaller = closer to camera), roughly on
            the same scale as x.
        image_shape: (height, width, channels) of the source image.
    """

    landmarks: List[List[Point3D]] = field(default_factory=list)
    image_shape: Tuple[int, int, int] = (0, 0, 0)

    @property
    def face_count(self) -> int:
        return len(self.landmarks)

    @property
    def has_detections(self) -> bool:
        return self.face_count > 0

    def landmarks_as_pixels(self, face_index: int = 0) -> List[Tuple[int, int]]:
        """Convert normalized landmarks for one face into pixel coordinates."""
        if face_index >= self.face_count:
            raise IndexError(
                f"face_index {face_index} out of range (detected {self.face_count} faces)"
            )
        h, w = self.image_shape[:2]
        return [(int(x * w), int(y * h)) for x, y, _ in self.landmarks[face_index]]


class FaceMeshDetector:
    """Detects 3D facial landmarks in images or video frames.

    Wraps ``mediapipe.tasks.python.vision.FaceLandmarker``. The first time
    a detector is created, the ~4MB model bundle is downloaded and cached
    under ``~/.cache/facemesh`` (override with the ``FACEMESH_MODEL_DIR``
    env var, or pass ``model_path`` explicitly).

    Thread-safety note: a FaceLandmarker instance is not thread-safe.
    Create one ``FaceMeshDetector`` per thread/process for concurrent use.

    Example:
        >>> with FaceMeshDetector(running_mode=RunningMode.IMAGE) as detector:
        ...     result, annotated = detector.process_and_draw(image_bgr)
    """

    def __init__(
        self,
        running_mode: RunningMode = RunningMode.IMAGE,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Args:
            running_mode: IMAGE for independent images (photo library,
                batch job); VIDEO for a sequence of frames from the same
                source (webcam, video file) where MediaPipe can use
                temporal tracking for speed and stability.
            max_num_faces: Maximum number of faces to detect per frame.
            min_detection_confidence: [0, 1] threshold for the initial
                face detection step.
            min_presence_confidence: [0, 1] threshold for face presence
                in the landmark stage.
            min_tracking_confidence: [0, 1] threshold for landmark
                tracking between frames (VIDEO mode only).
            model_path: Explicit path to a face_landmarker.task file.
                If omitted, the model is downloaded/cached automatically.

        Raises:
            ModelInitializationError: If the model can't be obtained, or
                the MediaPipe graph fails to initialize.
        """
        self.running_mode = running_mode
        self.max_num_faces = max_num_faces
        self._closed = False
        self._frame_timestamp_ms = 0  # monotonic clock for VIDEO mode

        resolved_model_path = Path(model_path) if model_path else get_model_path()
        if not resolved_model_path.is_file():
            raise ModelInitializationError(
                f"Model file not found at {resolved_model_path}"
            )

        mp_running_mode = (
            mp_vision.RunningMode.VIDEO
            if running_mode == RunningMode.VIDEO
            else mp_vision.RunningMode.IMAGE
        )

        try:
            options = mp_vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(resolved_model_path)),
                running_mode=mp_running_mode,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
            raise ModelInitializationError(
                f"Failed to initialize MediaPipe FaceLandmarker: {exc}"
            ) from exc

        logger.info(
            "FaceMeshDetector initialized (running_mode=%s, max_num_faces=%d)",
            running_mode.value,
            max_num_faces,
        )

    def _validate_image(self, image_bgr: np.ndarray) -> None:
        if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
            raise ValueError("received an empty or None image")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected an HxWx3 BGR image, got shape {image_bgr.shape}"
            )
        if self._closed:
            raise RuntimeError("called after close(); create a new detector")

    def process(self, image_bgr: np.ndarray) -> DetectionResult:
        """Run detection on a single BGR image (OpenCV's default format).

        Args:
            image_bgr: HxWx3 uint8 array in BGR order.

        Returns:
            A DetectionResult, possibly with zero faces if none were found.
        """
        self._validate_image(image_bgr)
        raw_result = self._run_landmarker(image_bgr)

        landmarks: List[List[Point3D]] = [
            [(lm.x, lm.y, lm.z) for lm in face] for face in raw_result.face_landmarks
        ]
        return DetectionResult(landmarks=landmarks, image_shape=image_bgr.shape)

    def process_and_draw(
        self,
        image_bgr: np.ndarray,
        draw_tesselation: bool = True,
        draw_contours: bool = True,
        draw_irises: bool = True,
    ) -> Tuple[DetectionResult, np.ndarray]:
        """Detect landmarks and return both the result and an annotated frame.

        This does one MediaPipe pass and gives you both the structured
        landmark data and a ready-to-display/save image.
        """
        self._validate_image(image_bgr)
        raw_result = self._run_landmarker(image_bgr)

        annotated = image_bgr.copy()
        h, w = image_bgr.shape[:2]
        landmarks: List[List[Point3D]] = []

        for face in raw_result.face_landmarks:
            landmarks.append([(lm.x, lm.y, lm.z) for lm in face])
            pixels = [(int(lm.x * w), int(lm.y * h)) for lm in face]

            if draw_tesselation:
                _draw_connections(
                    annotated, pixels, _TESSELATION_CONNECTIONS, _COLOR_TESSELATION, 1
                )
            if draw_contours:
                _draw_connections(
                    annotated, pixels, _CONTOUR_CONNECTIONS, _COLOR_CONTOURS, 1
                )
            if draw_irises:
                _draw_connections(
                    annotated, pixels, _IRIS_CONNECTIONS, _COLOR_IRISES, 1
                )

        result = DetectionResult(landmarks=landmarks, image_shape=image_bgr.shape)
        return result, annotated

    def _run_landmarker(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        if self.running_mode == RunningMode.VIDEO:
            # MediaPipe requires strictly increasing timestamps in VIDEO
            # mode; a wall-clock millisecond counter is sufficient and
            # avoids the caller having to track it themselves.
            now_ms = int(time.monotonic() * 1000)
            if now_ms <= self._frame_timestamp_ms:
                now_ms = self._frame_timestamp_ms + 1
            self._frame_timestamp_ms = now_ms
            return self._landmarker.detect_for_video(mp_image, now_ms)

        return self._landmarker.detect(mp_image)

    def close(self) -> None:
        """Release the underlying MediaPipe graph. Idempotent."""
        if not self._closed:
            self._landmarker.close()
            self._closed = True
            logger.info("FaceMeshDetector closed")

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup if the user forgot to call close()/use a
        # context manager. Never raise from __del__.
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def _draw_connections(image, pixels, connections, color, thickness) -> None:
    h, w = image.shape[:2]
    for start_idx, end_idx in connections:
        if start_idx >= len(pixels) or end_idx >= len(pixels):
            continue
        cv2.line(image, pixels[start_idx], pixels[end_idx], color, thickness, cv2.LINE_AA)


def _connections_from_mediapipe(name: str) -> List[Tuple[int, int]]:
    """Pull a named connection list out of MediaPipe's FaceLandmarksConnections,
    converting it to plain (start, end) index tuples so drawing doesn't
    depend on MediaPipe's dataclass shape at call time."""
    conn_list = getattr(mp_vision.FaceLandmarksConnections, name)
    return [(c.start, c.end) for c in conn_list]


_TESSELATION_CONNECTIONS = _connections_from_mediapipe("FACE_LANDMARKS_TESSELATION")
_CONTOUR_CONNECTIONS = _connections_from_mediapipe("FACE_LANDMARKS_CONTOURS")
_IRIS_CONNECTIONS = _connections_from_mediapipe(
    "FACE_LANDMARKS_LEFT_IRIS"
) + _connections_from_mediapipe("FACE_LANDMARKS_RIGHT_IRIS")
