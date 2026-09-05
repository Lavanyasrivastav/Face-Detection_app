"""Webcam / video-file processing pipeline.

Runs the detector on a live stream and displays it in an OpenCV window,
optionally recording the annotated output to disk. Designed to degrade
gracefully: a dropped frame or a momentarily disconnected camera logs a
warning instead of crashing the whole process.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import cv2

from .detector import FaceMeshDetector, RunningMode
from .exceptions import CameraNotFoundError
from .logging_config import get_logger

logger = get_logger(__name__)


class FPSCounter:
    """Simple exponential-moving-average FPS counter for on-screen display."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._fps: Optional[float] = None
        self._last_t: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last_t is not None:
            instant_fps = 1.0 / max(now - self._last_t, 1e-6)
            self._fps = (
                instant_fps
                if self._fps is None
                else self._alpha * instant_fps + (1 - self._alpha) * self._fps
            )
        self._last_t = now
        return self._fps or 0.0


def run_webcam(
    camera_index: int = 0,
    max_num_faces: int = 1,
    mirror: bool = True,
    output_video_path: Optional[Union[str, Path]] = None,
    window_name: str = "Face Mesh Detection",
    show_fps: bool = True,
) -> None:
    """Open a webcam, run live face mesh detection, and display it.

    Press 'q' or ESC in the display window to exit.

    Args:
        camera_index: OS camera device index (0 is usually the default cam).
        max_num_faces: Max simultaneous faces to track.
        mirror: Flip horizontally for a natural "mirror" selfie view.
        output_video_path: If given, write the annotated stream to this
            video file (mp4) alongside the live preview.
        window_name: Title of the OpenCV display window.
        show_fps: Overlay a rolling FPS counter.

    Raises:
        CameraNotFoundError: If the camera device cannot be opened.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise CameraNotFoundError(
            f"Could not open camera at index {camera_index}. "
            "Check that it's connected, not in use by another app, "
            "and that OS camera permissions are granted."
        )

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_video_path is not None:
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path), fourcc, fps_in, (frame_w, frame_h)
        )
        logger.info("Recording annotated stream to %s", output_video_path)

    fps_counter = FPSCounter()
    consecutive_failures = 0
    max_consecutive_failures = 30  # ~1s at 30fps before giving up

    logger.info(
        "Starting webcam stream (camera_index=%d, %dx%d @ %.1f fps). "
        "Press 'q' or ESC to quit.",
        camera_index,
        frame_w,
        frame_h,
        fps_in,
    )

    try:
        # VIDEO mode enables temporal tracking across frames, which is
        # faster and more stable than re-detecting from scratch every
        # frame (the appropriate mode for a live stream, vs IMAGE mode
        # for independent photos).
        with FaceMeshDetector(
            running_mode=RunningMode.VIDEO,
            max_num_faces=max_num_faces,
        ) as detector:
            while True:
                success, frame = cap.read()
                if not success:
                    consecutive_failures += 1
                    logger.warning(
                        "Failed to read frame (%d/%d consecutive failures)",
                        consecutive_failures,
                        max_consecutive_failures,
                    )
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive frame failures; stopping")
                        break
                    continue
                consecutive_failures = 0

                if mirror:
                    frame = cv2.flip(frame, 1)

                result, annotated = detector.process_and_draw(frame)

                if show_fps:
                    fps = fps_counter.tick()
                    cv2.putText(
                        annotated,
                        f"FPS: {fps:.1f}  Faces: {result.face_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                if writer is not None:
                    writer.write(annotated)

                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    logger.info("Quit key pressed; stopping stream")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Webcam session ended and resources released")
