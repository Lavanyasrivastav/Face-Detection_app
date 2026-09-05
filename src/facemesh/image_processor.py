"""Batch/single-image processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import cv2

from .detector import DetectionResult, FaceMeshDetector, RunningMode
from .exceptions import InvalidImageError
from .logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_image(path: Path):
    if not path.exists():
        raise InvalidImageError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported image extension '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    image = cv2.imread(str(path))
    if image is None:
        raise InvalidImageError(
            f"OpenCV failed to decode image (corrupt file?): {path}"
        )
    return image


def process_image(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    landmarks_json_path: Optional[Union[str, Path]] = None,
    max_num_faces: int = 5,
) -> DetectionResult:
    """Run face mesh detection on a single static image.

    Args:
        input_path: Path to the source image.
        output_path: If given, save the annotated image here.
        landmarks_json_path: If given, dump raw normalized landmark
            coordinates to this JSON file (one array of [x, y, z] points
            per detected face, 478 points including irises).
        max_num_faces: Maximum faces to detect in the image.

    Returns:
        The DetectionResult for the image.

    Raises:
        InvalidImageError: If the input path doesn't exist, has an
            unsupported extension, or can't be decoded.
    """
    input_path = Path(input_path)
    image = _load_image(input_path)
    logger.info("Loaded image %s (shape=%s)", input_path, image.shape)

    with FaceMeshDetector(
        running_mode=RunningMode.IMAGE,
        max_num_faces=max_num_faces,
    ) as detector:
        result, annotated = detector.process_and_draw(image)

    logger.info("Detected %d face(s) in %s", result.face_count, input_path.name)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        logger.info("Saved annotated image to %s", output_path)

    if landmarks_json_path is not None:
        landmarks_json_path = Path(landmarks_json_path)
        landmarks_json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_image": str(input_path),
            "image_shape": result.image_shape,
            "face_count": result.face_count,
            "faces": [
                [{"x": x, "y": y, "z": z} for x, y, z in face]
                for face in result.landmarks
            ],
        }
        landmarks_json_path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved landmark data to %s", landmarks_json_path)

    return result


def process_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    max_num_faces: int = 5,
    save_landmarks_json: bool = True,
) -> List[Path]:
    """Batch-process every supported image in a directory (non-recursive).

    Returns:
        List of output image paths that were successfully written.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise InvalidImageError(f"Not a directory: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        logger.warning("No supported images found in %s", input_dir)
        return []

    written: List[Path] = []
    for path in image_paths:
        out_image = output_dir / f"{path.stem}_mesh{path.suffix}"
        out_json = (
            output_dir / f"{path.stem}_landmarks.json"
            if save_landmarks_json
            else None
        )
        try:
            process_image(
                input_path=path,
                output_path=out_image,
                landmarks_json_path=out_json,
                max_num_faces=max_num_faces,
            )
            written.append(out_image)
        except InvalidImageError as exc:
            # One bad file shouldn't kill a batch job -- log and continue.
            logger.error("Skipping %s: %s", path, exc)

    logger.info("Batch complete: %d/%d images processed", len(written), len(image_paths))
    return written
