"""Unit tests for facemesh.detector.

These tests avoid requiring a real face image by testing:
  1. Pure-Python logic (DetectionResult) with no MediaPipe dependency.
  2. Input validation on FaceMeshDetector.process (empty/malformed arrays).
  3. Behavior on a real, faceless synthetic image (should return zero faces,
     not raise).

Tests that need mediapipe/opencv installed are skipped automatically if
those packages aren't available in the environment, so `pytest` still runs
cleanly in a minimal CI image without heavy CV dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.facemesh.detector import DetectionResult

cv2 = pytest.importorskip("cv2")
mediapipe = pytest.importorskip("mediapipe")

from src.facemesh.detector import FaceMeshDetector, RunningMode  # noqa: E402


class TestDetectionResult:
    def test_empty_result_has_no_detections(self):
        result = DetectionResult(landmarks=[], image_shape=(100, 100, 3))
        assert result.face_count == 0
        assert result.has_detections is False

    def test_face_count_matches_landmarks_length(self):
        fake_face = [(0.1, 0.2, 0.0)] * 468
        result = DetectionResult(landmarks=[fake_face, fake_face], image_shape=(480, 640, 3))
        assert result.face_count == 2
        assert result.has_detections is True

    def test_landmarks_as_pixels_converts_normalized_coords(self):
        face = [(0.5, 0.5, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 0.0)]
        result = DetectionResult(landmarks=[face], image_shape=(200, 400, 3))
        pixels = result.landmarks_as_pixels(0)
        assert pixels == [(200, 100), (400, 200), (0, 0)]

    def test_landmarks_as_pixels_out_of_range_raises(self):
        result = DetectionResult(landmarks=[], image_shape=(100, 100, 3))
        with pytest.raises(IndexError):
            result.landmarks_as_pixels(0)


class TestFaceMeshDetectorValidation:
    def test_rejects_none_image(self):
        with FaceMeshDetector(running_mode=RunningMode.IMAGE) as detector:
            with pytest.raises(ValueError):
                detector.process(None)

    def test_rejects_empty_image(self):
        with FaceMeshDetector(running_mode=RunningMode.IMAGE) as detector:
            with pytest.raises(ValueError):
                detector.process(np.array([]))

    def test_rejects_wrong_channel_count(self):
        grayscale = np.zeros((100, 100), dtype=np.uint8)
        with FaceMeshDetector(running_mode=RunningMode.IMAGE) as detector:
            with pytest.raises(ValueError):
                detector.process(grayscale)

    def test_raises_after_close(self):
        detector = FaceMeshDetector(running_mode=RunningMode.IMAGE)
        detector.close()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            detector.process(image)


class TestFaceMeshDetectorOnSyntheticImage:
    def test_blank_image_returns_zero_faces(self):
        """A solid-color image has no face; detection should return
        cleanly with zero results rather than raising."""
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        with FaceMeshDetector(running_mode=RunningMode.IMAGE, max_num_faces=1) as detector:
            result = detector.process(blank)
        assert result.face_count == 0
        assert result.image_shape == (480, 640, 3)

    def test_context_manager_closes_detector(self):
        with FaceMeshDetector(running_mode=RunningMode.IMAGE) as detector:
            assert detector._closed is False
        assert detector._closed is True
