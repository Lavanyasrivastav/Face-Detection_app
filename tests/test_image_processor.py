"""Unit tests for facemesh.image_processor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
mediapipe = pytest.importorskip("mediapipe")

from src.facemesh.exceptions import InvalidImageError  # noqa: E402
from src.facemesh.image_processor import process_directory, process_image  # noqa: E402


@pytest.fixture
def blank_image_path(tmp_path: Path) -> Path:
    img = np.full((240, 320, 3), 200, dtype=np.uint8)
    path = tmp_path / "blank.jpg"
    cv2.imwrite(str(path), img)
    return path


class TestProcessImage:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(InvalidImageError):
            process_image(tmp_path / "does_not_exist.jpg")

    def test_unsupported_extension_raises(self, tmp_path: Path):
        bad_file = tmp_path / "not_an_image.txt"
        bad_file.write_text("hello")
        with pytest.raises(InvalidImageError):
            process_image(bad_file)

    def test_processes_blank_image_without_error(self, blank_image_path: Path, tmp_path: Path):
        out_path = tmp_path / "out.jpg"
        result = process_image(blank_image_path, output_path=out_path)
        assert result.face_count == 0
        assert out_path.exists()

    def test_writes_landmarks_json(self, blank_image_path: Path, tmp_path: Path):
        json_path = tmp_path / "landmarks.json"
        process_image(blank_image_path, landmarks_json_path=json_path)
        assert json_path.exists()
        assert '"face_count": 0' in json_path.read_text()


class TestProcessDirectory:
    def test_empty_directory_returns_empty_list(self, tmp_path: Path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"
        result = process_directory(input_dir, output_dir)
        assert result == []

    def test_nonexistent_directory_raises(self, tmp_path: Path):
        with pytest.raises(InvalidImageError):
            process_directory(tmp_path / "nope", tmp_path / "out")

    def test_processes_all_images_in_directory(self, tmp_path: Path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        for i in range(3):
            img = np.full((100, 100, 3), 50 * i, dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"img_{i}.jpg"), img)

        output_dir = tmp_path / "out"
        written = process_directory(input_dir, output_dir, save_landmarks_json=False)
        assert len(written) == 3
        for path in written:
            assert path.exists()
