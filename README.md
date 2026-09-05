# Face Mesh Detection

Production-ready real-time face mesh detection built on [MediaPipe](https://mediapipe.dev), estimating 468 (or 478, with iris refinement) 3D facial landmarks from a single camera — no depth sensor required.

## Features

- **Typed, testable API** — `FaceMeshDetector` wraps MediaPipe behind a small interface (`DetectionResult`, custom exceptions) so the rest of the app never touches MediaPipe/OpenCV objects directly.
- **Three usage modes** — single image, batch directory, live webcam (with optional recording).
- **Robust error handling** — missing files, corrupt images, disconnected cameras, and empty frames raise specific exceptions instead of crashing.
- **Structured logging** instead of `print`.
- **Unit tests** with `pytest` (auto-skip if `mediapipe`/`opencv` aren't installed).
- **CI** via GitHub Actions (lint, format check, tests across Python 3.9–3.11).
- **Docker image** for batch/image-mode deployment.

## Installation

```bash
git clone <this-repo-url>
cd face-mesh-detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For development (tests, linting):

```bash
pip install -r requirements-dev.txt
```

Or install as a package (adds the `facemesh` console command):

```bash
pip install .
```

## Usage

### Single image

```bash
python main.py image --input photo.jpg --output result.jpg --landmarks-json landmarks.json
```

### Batch directory

```bash
python main.py batch --input-dir photos/ --output-dir results/
```

### Live webcam

```bash
python main.py webcam --camera 0
```

Press **q** or **Esc** to quit. Add `--record out.mp4` to save the annotated stream.

Run `python main.py <command> --help` for the full flag list (`--max-faces`, `--no-iris`, `--no-mirror`, etc.).

### Web demo (Streamlit)

A browser-based demo lives in `app.py` — no OpenCV desktop window required.

```bash
pip install -r requirements-app.txt
streamlit run app.py
```

This gives you an **Upload Image** tab out of the box. For a **Live Webcam** tab running in the browser (via WebRTC instead of an OpenCV window), install the optional extra first:

```bash
pip install -r requirements-web.txt
streamlit run app.py
```

If `streamlit-webrtc`/`av` aren't installed, the Live Webcam tab shows an install hint instead of crashing — the Upload Image tab always works.

> **Note:** `app.py` and the `streamlit-webrtc` live-webcam path haven't been runtime-tested end-to-end (this sandbox has neither `streamlit` nor a display/camera to verify against). The code was written and compile-checked against the documented, stable APIs for both libraries. Please treat it as a solid first pass and sanity-check it against your own machine before relying on it.

### As a library

```python
import cv2
from src.facemesh import FaceMeshDetector

image = cv2.imread("photo.jpg")

with FaceMeshDetector(static_image_mode=True, max_num_faces=1) as detector:
    result, annotated = detector.process_and_draw(image)

print(f"Detected {result.face_count} face(s)")
if result.has_detections:
    pixel_coords = result.landmarks_as_pixels(face_index=0)  # 468/478 (x, y) points

cv2.imwrite("annotated.jpg", annotated)
```

## Project layout

```
face-mesh-detection/
├── main.py                      # CLI entry point
├── app.py                       # Streamlit web demo (upload + live webcam)
├── src/facemesh/
│   ├── detector.py              # Core FaceMeshDetector + DetectionResult
│   ├── image_processor.py       # Single-image & batch pipelines
│   ├── video_processor.py       # Webcam/video pipeline
│   ├── model_manager.py         # Downloads/caches the FaceLandmarker model
│   ├── cli.py                   # argparse CLI (also the pip console_script)
│   ├── exceptions.py            # FaceMeshError and subclasses
│   └── logging_config.py        # Shared logger factory
├── tests/                       # pytest unit tests
├── Dockerfile
├── .github/workflows/ci.yml
├── requirements.txt             # Core (CLI/library)
├── requirements-app.txt         # + Streamlit (image upload demo)
├── requirements-web.txt         # + streamlit-webrtc (live webcam demo)
├── requirements-dev.txt         # + pytest/lint tools
└── pyproject.toml
```

## Testing

```bash
pytest --cov=src/facemesh --cov-report=term-missing
```

## Docker

```bash
docker build -t facemesh .
docker run -v $(pwd)/data:/app/data facemesh image --input /app/data/input/photo.jpg --output /app/data/output/result.jpg
```

## Troubleshooting

| Problem | Likely cause |
|---|---|
| `CameraNotFoundError` | Camera in use by another app, wrong `--camera` index, or missing OS camera permission. |
| `InvalidImageError` | File path wrong, unsupported extension, or corrupt image. |
| Low FPS on webcam | Try `--no-iris` (skips iris landmarks) or lower `--max-faces`. |
| `ModelInitializationError` | Corrupted MediaPipe install — try `pip install --force-reinstall mediapipe`. |

## Citation

```bibtex
@article{lugaresi2019mediapipe,
  title={Mediapipe: A framework for building perception pipelines},
  author={Lugaresi, Camillo and Tang, Jiuqiang and Nash, Hadon and McClanahan, Chris and Uboweja, Esha and Hays, Michael and Zhang, Fan and Chang, Chuo-Ling and Yong, Ming Guang and Lee, Juhyun and others},
  journal={arXiv preprint arXiv:1906.08172},
  year={2019}
}
```

## License

MIT
