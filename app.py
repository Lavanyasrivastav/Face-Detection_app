"""Streamlit web demo for face mesh detection.

Run with:
    streamlit run app.py

Two modes:
  - "Upload Image": always available, only needs streamlit + this repo's
    own dependencies (mediapipe/opencv/numpy).
  - "Live Webcam": real-time detection in the browser via streamlit-webrtc.
    This is an optional extra (see requirements-web.txt) -- the app
    degrades gracefully with an install hint if it's not present, so the
    image demo never breaks because of it.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import streamlit as st

from src.facemesh.detector import FaceMeshDetector, RunningMode
from src.facemesh.exceptions import FaceMeshError, ModelInitializationError

st.set_page_config(page_title="Face Mesh Detection", page_icon="🧑‍💻", layout="wide")


@st.cache_resource(show_spinner="Loading face mesh model (first run only)...")
def _get_image_detector(max_num_faces: int) -> FaceMeshDetector:
    """Cached per (max_num_faces) value so the model bundle is loaded once
    per session, not on every interaction."""
    return FaceMeshDetector(running_mode=RunningMode.IMAGE, max_num_faces=max_num_faces)


def _decode_upload(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image_bgr


def _landmarks_json(result) -> str:
    payload = {
        "face_count": result.face_count,
        "image_shape": result.image_shape,
        "faces": [
            [{"x": x, "y": y, "z": z} for x, y, z in face] for face in result.landmarks
        ],
    }
    return json.dumps(payload, indent=2)


def render_image_tab(max_num_faces: int) -> None:
    uploaded = st.file_uploader(
        "Upload a photo", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded is None:
        st.info("Upload an image above, or try the sample.")
        if st.button("Use a generated sample image"):
            uploaded = None  # sample path handled below
            _run_on_sample(max_num_faces)
        return

    image_bgr = _decode_upload(uploaded)
    if image_bgr is None:
        st.error("Couldn't decode that file — is it a valid image?")
        return

    _run_and_display(image_bgr, max_num_faces)


def _run_on_sample(max_num_faces: int) -> None:
    # A synthetic placeholder so the demo has something to click even
    # without a photo on hand. It won't contain a face, which is itself
    # a useful illustration of the "0 faces detected" path.
    sample = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.putText(
        sample, "No face in this sample - upload a real photo", (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA,
    )
    _run_and_display(sample, max_num_faces)


def _run_and_display(image_bgr: np.ndarray, max_num_faces: int) -> None:
    try:
        detector = _get_image_detector(max_num_faces)
        result, annotated = detector.process_and_draw(image_bgr)
    except ModelInitializationError as exc:
        st.error(
            "Couldn't load the face mesh model. This usually means the "
            f"model bundle failed to download: {exc}"
        )
        return
    except FaceMeshError as exc:
        st.error(f"Detection failed: {exc}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader(f"Annotated — {result.face_count} face(s)")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    if result.has_detections:
        st.download_button(
            "Download landmarks (JSON)",
            data=_landmarks_json(result),
            file_name="landmarks.json",
            mime="application/json",
        )


def render_webcam_tab(max_num_faces: int) -> None:
    try:
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
        import av
    except ImportError:
        st.warning(
            "Live webcam mode needs the optional `streamlit-webrtc` and `av` "
            "packages, which aren't installed.\n\n"
            "Install with:\n```\npip install -r requirements-web.txt\n```\n"
            "then restart the app. The **Upload Image** tab works without them."
        )
        return

    class _Processor(VideoProcessorBase):
        def __init__(self) -> None:
            # VIDEO mode (not cached/shared) -- each browser session gets
            # its own detector with its own tracking state.
            self.detector = FaceMeshDetector(
                running_mode=RunningMode.VIDEO, max_num_faces=max_num_faces
            )

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            try:
                _, annotated = self.detector.process_and_draw(img)
            except FaceMeshError:
                annotated = img
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    st.caption("Grant camera access in the browser prompt, then wait a moment for it to connect.")
    webrtc_streamer(
        key="facemesh-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=_Processor,
        media_stream_constraints={"video": True, "audio": False},
    )


def main() -> None:
    st.title("🧑‍💻 Face Mesh Detection")
    st.caption("468/478 3D facial landmarks, powered by MediaPipe FaceLandmarker.")

    max_num_faces = st.sidebar.slider("Max faces to detect", 1, 10, 1)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "First run downloads and caches a small model bundle "
        "(needs an internet connection once)."
    )

    tab_image, tab_webcam = st.tabs(["📷 Upload Image", "🎥 Live Webcam"])
    with tab_image:
        render_image_tab(max_num_faces)
    with tab_webcam:
        render_webcam_tab(max_num_faces)


if __name__ == "__main__":
    main()
