"""
facemesh
========

A production-ready wrapper around MediaPipe Face Mesh for detecting
468 (or 478, with iris refinement) 3D facial landmarks in images,
videos, and live webcam streams.
"""

from .detector import FaceMeshDetector, DetectionResult
from .exceptions import (
    FaceMeshError,
    CameraNotFoundError,
    InvalidImageError,
    ModelInitializationError,
)

__all__ = [
    "FaceMeshDetector",
    "DetectionResult",
    "FaceMeshError",
    "CameraNotFoundError",
    "InvalidImageError",
    "ModelInitializationError",
]

__version__ = "1.0.0"
