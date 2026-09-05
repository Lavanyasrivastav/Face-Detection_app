"""Custom exception types for the facemesh package.

Using specific exception types (instead of bare Exception/RuntimeError)
lets callers catch and handle failure modes precisely -- e.g. retrying
on a transient camera issue vs. aborting on a bad model file.
"""


class FaceMeshError(Exception):
    """Base class for all facemesh-specific errors."""


class CameraNotFoundError(FaceMeshError):
    """Raised when the requested camera device cannot be opened."""


class InvalidImageError(FaceMeshError):
    """Raised when an input image path is missing, unreadable, or empty."""


class ModelInitializationError(FaceMeshError):
    """Raised when the underlying MediaPipe model fails to initialize."""
