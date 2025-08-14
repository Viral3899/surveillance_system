"""Face detection and recognition module for surveillance system."""
from .face_detector import FaceDetector, FaceDetection
from .face_matcher import FaceMatcher
__all__ = ['FaceDetector', 'FaceDetection', 'FaceMatcher']