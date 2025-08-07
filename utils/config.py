"""
Configuration settings for the surveillance system.
"""
import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CameraConfig:
    """Camera configuration settings."""
    device_id: int = 0
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    buffer_size: int = 1

@dataclass
class DetectionConfig:
    """Motion detection configuration."""
    background_subtractor: str = "MOG2"  # MOG2, KNN, or GMG
    threshold: int = 25
    min_area: int = 500
    gaussian_blur: Tuple[int, int] = (21, 21)
    morphology_kernel_size: Tuple[int, int] = (5, 5)
    learning_rate: float = 0.01

@dataclass
class FaceConfig:
    """Face detection and recognition configuration."""
    model: str = "hog"  # hog or cnn
    tolerance: float = 0.6
    face_gallery_path: str = "face_gallery"
    encodings_file: str = "known_faces.pkl"
    detection_scale: float = 0.5  # Scale down for faster detection

@dataclass
class AnomalyConfig:
    """Anomaly detection configuration."""
    motion_threshold: float = 0.3
    time_window: int = 30  # seconds
    min_confidence: float = 0.7
    tracking_max_disappeared: int = 20

@dataclass
class LoggingConfig:
    """Logging and storage configuration."""
    log_level: str = "INFO"
    output_dir: str = "surveillance_output"
    save_clips: bool = True
    save_images: bool = True
    clip_duration: int = 10  # seconds
    max_storage_gb: float = 5.0

@dataclass
class GPUConfig:
    """GPU optimization settings."""
    use_cuda: bool = True
    memory_fraction: float = 0.7
    cache_cleanup_interval: int = 100  # frames

class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.camera = CameraConfig()
        self.detection = DetectionConfig()
        self.face = FaceConfig()
        self.anomaly = AnomalyConfig()
        self.logging = LoggingConfig()
        self.gpu = GPUConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create required directories if they don't exist."""
        directories = [
            self.face.face_gallery_path,
            self.logging.output_dir,
            os.path.join(self.logging.output_dir, "clips"),
            os.path.join(self.logging.output_dir, "images"),
            os.path.join(self.logging.output_dir, "logs")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Global configuration instance
config = Config()   