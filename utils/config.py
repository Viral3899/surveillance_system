"""
Configuration settings for the surveillance system.
Enhanced with Employee Attendance Module settings.
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

# 🆕 NEW: Attendance Configuration
@dataclass
class AttendanceConfig:
    """Employee attendance system configuration."""
    enabled: bool = True
    face_gallery_path: str = "faces"
    attendance_file: str = "attendance.xlsx"
    face_tolerance: float = 0.5
    cooldown_seconds: int = 5
    encodings_cache_file: str = "face_encodings.pkl"
    
    # Performance settings
    detection_scale: float = 1.0
    enable_gpu_acceleration: bool = True
    processing_threads: int = 2
    
    # Logging and reports
    enable_logging: bool = True
    log_file: str = "surveillance_data/logs/attendance.log"
    reports_directory: str = "attendance_reports"
    auto_backup: bool = True
    backup_interval_hours: int = 24
    
    # Data retention
    max_records: int = 50000
    retention_days: int = 90
    auto_cleanup: bool = True
    
    # Display settings
    show_visit_count: bool = True
    show_confidence: bool = True
    show_employee_name: bool = True
    bounding_box_color_known: Tuple[int, int, int] = (0, 255, 0)  # Green
    bounding_box_color_unknown: Tuple[int, int, int] = (0, 0, 255)  # Red
    
    # API settings
    enable_api_endpoints: bool = True
    api_rate_limit: int = 100  # requests per minute
    
    # Security settings
    require_authentication: bool = False
    encrypt_face_data: bool = False
    anonymize_logs: bool = False

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
        self.attendance = AttendanceConfig()  # 🆕 NEW: Attendance config
        self.anomaly = AnomalyConfig()
        self.logging = LoggingConfig()
        self.gpu = GPUConfig()
        
        # Load environment variables
        self._load_environment_variables()
        
        # Create necessary directories
        self._create_directories()
    
    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        # Camera settings
        self.camera.device_id = int(os.getenv("CAMERA_ID", self.camera.device_id))
        
        # Attendance settings
        self.attendance.enabled = os.getenv("ATTENDANCE_ENABLED", "true").lower() == "true"
        self.attendance.face_gallery_path = os.getenv("FACE_GALLERY_DIR", self.attendance.face_gallery_path)
        self.attendance.attendance_file = os.getenv("ATTENDANCE_FILE", self.attendance.attendance_file)
        self.attendance.face_tolerance = float(os.getenv("FACE_TOLERANCE", self.attendance.face_tolerance))
        self.attendance.cooldown_seconds = int(os.getenv("COOLDOWN_SECONDS", self.attendance.cooldown_seconds))
        self.attendance.auto_backup = os.getenv("AUTO_BACKUP", "true").lower() == "true"
        self.attendance.retention_days = int(os.getenv("RETENTION_DAYS", self.attendance.retention_days))
        
        # GPU settings
        self.gpu.use_cuda = os.getenv("ENABLE_GPU", "true").lower() == "true"
        self.attendance.enable_gpu_acceleration = self.gpu.use_cuda
        
        # Logging
        self.logging.log_level = os.getenv("LOG_LEVEL", self.logging.log_level)
    
    def _create_directories(self):
        """Create required directories if they don't exist."""
        directories = [
            # Existing directories
            self.face.face_gallery_path,
            self.logging.output_dir,
            os.path.join(self.logging.output_dir, "clips"),
            os.path.join(self.logging.output_dir, "images"),
            os.path.join(self.logging.output_dir, "logs"),
            
            # 🆕 NEW: Attendance directories
            self.attendance.face_gallery_path,
            self.attendance.reports_directory,
            os.path.join(self.attendance.reports_directory, "backup"),
            os.path.dirname(self.attendance.log_file) if "/" in self.attendance.log_file else "logs",
            "cache",
            "backup",
            "backup/daily",
            "backup/weekly",
            "backup/monthly"
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {e}")
    
    # 🆕 NEW: Attendance-specific methods
    def get_attendance_config_dict(self) -> dict:
        """Get attendance configuration as dictionary."""
        return {
            'enabled': self.attendance.enabled,
            'face_gallery_path': self.attendance.face_gallery_path,
            'attendance_file': self.attendance.attendance_file,
            'face_tolerance': self.attendance.face_tolerance,
            'cooldown_seconds': self.attendance.cooldown_seconds,
            'auto_backup': self.attendance.auto_backup,
            'retention_days': self.attendance.retention_days,
            'reports_directory': self.attendance.reports_directory
        }
    
    def update_attendance_config(self, **kwargs):
        """Update attendance configuration dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.attendance, key):
                setattr(self.attendance, key, value)
    
    def is_attendance_enabled(self) -> bool:
        """Check if attendance module is enabled."""
        return self.attendance.enabled
    
    def get_face_gallery_path(self) -> str:
        """Get the path to the employee face gallery."""
        return self.attendance.face_gallery_path
    
    def get_attendance_file_path(self) -> str:
        """Get the path to the attendance Excel file."""
        return self.attendance.attendance_file

# Global configuration instance
config = Config()

# 🆕 NEW: Attendance-specific configuration helpers
def get_attendance_config():
    """Get attendance configuration."""
    return config.attendance

def is_attendance_enabled():
    """Check if attendance module is enabled."""
    return config.is_attendance_enabled()

def update_attendance_settings(**kwargs):
    """Update attendance settings."""
    config.update_attendance_config(**kwargs)

config = Config()  # Ensure config is initialized