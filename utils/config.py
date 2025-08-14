"""
Configuration settings for the surveillance system.
Enhanced with Employee Attendance Module settings and proper error handling.
"""
import os
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path

# Configure logging for config module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    method: str = "background_subtraction"

@dataclass
class FaceConfig:
    """Face detection and recognition configuration."""
    model: str = "hog"  # hog or cnn
    tolerance: float = 0.6
    face_gallery_path: str = "face_gallery"
    encodings_file: str = "known_faces.pkl"
    detection_scale: float = 0.5  # Scale down for faster detection

@dataclass
class AttendanceConfig:
    """Employee attendance system configuration."""
    enabled: bool = True
    face_gallery_path: str = "faces"
    attendance_file: str = "attendance.xlsx"
    face_tolerance: float = 0.5
    cooldown_seconds: int = 600  # 10 minutes
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
    """Main configuration class with proper error handling and validation."""
    
    def __init__(self):
        self.camera = CameraConfig()
        self.detection = DetectionConfig()
        self.face = FaceConfig()
        self.attendance = AttendanceConfig()
        self.anomaly = AnomalyConfig()
        self.logging = LoggingConfig()
        self.gpu = GPUConfig()
        
        # Load environment variables
        self._load_environment_variables()
        
        # Validate configuration
        self._validate_configuration()
        
        # Create necessary directories
        self._create_directories()
    
    def _load_environment_variables(self):
        """Load configuration from environment variables with proper error handling."""
        try:
            # Camera settings
            self.camera.device_id = int(os.getenv("CAMERA_ID", self.camera.device_id))
            
            # Parse resolution if provided
            resolution_str = os.getenv("CAMERA_RESOLUTION")
            if resolution_str:
                try:
                    width, height = map(int, resolution_str.split('x'))
                    self.camera.resolution = (width, height)
                except ValueError:
                    logger.warning(f"Invalid resolution format: {resolution_str}, using default")
            
            # Attendance settings
            self.attendance.enabled = os.getenv("ATTENDANCE_ENABLED", "true").lower() == "true"
            self.attendance.face_gallery_path = os.getenv("FACE_GALLERY_DIR", self.attendance.face_gallery_path)
            self.attendance.attendance_file = os.getenv("ATTENDANCE_FILE", self.attendance.attendance_file)
            
            # Convert numeric environment variables with validation
            try:
                self.attendance.face_tolerance = float(os.getenv("FACE_TOLERANCE", self.attendance.face_tolerance))
                if not 0.0 <= self.attendance.face_tolerance <= 1.0:
                    raise ValueError("Face tolerance must be between 0.0 and 1.0")
            except ValueError as e:
                logger.warning(f"Invalid face tolerance, using default: {e}")
            
            try:
                self.attendance.cooldown_seconds = int(os.getenv("COOLDOWN_SECONDS", self.attendance.cooldown_seconds))
                if self.attendance.cooldown_seconds < 0:
                    raise ValueError("Cooldown seconds must be non-negative")
            except ValueError as e:
                logger.warning(f"Invalid cooldown seconds, using default: {e}")
            
            self.attendance.auto_backup = os.getenv("AUTO_BACKUP", "true").lower() == "true"
            
            try:
                self.attendance.retention_days = int(os.getenv("RETENTION_DAYS", self.attendance.retention_days))
                if self.attendance.retention_days < 1:
                    raise ValueError("Retention days must be positive")
            except ValueError as e:
                logger.warning(f"Invalid retention days, using default: {e}")
            
            # GPU settings
            self.gpu.use_cuda = os.getenv("ENABLE_GPU", "true").lower() == "true"
            self.attendance.enable_gpu_acceleration = self.gpu.use_cuda
            
            # Logging
            log_level = os.getenv("LOG_LEVEL", self.logging.log_level).upper()
            if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                self.logging.log_level = log_level
            else:
                logger.warning(f"Invalid log level: {log_level}, using default")
            
        except Exception as e:
            logger.error(f"Error loading environment variables: {e}")
            logger.info("Using default configuration values")
    
    def _validate_configuration(self):
        """Validate configuration values."""
        errors = []
        
        # Validate camera settings
        if self.camera.device_id < 0:
            errors.append("Camera device ID must be non-negative")
        
        if self.camera.fps <= 0:
            errors.append("Camera FPS must be positive")
        
        if any(dim <= 0 for dim in self.camera.resolution):
            errors.append("Camera resolution must have positive width and height")
        
        # Validate detection settings
        if self.detection.threshold < 0:
            errors.append("Detection threshold must be non-negative")
        
        if self.detection.min_area <= 0:
            errors.append("Minimum detection area must be positive")
        
        # Validate face settings
        if not 0.0 <= self.face.tolerance <= 1.0:
            errors.append("Face tolerance must be between 0.0 and 1.0")
        
        if not 0.1 <= self.face.detection_scale <= 1.0:
            errors.append("Face detection scale must be between 0.1 and 1.0")
        
        # Validate anomaly settings
        if not 0.0 <= self.anomaly.motion_threshold <= 1.0:
            errors.append("Motion threshold must be between 0.0 and 1.0")
        
        if self.anomaly.time_window <= 0:
            errors.append("Anomaly time window must be positive")
        
        # Validate GPU settings
        if not 0.1 <= self.gpu.memory_fraction <= 1.0:
            errors.append("GPU memory fraction must be between 0.1 and 1.0")
        
        # Validate attendance settings
        if not 0.0 <= self.attendance.face_tolerance <= 1.0:
            errors.append("Attendance face tolerance must be between 0.0 and 1.0")
        
        if self.attendance.cooldown_seconds < 0:
            errors.append("Attendance cooldown seconds must be non-negative")
        
        if self.attendance.retention_days < 1:
            errors.append("Attendance retention days must be positive")
        
        if errors:
            error_msg = "Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def _create_directories(self):
        """Create required directories if they don't exist."""
        directories = [
            # Existing directories
            self.face.face_gallery_path,
            self.logging.output_dir,
            os.path.join(self.logging.output_dir, "clips"),
            os.path.join(self.logging.output_dir, "images"),
            os.path.join(self.logging.output_dir, "logs"),
            
            # Attendance directories
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
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created/verified directory: {directory}")
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")
        
        # Fix attendance file extension if needed
        if not self.attendance.attendance_file.endswith(('.xlsx', '.xls')):
            self.attendance.attendance_file += '.xlsx'
            logger.info(f"Fixed attendance file extension: {self.attendance.attendance_file}")
        
        # Ensure log file directory exists
        log_dir = os.path.dirname(self.attendance.log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"Created log directory: {log_dir}")
            except Exception as e:
                logger.error(f"Could not create log directory {log_dir}: {e}")
    
    def get_effective_config(self) -> dict:
        """Get complete effective configuration as dictionary."""
        return {
            'camera': {
                'device_id': self.camera.device_id,
                'resolution': self.camera.resolution,
                'fps': self.camera.fps,
                'buffer_size': self.camera.buffer_size
            },
            'detection': {
                'background_subtractor': self.detection.background_subtractor,
                'threshold': self.detection.threshold,
                'min_area': self.detection.min_area,
                'learning_rate': self.detection.learning_rate,
                'method': self.detection.method
            },
            'face': {
                'model': self.face.model,
                'tolerance': self.face.tolerance,
                'detection_scale': self.face.detection_scale,
                'face_gallery_path': self.face.face_gallery_path
            },
            'attendance': self.get_attendance_config_dict(),
            'anomaly': {
                'motion_threshold': self.anomaly.motion_threshold,
                'time_window': self.anomaly.time_window,
                'min_confidence': self.anomaly.min_confidence
            },
            'logging': {
                'log_level': self.logging.log_level,
                'output_dir': self.logging.output_dir,
                'save_clips': self.logging.save_clips,
                'save_images': self.logging.save_images
            },
            'gpu': {
                'use_cuda': self.gpu.use_cuda,
                'memory_fraction': self.gpu.memory_fraction
            }
        }
    
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
            'reports_directory': self.attendance.reports_directory,
            'enable_logging': self.attendance.enable_logging,
            'log_file': self.attendance.log_file,
            'show_visit_count': self.attendance.show_visit_count,
            'show_confidence': self.attendance.show_confidence,
            'show_employee_name': self.attendance.show_employee_name,
            'processing_threads': self.attendance.processing_threads,
            'detection_scale': self.attendance.detection_scale,
            'enable_gpu_acceleration': self.attendance.enable_gpu_acceleration
        }
    
    def update_attendance_config(self, **kwargs):
        """Update attendance configuration dynamically with validation."""
        for key, value in kwargs.items():
            if hasattr(self.attendance, key):
                # Validate specific fields
                if key == 'face_tolerance' and not 0.0 <= value <= 1.0:
                    logger.warning(f"Invalid face tolerance: {value}, skipping")
                    continue
                elif key == 'cooldown_seconds' and value < 0:
                    logger.warning(f"Invalid cooldown seconds: {value}, skipping")
                    continue
                elif key == 'retention_days' and value < 1:
                    logger.warning(f"Invalid retention days: {value}, skipping")
                    continue
                
                setattr(self.attendance, key, value)
                logger.info(f"Updated attendance config: {key} = {value}")
            else:
                logger.warning(f"Unknown attendance config key: {key}")
    
    def is_attendance_enabled(self) -> bool:
        """Check if attendance module is enabled."""
        return self.attendance.enabled
    
    def get_face_gallery_path(self) -> str:
        """Get the path to the employee face gallery."""
        return self.attendance.face_gallery_path
    
    def get_attendance_file_path(self) -> str:
        """Get the path to the attendance Excel file."""
        return self.attendance.attendance_file
    
    def validate_and_fix_paths(self):
        """Validate and fix file paths."""
        # Ensure attendance file has proper extension
        if not self.attendance.attendance_file.endswith(('.xlsx', '.xls')):
            self.attendance.attendance_file += '.xlsx'
            logger.info(f"Fixed attendance file extension: {self.attendance.attendance_file}")

# Global configuration instance with error handling
try:
    config = Config()
    logger.info("Configuration initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize configuration: {e}")
    # Create minimal fallback configuration
    config = Config.__new__(Config)
    config.camera = CameraConfig()
    config.detection = DetectionConfig()
    config.face = FaceConfig()
    config.attendance = AttendanceConfig()
    config.anomaly = AnomalyConfig()
    config.logging = LoggingConfig()
    config.gpu = GPUConfig()
    logger.warning("Using fallback configuration")

# Attendance-specific configuration helpers
def get_attendance_config():
    """Get attendance configuration."""
    return config.attendance

def is_attendance_enabled():
    """Check if attendance module is enabled."""
    return config.is_attendance_enabled()

def update_attendance_settings(**kwargs):
    """Update attendance settings."""
    config.update_attendance_config(**kwargs)

def validate_config():
    """Validate current configuration."""
    try:
        config._validate_configuration()
        return True
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def get_config_summary():
    """Get a summary of current configuration."""
    return {
        'camera_device': config.camera.device_id,
        'camera_resolution': config.camera.resolution,
        'attendance_enabled': config.attendance.enabled,
        'face_gallery_path': config.attendance.face_gallery_path,
        'attendance_file': config.attendance.attendance_file,
        'cooldown_seconds': config.attendance.cooldown_seconds,
        'gpu_enabled': config.gpu.use_cuda,
        'logging_level': config.logging.log_level
    }

# Configuration validation utility
def check_system_requirements():
    """Check if system meets configuration requirements."""
    requirements = {
        'python_version': True,  # Already running Python
        'directories_writable': True,
        'camera_accessible': True,
        'gpu_available': False
    }
    
    try:
        # Check directory write permissions
        import tempfile
        test_dirs = [
            config.logging.output_dir,
            config.attendance.face_gallery_path,
            config.attendance.reports_directory
        ]
        
        for test_dir in test_dirs:
            try:
                os.makedirs(test_dir, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=test_dir, delete=True):
                    pass
            except Exception:
                requirements['directories_writable'] = False
                break
        
        # Check camera access
        try:
            import cv2
            cap = cv2.VideoCapture(config.camera.device_id)
            if cap.isOpened():
                cap.release()
            else:
                requirements['camera_accessible'] = False
        except Exception:
            requirements['camera_accessible'] = False
        
        # Check GPU availability
        if config.gpu.use_cuda:
            try:
                import torch
                requirements['gpu_available'] = torch.cuda.is_available()
            except ImportError:
                requirements['gpu_available'] = False
        
    except Exception as e:
        logger.error(f"Error checking system requirements: {e}")
    
    return requirements

if __name__ == "__main__":
    # Configuration testing and validation
    print("Configuration System Test")
    print("-" * 40)
    
    # Print current configuration
    print("Current Configuration:")
    config_summary = get_config_summary()
    for key, value in config_summary.items():
        print(f"  {key}: {value}")
    
    # Validate configuration
    print(f"\nConfiguration Valid: {validate_config()}")
    
    # Check system requirements
    print("\nSystem Requirements Check:")
    requirements = check_system_requirements()
    for req, status in requirements.items():
        status_str = "✓" if status else "✗"
        print(f"  {req}: {status_str}")
    
    # Test attendance configuration
    print(f"\nAttendance Module: {'Enabled' if is_attendance_enabled() else 'Disabled'}")
    if is_attendance_enabled():
        att_config = get_attendance_config()
        print(f"  Face Gallery: {att_config.face_gallery_path}")
        print(f"  Attendance File: {att_config.attendance_file}")
        print(f"  Cooldown: {att_config.cooldown_seconds} seconds")
        print(f"  Tolerance: {att_config.face_tolerance}")
    
    print("\nConfiguration test completed.")