"""
Logging utilities for the surveillance system.
Enhanced with Employee Attendance Module logging.
"""
import logging
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Any, Dict, List
from .config import config

class SurveillanceLogger:
    """Enhanced logger for surveillance system events and media."""
    
    def __init__(self, name: str = "surveillance"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.logging.log_level))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording_start_time: Optional[datetime] = None
        
        # 🆕 NEW: Attendance-specific logging
        self.attendance_logger = self._setup_attendance_logger()
        self.attendance_events = []  # Store recent attendance events
        self.max_attendance_events = 1000
    
    def _setup_handlers(self):
        """Setup logging handlers for file and console output."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = os.path.join(config.logging.output_dir, "logs", "surveillance.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    # 🆕 NEW: Setup attendance-specific logger
    def _setup_attendance_logger(self):
        """Setup separate logger for attendance events."""
        if not config.attendance.enabled:
            return None
        
        attendance_logger = logging.getLogger("attendance")
        attendance_logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not attendance_logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - ATTENDANCE - %(levelname)s - %(message)s'
            )
            
            # Attendance file handler
            if config.attendance.enable_logging:
                attendance_file_handler = logging.FileHandler(config.attendance.log_file)
                attendance_file_handler.setFormatter(formatter)
                attendance_logger.addHandler(attendance_file_handler)
            
            # Also log to main surveillance log
            main_handler = logging.FileHandler(
                os.path.join(config.logging.output_dir, "logs", "surveillance.log")
            )
            main_handler.setFormatter(formatter)
            attendance_logger.addHandler(main_handler)
        
        return attendance_logger
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def log_event(self, event_type: str, details: dict):
        """Log a surveillance event with structured details."""
        timestamp = datetime.now().isoformat()
        message = f"EVENT: {event_type} | {details} | {timestamp}"
        self.info(message)
    
    # 🆕 NEW: Attendance-specific logging methods
    def log_attendance_event(self, employee_id: str, event_type: str = "DETECTED", 
                           details: Dict = None, confidence: float = 0.0):
        """Log an attendance-related event."""
        if not self.attendance_logger:
            return
        
        timestamp = datetime.now()
        event_details = details or {}
        
        # Create attendance event record
        attendance_event = {
            'timestamp': timestamp.isoformat(),
            'employee_id': employee_id,
            'event_type': event_type,
            'confidence': confidence,
            'details': event_details
        }
        
        # Add to recent events
        self.attendance_events.append(attendance_event)
        if len(self.attendance_events) > self.max_attendance_events:
            self.attendance_events = self.attendance_events[-self.max_attendance_events:]
        
        # Log the event
        message = f"Employee: {employee_id} | Event: {event_type} | Confidence: {confidence:.2f}"
        if event_details:
            message += f" | Details: {event_details}"
        
        self.attendance_logger.info(message)
        
        # Also log significant events to main log
        if event_type in ["FIRST_DETECTION", "NEW_EMPLOYEE", "ATTENDANCE_LOGGED"]:
            self.info(f"ATTENDANCE - {message}")
    
    def log_attendance_detection(self, employee_id: str, visit_count: int, 
                               confidence: float, location: tuple = None):
        """Log attendance detection event."""
        details = {
            'visit_count': visit_count,
            'location': location
        }
        self.log_attendance_event(
            employee_id, 
            "ATTENDANCE_LOGGED", 
            details, 
            confidence
        )
    
    def log_face_recognition_stats(self, total_faces: int, known_faces: int, 
                                 unknown_faces: int, processing_time: float):
        """Log face recognition statistics."""
        if not self.attendance_logger:
            return
        
        stats_message = (f"Face Recognition Stats - Total: {total_faces}, "
                        f"Known: {known_faces}, Unknown: {unknown_faces}, "
                        f"Processing Time: {processing_time:.3f}s")
        
        self.attendance_logger.debug(stats_message)
    
    def log_system_performance(self, fps: float, memory_usage: float, 
                             gpu_usage: float = None):
        """Log system performance metrics."""
        perf_details = {
            'fps': fps,
            'memory_usage_mb': memory_usage
        }
        if gpu_usage is not None:
            perf_details['gpu_usage_percent'] = gpu_usage
        
        self.log_attendance_event("SYSTEM", "PERFORMANCE", perf_details)
    
    def get_recent_attendance_events(self, hours: int = 24) -> List[Dict]:
        """Get recent attendance events within specified hours."""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        recent_events = []
        for event in self.attendance_events:
            event_time = datetime.fromisoformat(event['timestamp']).timestamp()
            if event_time >= cutoff_time:
                recent_events.append(event)
        
        return recent_events
    
    def get_attendance_summary(self, hours: int = 24) -> Dict:
        """Get attendance summary for specified time period."""
        recent_events = self.get_recent_attendance_events(hours)
        
        # Count events by employee
        employee_counts = {}
        event_types = {}
        
        for event in recent_events:
            emp_id = event['employee_id']
            event_type = event['event_type']
            
            if emp_id not in employee_counts:
                employee_counts[emp_id] = 0
            if event_type == "ATTENDANCE_LOGGED":
                employee_counts[emp_id] += 1
            
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
        
        return {
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'unique_employees': len(employee_counts),
            'employee_attendance_counts': employee_counts,
            'event_type_counts': event_types,
            'last_updated': datetime.now().isoformat()
        }
    
    def save_image(self, frame: np.ndarray, event_type: str, metadata: dict = None) -> str:
        """Save an image with timestamp and metadata."""
        if not config.logging.save_images:
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{event_type}_{timestamp}.jpg"
        filepath = os.path.join(config.logging.output_dir, "images", filename)
        
        try:
            cv2.imwrite(filepath, frame)
            self.info(f"Image saved: {filename}")
            
            if metadata:
                self.log_event("IMAGE_SAVED", {
                    "filename": filename,
                    "event_type": event_type,
                    **metadata
                })
            
            return filepath
        except Exception as e:
            self.error(f"Failed to save image: {e}")
            return ""
    
    # 🆕 NEW: Attendance-specific image saving
    def save_attendance_image(self, frame: np.ndarray, employee_id: str, 
                            visit_count: int, confidence: float) -> str:
        """Save image for attendance event."""
        metadata = {
            'employee_id': employee_id,
            'visit_count': visit_count,
            'confidence': confidence,
            'event_category': 'attendance'
        }
        
        filename = f"attendance_{employee_id}_{visit_count}"
        filepath = self.save_image(frame, filename, metadata)
        
        if filepath:
            self.log_attendance_event(employee_id, "IMAGE_SAVED", {
                'filepath': filepath,
                'visit_count': visit_count
            })
        
        return filepath
    
    def start_recording(self, frame: np.ndarray, event_type: str) -> bool:
        """Start video recording for an event."""
        if not config.logging.save_clips or self.video_writer is not None:
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{event_type}_{timestamp}.mp4"
        filepath = os.path.join(config.logging.output_dir, "clips", filename)
        
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        try:
            self.video_writer = cv2.VideoWriter(
                filepath, fourcc, config.camera.fps, (width, height)
            )
            self.recording_start_time = datetime.now()
            
            if self.video_writer.isOpened():
                self.info(f"Started recording: {filename}")
                self.log_event("RECORDING_STARTED", {
                    "filename": filename,
                    "event_type": event_type,
                    "resolution": f"{width}x{height}"
                })
                return True
            else:
                self.error(f"Failed to open video writer for {filename}")
                return False
                
        except Exception as e:
            self.error(f"Failed to start recording: {e}")
            return False
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """Write a frame to the current video recording."""
        if self.video_writer is None:
            return False
        
        try:
            self.video_writer.write(frame)
            return True
        except Exception as e:
            self.error(f"Failed to write frame: {e}")
            return False
    
    def stop_recording(self) -> bool:
        """Stop the current video recording."""
        if self.video_writer is None:
            return False
        
        try:
            duration = (datetime.now() - self.recording_start_time).total_seconds()
            self.video_writer.release()
            self.video_writer = None
            self.recording_start_time = None
            
            self.info(f"Stopped recording (duration: {duration:.2f}s)")
            self.log_event("RECORDING_STOPPED", {"duration": duration})
            return True
            
        except Exception as e:
            self.error(f"Failed to stop recording: {e}")
            return False
    
    def should_stop_recording(self) -> bool:
        """Check if recording should be stopped based on duration."""
        if self.recording_start_time is None:
            return False
        
        duration = (datetime.now() - self.recording_start_time).total_seconds()
        return duration >= config.logging.clip_duration
    
    def cleanup_old_files(self):
        """Clean up old files to manage storage space."""
        # Implementation for cleaning old files based on storage limits
        pass
    
    # 🆕 NEW: Attendance data cleanup
    def cleanup_old_attendance_data(self):
        """Clean up old attendance logs and data."""
        if not config.attendance.auto_cleanup:
            return
        
        try:
            cutoff_date = datetime.now().timestamp() - (config.attendance.retention_days * 86400)
            
            # Clean up old attendance events
            self.attendance_events = [
                event for event in self.attendance_events
                if datetime.fromisoformat(event['timestamp']).timestamp() >= cutoff_date
            ]
            
            self.info(f"Cleaned up attendance data older than {config.attendance.retention_days} days")
            
        except Exception as e:
            self.error(f"Failed to cleanup attendance data: {e}")

# Global logger instance
logger = SurveillanceLogger()

# 🆕 NEW: Attendance-specific logger functions
def log_attendance_detection(employee_id: str, visit_count: int, confidence: float):
    """Log attendance detection."""
    logger.log_attendance_detection(employee_id, visit_count, confidence)

def log_attendance_event(employee_id: str, event_type: str, details: dict = None):
    """Log attendance event."""
    logger.log_attendance_event(employee_id, event_type, details)

def get_attendance_summary(hours: int = 24):
    """Get attendance summary."""
    return logger.get_attendance_summary(hours)

def save_attendance_image(frame, employee_id: str, visit_count: int, confidence: float):
    """Save attendance image."""
    return logger.save_attendance_image(frame, employee_id, visit_count, confidence)