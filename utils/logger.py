"""
Enhanced logging utilities for the surveillance system.
Includes attendance module logging and comprehensive error handling.
"""
import logging
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Any, Dict, List
from pathlib import Path
import threading
import queue
import json

class SurveillanceLogger:
    """Enhanced logger for surveillance system events and media with proper error handling."""
    
    def __init__(self, name: str = "surveillance"):
        self.logger = logging.getLogger(name)
        self._setup_logger()
        
        # Video recording
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording_start_time: Optional[datetime] = None
        self._recording_lock = threading.Lock()
        
        # Attendance-specific logging
        self.attendance_logger = self._setup_attendance_logger()
        self.attendance_events = []
        self.max_attendance_events = 1000
        self._events_lock = threading.Lock()
        
        # Async logging queue for performance
        self.log_queue = queue.Queue(maxsize=1000)
        self.log_worker_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_worker_thread.start()
        
        self.logger.info("Surveillance logger initialized successfully")
    
    def _setup_logger(self):
        """Setup logging handlers with proper error handling."""
        try:
            # Import config here to avoid circular imports
            try:
                from utils.config import config
                log_level = getattr(logging, config.logging.log_level.upper(), logging.INFO)
                output_dir = config.logging.output_dir
            except ImportError:
                log_level = logging.INFO
                output_dir = "surveillance_output"
            
            # Clear existing handlers to avoid duplicates
            self.logger.handlers.clear()
            
            # Set log level
            self.logger.setLevel(log_level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            self.logger.addHandler(console_handler)
            
            # File handler
            try:
                log_dir = Path(output_dir) / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                
                log_file = log_dir / "surveillance.log"
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(formatter)
                file_handler.setLevel(log_level)
                self.logger.addHandler(file_handler)
                
            except Exception as e:
                self.logger.warning(f"Could not setup file logging: {e}")
            
            # Prevent propagation to root logger
            self.logger.propagate = False
            
        except Exception as e:
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(name)
            self.logger.error(f"Failed to setup enhanced logging: {e}")
    
    def _setup_attendance_logger(self):
        """Setup separate logger for attendance events."""
        try:
            try:
                from utils.config import config
                if not config.attendance.enabled or not config.attendance.enable_logging:
                    return None
                log_file = config.attendance.log_file
                output_dir = config.logging.output_dir
            except ImportError:
                log_file = "logs/attendance.log"
                output_dir = "surveillance_output"
            
            attendance_logger = logging.getLogger("attendance")
            attendance_logger.handlers.clear()
            attendance_logger.setLevel(logging.INFO)
            attendance_logger.propagate = False
            
            formatter = logging.Formatter(
                '%(asctime)s - ATTENDANCE - %(levelname)s - %(message)s'
            )
            
            # Attendance file handler
            try:
                log_file_path = Path(log_file)
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                attendance_file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
                attendance_file_handler.setFormatter(formatter)
                attendance_logger.addHandler(attendance_file_handler)
                
            except Exception as e:
                self.logger.warning(f"Could not setup attendance file logging: {e}")
            
            # Also log to main surveillance log
            try:
                main_log_dir = Path(output_dir) / "logs"
                main_log_file = main_log_dir / "surveillance.log"
                
                main_handler = logging.FileHandler(main_log_file, encoding='utf-8')
                main_handler.setFormatter(formatter)
                attendance_logger.addHandler(main_handler)
                
            except Exception as e:
                self.logger.warning(f"Could not setup attendance main logging: {e}")
            
            self.logger.info("Attendance logging initialized")
            return attendance_logger
            
        except Exception as e:
            self.logger.error(f"Failed to setup attendance logging: {e}")
            return None
    
    def _log_worker(self):
        """Background worker for async logging."""
        while True:
            try:
                log_entry = self.log_queue.get(timeout=1.0)
                if log_entry is None:  # Shutdown signal
                    break
                
                level, message, kwargs = log_entry
                getattr(self.logger, level)(message, **kwargs)
                
            except queue.Empty:
                continue
            except Exception as e:
                # Use print to avoid infinite loop
                print(f"Log worker error: {e}")
    
    def _async_log(self, level: str, message: str, **kwargs):
        """Add log entry to async queue."""
        try:
            self.log_queue.put_nowait((level, message, kwargs))
        except queue.Full:
            # If queue is full, log synchronously
            getattr(self.logger, level)(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._async_log('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._async_log('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._async_log('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        # Error messages are logged synchronously for immediate visibility
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        # Critical messages are logged synchronously
        self.logger.critical(message, **kwargs)
    
    def log_event(self, event_type: str, details: dict):
        """Log a surveillance event with structured details."""
        try:
            timestamp = datetime.now().isoformat()
            message = f"EVENT: {event_type} | {json.dumps(details)} | {timestamp}"
            self.info(message)
        except Exception as e:
            self.error(f"Failed to log event: {e}")
    
    def log_attendance_event(self, employee_id: str, event_type: str = "DETECTED", 
                           details: Dict = None, confidence: float = 0.0):
        """Log an attendance-related event with error handling."""
        try:
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
            
            # Thread-safe addition to events list
            with self._events_lock:
                self.attendance_events.append(attendance_event)
                if len(self.attendance_events) > self.max_attendance_events:
                    self.attendance_events = self.attendance_events[-self.max_attendance_events:]
            
            # Log the event
            message = f"Employee: {employee_id} | Event: {event_type} | Confidence: {confidence:.2f}"
            if event_details:
                message += f" | Details: {json.dumps(event_details)}"
            
            self.attendance_logger.info(message)
            
            # Also log significant events to main log
            if event_type in ["FIRST_DETECTION", "NEW_EMPLOYEE", "ATTENDANCE_LOGGED"]:
                self.info(f"ATTENDANCE - {message}")
                
        except Exception as e:
            self.error(f"Failed to log attendance event: {e}")
    
    def log_attendance_detection(self, employee_id: str, visit_count: int, 
                               confidence: float, location: tuple = None):
        """Log attendance detection event."""
        try:
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
        except Exception as e:
            self.error(f"Failed to log attendance detection: {e}")
    
    def log_face_recognition_stats(self, total_faces: int, known_faces: int, 
                                 unknown_faces: int, processing_time: float):
        """Log face recognition statistics."""
        try:
            if not self.attendance_logger:
                return
            
            stats_message = (f"Face Recognition Stats - Total: {total_faces}, "
                            f"Known: {known_faces}, Unknown: {unknown_faces}, "
                            f"Processing Time: {processing_time:.3f}s")
            
            self.attendance_logger.debug(stats_message)
        except Exception as e:
            self.error(f"Failed to log face recognition stats: {e}")
    
    def log_system_performance(self, fps: float, memory_usage: float, 
                             gpu_usage: float = None):
        """Log system performance metrics."""
        try:
            perf_details = {
                'fps': fps,
                'memory_usage_mb': memory_usage
            }
            if gpu_usage is not None:
                perf_details['gpu_usage_percent'] = gpu_usage
            
            self.log_attendance_event("SYSTEM", "PERFORMANCE", perf_details)
        except Exception as e:
            self.error(f"Failed to log system performance: {e}")
    
    def get_recent_attendance_events(self, hours: int = 24) -> List[Dict]:
        """Get recent attendance events within specified hours."""
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            
            with self._events_lock:
                recent_events = []
                for event in self.attendance_events:
                    event_time = datetime.fromisoformat(event['timestamp']).timestamp()
                    if event_time >= cutoff_time:
                        recent_events.append(event.copy())
            
            return recent_events
        except Exception as e:
            self.error(f"Failed to get recent attendance events: {e}")
            return []
    
    def get_attendance_summary(self, hours: int = 24) -> Dict:
        """Get attendance summary for specified time period."""
        try:
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
        except Exception as e:
            self.error(f"Failed to get attendance summary: {e}")
            return {
                'time_period_hours': hours,
                'total_events': 0,
                'unique_employees': 0,
                'employee_attendance_counts': {},
                'event_type_counts': {},
                'last_updated': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def save_image(self, frame: np.ndarray, event_type: str, metadata: dict = None) -> str:
        """Save an image with timestamp and metadata."""
        try:
            try:
                from utils.config import config
                if not config.logging.save_images:
                    return ""
                output_dir = config.logging.output_dir
            except ImportError:
                output_dir = "surveillance_output"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{event_type}_{timestamp}.jpg"
            
            # Ensure images directory exists
            images_dir = Path(output_dir) / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = images_dir / filename
            
            # Validate frame
            if frame is None or frame.size == 0:
                self.warning(f"Invalid frame for image save: {filename}")
                return ""
            
            # Save image
            success = cv2.imwrite(str(filepath), frame)
            if success:
                self.info(f"Image saved: {filename}")
                
                if metadata:
                    self.log_event("IMAGE_SAVED", {
                        "filename": filename,
                        "event_type": event_type,
                        **metadata
                    })
                
                return str(filepath)
            else:
                self.error(f"Failed to save image: {filename}")
                return ""
                
        except Exception as e:
            self.error(f"Error saving image: {e}")
            return ""
    
    def save_attendance_image(self, frame: np.ndarray, employee_id: str, 
                            visit_count: int, confidence: float) -> str:
        """Save image for attendance event."""
        try:
            metadata = {
                'employee_id': employee_id,
                'visit_count': visit_count,
                'confidence': confidence,
                'event_category': 'attendance'
            }
            
            filename_base = f"attendance_{employee_id}_{visit_count}"
            filepath = self.save_image(frame, filename_base, metadata)
            
            if filepath:
                self.log_attendance_event(employee_id, "IMAGE_SAVED", {
                    'filepath': filepath,
                    'visit_count': visit_count
                })
            
            return filepath
            
        except Exception as e:
            self.error(f"Error saving attendance image: {e}")
            return ""
    
    def start_recording(self, frame: np.ndarray, event_type: str) -> bool:
        """Start video recording for an event with proper error handling."""
        try:
            try:
                from utils.config import config
                if not config.logging.save_clips:
                    return False
                output_dir = config.logging.output_dir
                fps = config.camera.fps
            except ImportError:
                return False
            
            with self._recording_lock:
                if self.video_writer is not None:
                    self.warning("Recording already in progress")
                    return False
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{event_type}_{timestamp}.mp4"
                
                # Ensure clips directory exists
                clips_dir = Path(output_dir) / "clips"
                clips_dir.mkdir(parents=True, exist_ok=True)
                
                filepath = clips_dir / filename
                
                # Validate frame
                if frame is None or frame.size == 0:
                    self.error("Invalid frame for video recording")
                    return False
                
                height, width = frame.shape[:2]
                if height <= 0 or width <= 0:
                    self.error(f"Invalid frame dimensions: {width}x{height}")
                    return False
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    str(filepath), fourcc, fps, (width, height)
                )
                
                if self.video_writer.isOpened():
                    self.recording_start_time = datetime.now()
                    self.info(f"Started recording: {filename}")
                    self.log_event("RECORDING_STARTED", {
                        "filename": filename,
                        "event_type": event_type,
                        "resolution": f"{width}x{height}"
                    })
                    return True
                else:
                    self.error(f"Failed to open video writer for {filename}")
                    self.video_writer = None
                    return False
                    
        except Exception as e:
            self.error(f"Error starting recording: {e}")
            with self._recording_lock:
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
            return False
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """Write a frame to the current video recording."""
        try:
            with self._recording_lock:
                if self.video_writer is None or not self.video_writer.isOpened():
                    return False
                
                if frame is None or frame.size == 0:
                    self.warning("Invalid frame for video writing")
                    return False
                
                self.video_writer.write(frame)
                return True
                
        except Exception as e:
            self.error(f"Error writing video frame: {e}")
            return False
    
    def stop_recording(self) -> bool:
        """Stop the current video recording."""
        try:
            with self._recording_lock:
                if self.video_writer is None:
                    return False
                
                duration = 0
                if self.recording_start_time:
                    duration = (datetime.now() - self.recording_start_time).total_seconds()
                
                self.video_writer.release()
                self.video_writer = None
                self.recording_start_time = None
                
                self.info(f"Stopped recording (duration: {duration:.2f}s)")
                self.log_event("RECORDING_STOPPED", {"duration": duration})
                return True
                
        except Exception as e:
            self.error(f"Error stopping recording: {e}")
            return False
    
    def should_stop_recording(self) -> bool:
        """Check if recording should be stopped based on duration."""
        try:
            try:
                from utils.config import config
                clip_duration = config.logging.clip_duration
            except ImportError:
                clip_duration = 10
            
            if self.recording_start_time is None:
                return False
            
            duration = (datetime.now() - self.recording_start_time).total_seconds()
            return duration >= clip_duration
            
        except Exception as e:
            self.error(f"Error checking recording duration: {e}")
            return True  # Stop recording on error
    
    def cleanup_old_files(self, days_to_keep: int = 30):
        """Clean up old files to manage storage space."""
        try:
            try:
                from utils.config import config
                output_dir = config.logging.output_dir
            except ImportError:
                output_dir = "surveillance_output"
            
            import time
            
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            
            # Clean up old images
            images_dir = Path(output_dir) / "images"
            if images_dir.exists():
                for file_path in images_dir.glob("*.jpg"):
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        self.debug(f"Deleted old image: {file_path}")
            
            # Clean up old videos
            clips_dir = Path(output_dir) / "clips"
            if clips_dir.exists():
                for file_path in clips_dir.glob("*.mp4"):
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        self.debug(f"Deleted old video: {file_path}")
            
            self.info(f"Cleaned up files older than {days_to_keep} days")
            
        except Exception as e:
            self.error(f"Error cleaning up old files: {e}")
    
    def cleanup_old_attendance_data(self):
        """Clean up old attendance logs and data."""
        try:
            try:
                from utils.config import config
                if not config.attendance.auto_cleanup:
                    return
                retention_days = config.attendance.retention_days
            except ImportError:
                retention_days = 90
            
            cutoff_date = datetime.now().timestamp() - (retention_days * 86400)
            
            # Clean up old attendance events
            with self._events_lock:
                original_count = len(self.attendance_events)
                self.attendance_events = [
                    event for event in self.attendance_events
                    if datetime.fromisoformat(event['timestamp']).timestamp() >= cutoff_date
                ]
                cleaned_count = original_count - len(self.attendance_events)
            
            if cleaned_count > 0:
                self.info(f"Cleaned up {cleaned_count} old attendance events")
            
        except Exception as e:
            self.error(f"Error cleaning up attendance data: {e}")
    
    def get_log_statistics(self) -> Dict:
        """Get logging system statistics."""
        try:
            with self._events_lock:
                attendance_events_count = len(self.attendance_events)
            
            return {
                'attendance_events_count': attendance_events_count,
                'log_queue_size': self.log_queue.qsize(),
                'is_recording': self.video_writer is not None,
                'recording_duration': (
                    (datetime.now() - self.recording_start_time).total_seconds()
                    if self.recording_start_time else 0
                ),
                'attendance_logging_enabled': self.attendance_logger is not None
            }
        except Exception as e:
            self.error(f"Error getting log statistics: {e}")
            return {}
    
    def shutdown(self):
        """Graceful shutdown of logging system."""
        try:
            self.info("Shutting down surveillance logger")
            
            # Stop any active recording
            if self.video_writer is not None:
                self.stop_recording()
            
            # Signal log worker to stop
            self.log_queue.put(None)
            
            # Wait for log worker to finish
            if self.log_worker_thread.is_alive():
                self.log_worker_thread.join(timeout=5.0)
            
            # Final cleanup
            self.cleanup_old_attendance_data()
            
            self.info("Surveillance logger shutdown complete")
            
        except Exception as e:
            print(f"Error during logger shutdown: {e}")

# Global logger instance with error handling
try:
    logger = SurveillanceLogger()
except Exception as e:
    # Fallback to basic logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("surveillance_fallback")
    logger.error(f"Failed to initialize surveillance logger: {e}")

# Attendance-specific logger functions
def log_attendance_detection(employee_id: str, visit_count: int, confidence: float):
    """Log attendance detection."""
    try:
        logger.log_attendance_detection(employee_id, visit_count, confidence)
    except Exception as e:
        print(f"Error logging attendance detection: {e}")

def log_attendance_event(employee_id: str, event_type: str, details: dict = None):
    """Log attendance event."""
    try:
        logger.log_attendance_event(employee_id, event_type, details)
    except Exception as e:
        print(f"Error logging attendance event: {e}")

def get_attendance_summary(hours: int = 24):
    """Get attendance summary."""
    try:
        return logger.get_attendance_summary(hours)
    except Exception as e:
        print(f"Error getting attendance summary: {e}")
        return {}

def save_attendance_image(frame, employee_id: str, visit_count: int, confidence: float):
    """Save attendance image."""
    try:
        return logger.save_attendance_image(frame, employee_id, visit_count, confidence)
    except Exception as e:
        print(f"Error saving attendance image: {e}")
        return ""