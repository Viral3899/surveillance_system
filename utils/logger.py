"""
Logging utilities for the surveillance system.
"""
import logging
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Any
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

# Global logger instance
logger = SurveillanceLogger()