#!/usr/bin/env python3
"""
Main surveillance system controller.
Orchestrates all components for real-time AI-powered surveillance.
"""
import cv2
import numpy as np
import time
import signal
import sys
import threading
from datetime import datetime
from typing import Optional, Dict, Any

# Import surveillance modules

from camera.stream_handler import CameraStream, MultiCameraManager
from detection.motion_detector import MotionDetector, MotionDetectionMethod
from face_recognition_s.face_detector import FaceDetector
from face_recognition_s.face_matcher import FaceMatcher
from anomaly.anomaly_detector import AnomalyDetector
from attendance.storage import AttendanceRepository
from utils.config import config
from utils.logger import logger

# GPU memory management with optional torch dependency
try:
    import torch  # type: ignore

    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
except ImportError:
    torch = None  # type: ignore
    print("PyTorch not available, running in CPU-only mode")
    TORCH_AVAILABLE = False

class SurveillanceSystem:
    """Main surveillance system controller."""
    
    def __init__(
        self,
        camera_id: int = 0,
        gui_mode: bool = True,
        enable_signal_handlers: bool = True,
    ):
        self.camera_id = camera_id
        self.gui_mode = gui_mode
        self.enable_signal_handlers = enable_signal_handlers
        self.running = False
        
        # Initialize components
        self.camera_stream: Optional[CameraStream] = None
        self.motion_detector: Optional[MotionDetector] = None
        self.face_detector: Optional[FaceDetector] = None
        self.face_matcher: Optional[FaceMatcher] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        
        # Attendance tracking
        self.attendance_repo: Optional[AttendanceRepository] = None
        self.attendance_stats = {
            'total_detections': 0,
            'known_faces': 0,
            'unknown_faces': 0,
            'today_attendance': {},
            'last_detection_time': None
        }
        # Track last visit time per employee for cooldown
        self.last_visit_times = {}  # {employee_id: datetime}
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_display = 0
        self.last_fps_update = time.time()
        
        # GPU optimization
        self.use_cuda = TORCH_AVAILABLE and config.gpu.use_cuda and torch.cuda.is_available() if TORCH_AVAILABLE else False
        self.gpu_cleanup_counter = 0
        
        # Initialize system
        self._initialize_system()
        
        # Set up signal handlers for graceful shutdown (optional)
        if self.enable_signal_handlers:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_system(self):
        """Initialize all surveillance components."""
        logger.info("Initializing surveillance system.")
        
        try:
            # Initialize camera
            self.camera_stream = CameraStream(self.camera_id)
            logger.info("Camera stream initialized")
            
            # Initialize motion detector
            detection_method = MotionDetectionMethod.BACKGROUND_SUBTRACTION
            if hasattr(config.detection, 'method'):
                detection_method = getattr(MotionDetectionMethod, config.detection.method.upper())
            self.motion_detector = MotionDetector(detection_method)
            logger.info("Motion detector initialized")
            
            # Initialize face components
            self.face_detector = FaceDetector()
            self.face_matcher = FaceMatcher()
            logger.info("Face recognition system initialized")
            
            # Initialize anomaly detector
            self.anomaly_detector = AnomalyDetector()
            logger.info("Anomaly detector initialized")
            
            # Initialize attendance repository if enabled
            if config.attendance.enabled:
                if config.attendance.database_type == "mysql":
                    mysql_config = {
                        "host": config.attendance.mysql_host,
                        "port": config.attendance.mysql_port,
                        "user": config.attendance.mysql_user,
                        "password": config.attendance.mysql_password,
                        "database": config.attendance.mysql_database,
                    }
                    self.attendance_repo = AttendanceRepository(
                        database_type="mysql",
                        mysql_config=mysql_config
                    )
                    logger.info(f"Attendance tracking initialized with MySQL: {config.attendance.mysql_database}")
                else:
                    self.attendance_repo = AttendanceRepository(
                        database_path=config.attendance.database_file,
                        database_type="sqlite"
                    )
                    logger.info(f"Attendance tracking initialized with SQLite: {config.attendance.database_file}")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize surveillance system: {e}")
            raise
    
    def start(self):
        """Start the surveillance system."""
        if self.running:
            logger.warning("System is already running")
            return
        
        logger.info("Starting surveillance system.")
        
        try:
            # Start camera stream
            if not self.camera_stream.start_stream():
                raise RuntimeError("Failed to start camera stream")
            
            self.running = True
            self.start_time = time.time()
            
            if self.gui_mode:
                self._run_with_gui()
            else:
                self._run_headless()
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error during surveillance: {e}")
        finally:
            self.stop()
    
    def _run_with_gui(self):
        """Run surveillance with GUI display."""
        logger.info("Starting surveillance with GUI.")
        
        # Create display windows
        cv2.namedWindow('Surveillance Feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                success = self._process_frame()
                if not success:
                    break
                
                # Handle GUI events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit key pressed")
                    break
                elif key == ord('r'):
                    logger.info("Resetting detectors.")
                    self._reset_detectors()
                elif key == ord('l'):
                    logger.info("Toggling learning mode.")
                    self.anomaly_detector.set_learning_mode(not self.anomaly_detector.is_learning)
                elif key == ord('s'):
                    logger.info("Saving current frame.")
                    self._save_current_frame()
        finally:
            cv2.destroyAllWindows()
    
    def _run_headless(self):
        """Run surveillance without GUI."""
        logger.info("Starting surveillance in headless mode.")
        
        while self.running:
            success = self._process_frame()
            if not success:
                break
            
            # Print status every 100 frames
            if self.frame_count % 100 == 0:
                self._print_status()
    
    def _process_frame(self) -> bool:
        """Process a single frame through the surveillance pipeline."""
        try:
            # Reset daily attendance stats if needed
            self._check_daily_reset()
            
            # Get frame from camera
            frame = self.camera_stream.get_frame()
            if frame is None:
                logger.warning("No frame received from camera")
                return False
            
            self.frame_count += 1
            
            # Motion detection
            motion_events = self.motion_detector.detect_motion(frame)
            
            # Face detection (only if motion detected or periodically)
            face_detections = []
            if motion_events or self.frame_count % 10 == 0:  # Face detection every 10 frames
                face_detections = self.face_detector.detect_faces(frame, return_encodings=True)
                
                # Face matching
                if face_detections:
                    face_detections = self.face_matcher.match_faces(face_detections)
                    
                    # Update attendance stats and log attendance
                    self._process_attendance(face_detections)
            
            # Anomaly detection
            anomalies = self.anomaly_detector.detect_anomalies(
                motion_events, face_detections, frame
            )
            
            # Handle detected anomalies
            if anomalies:
                self._handle_anomalies(frame, anomalies, motion_events, face_detections)
            
            # Display results if GUI mode
            if self.gui_mode:
                self._display_results(frame, motion_events, face_detections, anomalies)
            
            # Update FPS
            self._update_fps()
            
            # GPU memory cleanup
            self._cleanup_gpu_memory()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return False
    
    def _handle_anomalies(self, frame, anomalies, motion_events, face_detections):
        """Handle detected anomalies (logging, alerts, recording)."""
        for anomaly in anomalies:
            # Log anomaly event
            logger.log_event("ANOMALY_DETECTED", {
                'type': anomaly.anomaly_type.value,
                'confidence': anomaly.confidence,
                'location': anomaly.location,
                'description': anomaly.description
            })
            
            # Save image for high-confidence anomalies
            if anomaly.confidence > 0.7:
                logger.save_image(frame, f"anomaly_{anomaly.anomaly_type.value}", {
                    'confidence': anomaly.confidence,
                    'location': anomaly.location
                })
            
            # Start recording for severe anomalies
            if anomaly.confidence > 0.8 and not logger.video_writer:
                logger.start_recording(frame, f"anomaly_{anomaly.anomaly_type.value}")
        
        # Continue recording if already started
        if logger.video_writer:
            logger.write_frame(frame)
            
            # Stop recording after duration or if no more high-confidence anomalies
            if logger.should_stop_recording() or not any(a.confidence > 0.8 for a in anomalies):
                logger.stop_recording()
    
    def _display_results(self, frame, motion_events, face_detections, anomalies):
        """Display all surveillance results in a unified single screen."""
        # Create unified display frame
        display_frame = frame.copy()
        
        # Draw motion detection overlay
        if motion_events:
            display_frame = self.motion_detector.draw_motion_overlay(display_frame, motion_events)
        
        # Draw face detection and recognition overlay
        if face_detections:
            display_frame = self._draw_face_attendance_overlay(display_frame, face_detections)
        
        # Draw anomaly overlay
        if anomalies:
            display_frame = self.anomaly_detector.draw_anomaly_overlay(display_frame, anomalies)
        
        # Add comprehensive status overlay with all information
        display_frame = self._add_comprehensive_overlay(display_frame, motion_events, face_detections, anomalies)
        
        # Display unified frame
        cv2.imshow('AI Surveillance System - All Features', display_frame)
    
    def _process_attendance(self, face_detections):
        """Process attendance for detected faces with 30-minute cooldown."""
        if not config.attendance.enabled or not self.attendance_repo:
            return
        
        current_time = datetime.now()
        cooldown_seconds = config.attendance.cooldown_seconds  # 30 minutes (1800 seconds)
        
        for detection in face_detections:
            face_id = getattr(detection, 'face_id', None) or getattr(detection, 'id', 'Unknown')
            face_name = getattr(detection, 'name', face_id)
            confidence = getattr(detection, 'confidence', 0.0)
            
            if face_id != "Unknown" and confidence > 0.6:
                # Check if enough time has passed since last visit (30-minute cooldown)
                should_log_visit = True
                time_since_last_visit = None
                
                if face_id in self.last_visit_times:
                    last_visit_time = self.last_visit_times[face_id]
                    time_since_last_visit = (current_time - last_visit_time).total_seconds()
                    
                    # Only log if 30 minutes (1800 seconds) have passed
                    if time_since_last_visit < cooldown_seconds:
                        should_log_visit = False
                        logger.debug(
                            f"Cooldown active for {face_name}: "
                            f"{int(time_since_last_visit)}s / {cooldown_seconds}s"
                        )
                
                # Update last seen time (always update for display)
                if face_id not in self.attendance_stats['today_attendance']:
                    self.attendance_stats['today_attendance'][face_id] = {
                        'name': face_name,
                        'count': 0,
                        'last_seen': current_time,
                        'last_visit': None
                    }
                
                self.attendance_stats['today_attendance'][face_id]['last_seen'] = current_time
                self.attendance_stats['known_faces'] += 1
                
                # Only log attendance if cooldown period has passed
                if should_log_visit:
                    # Record attendance as new visit
                    record = {
                        "Employee_ID": face_id,
                        "Employee_Name": face_name,
                        "Timestamp": current_time.isoformat(),
                        "Visit_Type": "IN",
                        "Visit_Count": 1,
                        "Confidence": confidence
                    }
                    
                    try:
                        self.attendance_repo.record_attendance(record)
                        self.attendance_stats['total_detections'] += 1
                        self.attendance_stats['last_detection_time'] = current_time
                        
                        # Update visit count and last visit time
                        self.attendance_stats['today_attendance'][face_id]['count'] += 1
                        self.attendance_stats['today_attendance'][face_id]['last_visit'] = current_time
                        self.last_visit_times[face_id] = current_time
                        
                        logger.info(
                            f"New visit logged for {face_name} "
                            f"(Visit #{self.attendance_stats['today_attendance'][face_id]['count']})"
                        )
                        
                    except Exception as e:
                        logger.error(f"Error recording attendance: {e}")
                else:
                    # Still update last seen for display, but don't log as new visit
                    remaining_cooldown = int(cooldown_seconds - time_since_last_visit)
                    logger.debug(
                        f"{face_name} detected but in cooldown "
                        f"({remaining_cooldown}s remaining)"
                    )
            else:
                self.attendance_stats['unknown_faces'] += 1
    
    def _draw_face_attendance_overlay(self, frame, face_detections):
        """Draw face detection with attendance information."""
        for detection in face_detections:
            bbox = getattr(detection, 'bbox', None) or getattr(detection, 'location', None)
            if not bbox:
                continue
            
            face_id = getattr(detection, 'face_id', None) or getattr(detection, 'id', 'Unknown')
            face_name = getattr(detection, 'name', face_id)
            confidence = getattr(detection, 'confidence', 0.0)
            
            # Draw bounding box
            if face_id != "Unknown":
                color = (0, 255, 0)  # Green for known faces
                thickness = 2
            else:
                color = (0, 0, 255)  # Red for unknown faces
                thickness = 1
            
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with attendance info
            label = f"{face_name}"
            if face_id != "Unknown":
                visit_count = self.attendance_stats['today_attendance'].get(face_id, {}).get('count', 0)
                last_visit = self.attendance_stats['today_attendance'].get(face_id, {}).get('last_visit')
                
                # Check if in cooldown period
                in_cooldown = False
                if face_id in self.last_visit_times and last_visit:
                    time_since_visit = (datetime.now() - last_visit).total_seconds()
                    if time_since_visit < config.attendance.cooldown_seconds:
                        in_cooldown = True
                        remaining = int(config.attendance.cooldown_seconds - time_since_visit)
                        minutes = remaining // 60
                        seconds = remaining % 60
                        label += f" (Visits: {visit_count}, Wait: {minutes}m {seconds}s)"
                    else:
                        label += f" (Visits: {visit_count})"
                else:
                    label += f" (Visits: {visit_count})"
            label += f" {confidence:.2f}"
            
            # Background for text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
        
        return frame
    
    def _add_comprehensive_overlay(self, frame, motion_events, face_detections, anomalies):
        """Add comprehensive status overlay showing all system information."""
        height, width = frame.shape[:2]
        
        # Top-left: System Status
        status_lines = [
            "=== SYSTEM STATUS ===",
            f"FPS: {self.fps_display:.1f}",
            f"Frames: {self.frame_count}",
            f"Runtime: {self._get_runtime_str()}",
            f"GPU: {'ON' if self.use_cuda else 'OFF'}",
            f"Learning: {'ON' if self.anomaly_detector.is_learning else 'OFF'}",
            "",
            "=== DETECTION STATS ===",
            f"Motion Events: {len(motion_events)}",
            f"Faces Detected: {len(face_detections)}",
            f"Anomalies: {len(anomalies)}",
        ]
        
        # Top-right: Attendance Statistics
        attendance_lines = [
            "=== ATTENDANCE TRACKING ===",
            f"Total Detections: {self.attendance_stats['total_detections']}",
            f"Known Faces: {self.attendance_stats['known_faces']}",
            f"Unknown Faces: {self.attendance_stats['unknown_faces']}",
            f"Today's Visitors: {len(self.attendance_stats['today_attendance'])}",
            "",
            "=== TODAY'S ATTENDANCE ===",
        ]
        
        # Add top 5 recent visitors
        sorted_attendance = sorted(
            self.attendance_stats['today_attendance'].items(),
            key=lambda x: x[1]['last_seen'],
            reverse=True
        )[:5]
        
        for i, (emp_id, data) in enumerate(sorted_attendance):
            name = data['name'][:15] if len(data['name']) > 15 else data['name']
            attendance_lines.append(f"{i+1}. {name}: {data['count']} visits")
        
        # Bottom-left: Controls
        help_lines = [
            "=== CONTROLS ===",
            "Q - Quit System",
            "R - Reset Detectors",
            "L - Toggle Learning",
            "S - Save Frame",
            "A - Show Attendance",
        ]
        
        # Draw System Status (Top-left)
        status_y_start = 10
        status_x = 10
        status_height = len(status_lines) * 22 + 20
        cv2.rectangle(frame, (status_x, status_y_start), 
                     (status_x + 300, status_y_start + status_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (status_x, status_y_start), 
                     (status_x + 300, status_y_start + status_height), (0, 255, 0), 2)
        
        for i, line in enumerate(status_lines):
            y_pos = status_y_start + 25 + i * 22
            color = (0, 255, 0) if line.startswith("===") else (255, 255, 255)
            thickness = 1 if not line.startswith("===") else 1
            cv2.putText(frame, line, (status_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA)
        
        # Draw Attendance Stats (Top-right)
        att_x = width - 350
        att_y_start = 10
        att_height = len(attendance_lines) * 22 + 20
        cv2.rectangle(frame, (att_x, att_y_start), 
                     (att_x + 340, att_y_start + att_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (att_x, att_y_start), 
                     (att_x + 340, att_y_start + att_height), (255, 165, 0), 2)
        
        for i, line in enumerate(attendance_lines):
            y_pos = att_y_start + 25 + i * 22
            color = (255, 165, 0) if line.startswith("===") else (255, 255, 255)
            cv2.putText(frame, line, (att_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # Draw Controls (Bottom-left)
        help_y_start = height - len(help_lines) * 22 - 30
        help_x = 10
        help_height = len(help_lines) * 22 + 20
        cv2.rectangle(frame, (help_x, help_y_start), 
                     (help_x + 220, help_y_start + help_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (help_x, help_y_start), 
                     (help_x + 220, help_y_start + help_height), (255, 255, 255), 1)
        
        for i, line in enumerate(help_lines):
            y_pos = help_y_start + 25 + i * 22
            color = (255, 255, 0) if line.startswith("===") else (255, 255, 255)
            cv2.putText(frame, line, (help_x + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # Draw timestamp (Bottom-right)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (ts_width, ts_height), _ = cv2.getTextSize(
            timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        ts_x = width - ts_width - 20
        ts_y = height - 10
        cv2.rectangle(frame, (ts_x - 5, ts_y - ts_height - 5), 
                     (ts_x + ts_width + 5, ts_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, timestamp, (ts_x, ts_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        return frame
    
    def _add_status_overlay(self, frame):
        """Legacy method - redirects to comprehensive overlay."""
        return self._add_comprehensive_overlay(frame, [], [], [])
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            elapsed = current_time - self.start_time
            self.fps_display = self.frame_count / elapsed if elapsed > 0 else 0
            self.last_fps_update = current_time
    
    def _cleanup_gpu_memory(self):
        """Periodic GPU memory cleanup."""
        self.gpu_cleanup_counter += 1
        if (TORCH_AVAILABLE and self.use_cuda and torch.cuda.is_available() and 
            self.gpu_cleanup_counter % config.gpu.cache_cleanup_interval == 0):
            torch.cuda.empty_cache()
            logger.debug(f"GPU memory cleaned up at frame {self.frame_count}")
    
    def _reset_detectors(self):
        """Reset all detectors."""
        if self.motion_detector:
            self.motion_detector.reset_detector()
        if self.anomaly_detector:
            self.anomaly_detector.reset_learning()
    
    def _save_current_frame(self):
        """Save the current frame."""
        frame = self.camera_stream.get_frame()
        if frame is not None:
            logger.save_image(frame, "manual_save")
    
    def _check_daily_reset(self):
        """Reset daily attendance statistics if it's a new day."""
        if not config.attendance.enabled:
            return
        
        current_date = datetime.now().date()
        if not hasattr(self, '_last_reset_date') or self._last_reset_date != current_date:
            self.attendance_stats['today_attendance'] = {}
            self.attendance_stats['total_detections'] = 0
            self.last_visit_times = {}  # Reset visit times for new day
            self._last_reset_date = current_date
            logger.info(f"Daily attendance stats reset for {current_date}")
    
    def _get_runtime_str(self) -> str:
        """Get formatted runtime string."""
        runtime = time.time() - self.start_time
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _print_status(self):
        """Print status information (headless mode)."""
        runtime = self._get_runtime_str()
        current_fps = self.frame_count / (time.time() - self.start_time)
        
        # Get component statistics
        motion_stats = self.motion_detector.get_motion_statistics()
        face_stats = self.face_detector.get_detection_statistics()
        anomaly_stats = self.anomaly_detector.get_anomaly_statistics()
        
        print(f"\n--- Surveillance Status ---")
        print(f"Runtime: {runtime} | Frames: {self.frame_count} | FPS: {current_fps:.1f}")
        print(f"Motion: {motion_stats.get('average_events_per_frame', 0):.2f} events/frame")
        print(f"Faces: {face_stats.get('total_detections', 0)} total detections")
        print(f"Anomalies: {anomaly_stats.get('total_anomalies_detected', 0)} total")
        print(f"Learning: {'ON' if self.anomaly_detector.is_learning else 'OFF'}")
        print(f"GPU: {'ON' if self.use_cuda else 'OFF'}")
        print("-" * 50)
    
    def stop(self):
        """Stop the surveillance system."""
        if not self.running:
            return
        
        logger.info("Stopping surveillance system.")
        self.running = False
        
        # Stop recording if active
        if logger.video_writer:
            logger.stop_recording()
        
        # Stop camera stream
        if self.camera_stream:
            self.camera_stream.stop_stream()
        
        # Save final statistics
        self._save_final_statistics()
        
        # GPU cleanup
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Surveillance system stopped")
    
    def _save_final_statistics(self):
        """Save final run statistics."""
        try:
            runtime = time.time() - self.start_time
            avg_fps = self.frame_count / runtime if runtime > 0 else 0
            
            stats = {
                'session_info': {
                    'start_time': time.ctime(self.start_time),
                    'end_time': time.ctime(),
                    'runtime_seconds': runtime,
                    'total_frames': self.frame_count,
                    'average_fps': avg_fps
                },
                'motion_detection': self.motion_detector.get_motion_statistics(),
                'face_detection': self.face_detector.get_detection_statistics(),
                'face_recognition': self.face_matcher.get_recognition_statistics(),
                'anomaly_detection': self.anomaly_detector.get_anomaly_statistics()
            }
            
            # Save to log
            logger.log_event("SESSION_COMPLETE", stats)
            
        except Exception as e:
            logger.error(f"Failed to save final statistics: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.running:
            return {'status': 'stopped'}
        
        runtime = time.time() - self.start_time
        current_fps = self.frame_count / runtime if runtime > 0 else 0
        
        return {
            'status': 'running',
            'runtime_seconds': runtime,
            'frames_processed': self.frame_count,
            'current_fps': current_fps,
            'camera_info': self.camera_stream.get_camera_info() if self.camera_stream else {},
            'motion_stats': self.motion_detector.get_motion_statistics() if self.motion_detector else {},
            'face_stats': self.face_detector.get_detection_statistics() if self.face_detector else {},
            'anomaly_stats': self.anomaly_detector.get_anomaly_statistics() if self.anomaly_detector else {},
            'gpu_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'learning_mode': self.anomaly_detector.is_learning if self.anomaly_detector else False
        }


class SurveillanceController:
    """High-level controller that manages SurveillanceSystem lifecycle."""

    def __init__(
        self,
        camera_id: int = 0,
        gui_mode: bool = True,
        enable_signal_handlers: bool = False,
    ):
        self.camera_id = camera_id
        self.gui_mode = gui_mode
        self.enable_signal_handlers = enable_signal_handlers
        self._system: Optional[SurveillanceSystem] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._system is not None and self._system.running

    def start(self, background: bool = False) -> bool:
        """Start the surveillance system."""
        with self._lock:
            if self.is_running:
                logger.warning("SurveillanceController: system already running")
                return False

            self._system = SurveillanceSystem(
                camera_id=self.camera_id,
                gui_mode=self.gui_mode,
                enable_signal_handlers=self.enable_signal_handlers,
            )

            if background:
                self._thread = threading.Thread(
                    target=self._run_system, name="SurveillanceThread", daemon=True
                )
                self._thread.start()
                return True

        try:
            self._system.start()
        finally:
            self._cleanup_after_stop()
        return True

    def start_background(self) -> bool:
        return self.start(background=True)

    def stop(self) -> None:
        with self._lock:
            system = self._system
            thread = self._thread
        if system:
            system.stop()
        if thread and thread.is_alive():
            thread.join(timeout=5)
        self._cleanup_after_stop()

    def restart(self, gui_mode: Optional[bool] = None) -> bool:
        if gui_mode is not None:
            self.gui_mode = gui_mode
        self.stop()
        return self.start(background=True)

    def reset_detectors(self):
        system = self._system
        if system:
            system._reset_detectors()

    def get_status(self) -> Dict[str, Any]:
        system = self._system
        if system:
            return system.get_system_status()
        return {'status': 'stopped'}

    def set_gui_mode(self, gui_mode: bool):
        self.gui_mode = gui_mode
        if self.is_running:
            logger.warning("GUI mode change will apply on next restart.")

    def update_camera(self, camera_id: int):
        self.camera_id = camera_id
        if self.is_running:
            logger.warning("Camera update requires restart to take effect.")

    def get_system(self) -> Optional[SurveillanceSystem]:
        return self._system

    def _run_system(self):
        assert self._system is not None
        try:
            self._system.start()
        finally:
            self._cleanup_after_stop()

    def _cleanup_after_stop(self):
        with self._lock:
            self._thread = None
            self._system = None


def is_cuda_ready() -> bool:
    """Return True if CUDA support is available."""
    return bool(TORCH_AVAILABLE and torch is not None and torch.cuda.is_available())


def get_cuda_device_name() -> Optional[str]:
    """Return the CUDA device name if available."""
    if not is_cuda_ready():
        return None
    try:
        return torch.cuda.get_device_name()  # type: ignore[attr-defined]
    except Exception:
        return None

class SurveillanceWebUI:
    """Optional web interface for surveillance system."""
    
    def __init__(self, surveillance_system: SurveillanceSystem):
        self.surveillance_system = surveillance_system
        self.app = None
    
    def create_streamlit_app(self):
        """Create Streamlit web interface."""
        try:
            import streamlit as st
            
            st.set_page_config(
                page_title="AI Surveillance System",
                page_icon="📹",
                layout="wide"
            )
            
            st.title("🔍 AI-Powered Surveillance System")
            
            # Sidebar controls
            with st.sidebar:
                st.header("System Controls")
                
                if st.button("Start System" if not self.surveillance_system.running else "Stop System"):
                    if not self.surveillance_system.running:
                        threading.Thread(target=self.surveillance_system.start, daemon=True).start()
                        st.success("System starting.")
                    else:
                        self.surveillance_system.stop()
                        st.info("System stopped")
                
                if st.button("Reset Detectors"):
                    self.surveillance_system._reset_detectors()
                    st.info("Detectors reset")
                
                st.header("Configuration")
                
                # Motion detection settings
                motion_threshold = st.slider("Motion Threshold", 0.1, 1.0, 
                                           config.anomaly.motion_threshold, 0.1)
                if st.button("Update Motion Threshold"):
                    self.surveillance_system.anomaly_detector.update_thresholds(
                        motion_threshold=motion_threshold
                    )
                    st.success("Threshold updated")
                
                # Learning mode toggle
                learning_enabled = st.checkbox("Learning Mode", 
                                             value=self.surveillance_system.anomaly_detector.is_learning)
                if st.button("Apply Learning Mode"):
                    self.surveillance_system.anomaly_detector.set_learning_mode(learning_enabled)
                    st.success(f"Learning mode {'enabled' if learning_enabled else 'disabled'}")
            
            # Main content
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header("System Status")
                status = self.surveillance_system.get_system_status()
                
                if status['status'] == 'running':
                    st.success("System Running")
                    st.metric("FPS", f"{status['current_fps']:.1f}")
                    st.metric("Frames Processed", status['frames_processed'])
                    st.metric("Runtime", f"{status['runtime_seconds']:.0f}s")
                else:
                    st.error("System Stopped")
                
                # Performance metrics
                if status['status'] == 'running':
                    st.subheader("Performance Metrics")
                    
                    motion_stats = status.get('motion_stats', {})
                    face_stats = status.get('face_stats', {})
                    anomaly_stats = status.get('anomaly_stats', {})
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Motion Events", motion_stats.get('average_events_per_frame', 0))
                    with col_b:
                        st.metric("Face Detections", face_stats.get('total_detections', 0))
                    with col_c:
                        st.metric("Anomalies", anomaly_stats.get('total_anomalies_detected', 0))
            
            with col2:
                st.header("Face Database")
                if hasattr(self.surveillance_system, 'face_matcher'):
                    db_info = self.surveillance_system.face_matcher.get_face_database_info()
                    st.metric("Known Faces", db_info['total_faces'])
                    
                    if db_info['faces']:
                        st.subheader("Recent Recognitions")
                        for face in db_info['faces'][:5]:  # Top 5
                            st.write(f"**{face['name']}**: {face['recognition_count']} recognitions")
                
                st.header("Recent Anomalies")
                if (hasattr(self.surveillance_system, 'anomaly_detector') and 
                    self.surveillance_system.anomaly_detector.anomaly_history):
                    recent_anomalies = list(self.surveillance_system.anomaly_detector.anomaly_history)[-5:]
                    for anomaly in reversed(recent_anomalies):
                        st.write(f"🚨 **{anomaly.anomaly_type.value}** ({anomaly.confidence:.2f})")
                        st.caption(anomaly.description)
            
            # Auto-refresh
            time.sleep(2)
            st.rerun()
            
        except ImportError:
            logger.error("Streamlit not available. Install with: pip install streamlit")
        except Exception as e:
            logger.error(f"Web UI error: {e}")