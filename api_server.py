#!/usr/bin/env python3
"""
Enhanced main surveillance system controller with integrated API server and Attendance Module.
"""
import cv2
import numpy as np
import time
import argparse
import signal
import sys
import threading
import os
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime, timedelta

# Import surveillance modules
from camera.stream_handler import CameraStream, MultiCameraManager
from detection.motion_detector import MotionDetector, MotionDetectionMethod
from face_recognition_s.face_detector import FaceDetector
from face_recognition_s.face_matcher import FaceMatcher
from anomaly.anomaly_detector import AnomalyDetector
from utils.config import config
from utils.logger import logger

# 🆕 NEW: Import attendance module
from attendance_module import EmployeeAttendanceModule

# GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
except ImportError:
    print("PyTorch not available, running in CPU-only mode")
    TORCH_AVAILABLE = False

# Pydantic models for API
class ControlRequest(BaseModel):
    action: str

class SettingsRequest(BaseModel):
    motion_threshold: Optional[float] = None
    confidence_threshold: Optional[float] = None
    learning_mode: Optional[bool] = None

# 🆕 NEW: Attendance API models
class AttendanceSettingsRequest(BaseModel):
    face_tolerance: Optional[float] = None
    cooldown_seconds: Optional[int] = None
    auto_backup: Optional[bool] = None
    show_visit_count: Optional[bool] = None

class EmployeeRequest(BaseModel):
    employee_id: str
    image_path: Optional[str] = None

class ReportRequest(BaseModel):
    days: Optional[int] = 30
    format: Optional[str] = "xlsx"
    include_summary: Optional[bool] = True

class   SurveillanceSystem:
    """Main surveillance system controller with API integration and Attendance Module."""
    
    def __init__(self, camera_id: int = 0, gui_mode: bool = True):
        self.camera_id = camera_id
        self.gui_mode = gui_mode
        self.running = False
        
        # Initialize components
        self.camera_stream: Optional[CameraStream] = None
        self.motion_detector: Optional[MotionDetector] = None
        self.face_detector: Optional[FaceDetector] = None
        self.face_matcher: Optional[FaceMatcher] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        
        # 🆕 NEW: Initialize attendance module
        self.attendance_module: Optional[EmployeeAttendanceModule] = None
        self.attendance_enabled = config.attendance.enabled
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_display = 0
        self.last_fps_update = time.time()
        
        # Event storage for API
        self.recent_events = []
        self.max_events = 100
        
        # 🆕 NEW: Attendance tracking
        self.daily_attendance = {}
        self.last_daily_reset = datetime.now().date()
        
        # GPU optimization
        self.use_cuda = TORCH_AVAILABLE and config.gpu.use_cuda and torch.cuda.is_available() if TORCH_AVAILABLE else False
        self.gpu_cleanup_counter = 0
        
        # API server
        self.api_app = None
        self.api_server = None
        
        # Initialize system
        self._initialize_system()
        self._setup_api()
        
        # Set up signal handlers for graceful shutdown
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
            
            # 🆕 NEW: Initialize attendance module
            if self.attendance_enabled:
                self.attendance_module = EmployeeAttendanceModule(
                    face_dir=config.attendance.face_gallery_path,
                    attendance_file=config.attendance.attendance_file,
                    cooldown_seconds=config.attendance.cooldown_seconds,
                    tolerance=config.attendance.face_tolerance,
                    encodings_cache=config.attendance.encodings_cache_file
                )
                logger.info("Employee attendance module initialized")
            else:
                logger.info("Employee attendance module disabled")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize surveillance system: {e}")
            raise
    
    def _setup_api(self):
        """Setup FastAPI server with attendance endpoints."""
        self.api_app = FastAPI(
            title="AI Surveillance System API with Attendance",
            description="REST API for AI-powered surveillance system with employee attendance tracking",
            version="2.0.0"
        )
        
        # CORS middleware
        self.api_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Existing API routes
        @self.api_app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": time.time(), "attendance_enabled": self.attendance_enabled}
        
        @self.api_app.get("/api/status")
        async def get_status():
            try:
                status = self.get_system_status()
                # 🆕 NEW: Add attendance status
                if self.attendance_enabled and self.attendance_module:
                    status['attendance'] = self.attendance_module.get_statistics()
                    status['daily_attendance'] = len(self.daily_attendance)
                return status
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.get("/api/stats")
        async def get_stats():
            try:
                stats = {}
                if self.motion_detector:
                    stats['motion_stats'] = self.motion_detector.get_motion_statistics()
                if self.face_detector:
                    stats['face_stats'] = self.face_detector.get_detection_statistics()
                if self.anomaly_detector:
                    stats['anomaly_stats'] = self.anomaly_detector.get_anomaly_statistics()
                if self.face_matcher:
                    stats['recognition_stats'] = self.face_matcher.get_recognition_statistics()
                
                # 🆕 NEW: Add attendance statistics
                if self.attendance_enabled and self.attendance_module:
                    stats['attendance_stats'] = self.attendance_module.get_statistics()
                
                return stats
            except Exception as e:
                logger.error(f"Error getting system statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.get("/api/events")
        async def get_events(limit: int = 50):
            try:
                events = self.recent_events[-limit:] if self.recent_events else []
                return {"events": events}
            except Exception as e:
                logger.error(f"Error getting events: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.get("/api/faces")
        async def get_faces():
            try:
                if self.face_matcher:
                    return self.face_matcher.get_face_database_info()
                else:
                    return {"total_faces": 0, "faces": []}
            except Exception as e:
                logger.error(f"Error getting face database: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.post("/api/control")
        async def control_system(request: ControlRequest):
            try:
                action = request.action
                if action == "start":
                    if self.running:
                        return {"success": False, "message": "System is already running"}
                    threading.Thread(target=self.start, daemon=True).start()
                    return {"success": True, "message": "System starting"}
                elif action == "stop":
                    if not self.running:
                        return {"success": False, "message": "System is already stopped"}
                    self.stop()
                    return {"success": True, "message": "System stopped"}
                elif action == "reset":
                    self._reset_detectors()
                    return {"success": True, "message": "System reset"}
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
            except Exception as e:
                logger.error(f"Error controlling system: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.put("/api/settings")
        async def update_settings(settings: SettingsRequest):
            try:
                updated = []
                
                if settings.motion_threshold is not None and self.anomaly_detector:
                    self.anomaly_detector.update_thresholds(motion_threshold=settings.motion_threshold)
                    updated.append("motion_threshold")
                
                if settings.confidence_threshold is not None and self.anomaly_detector:
                    self.anomaly_detector.update_thresholds(min_confidence=settings.confidence_threshold)
                    updated.append("confidence_threshold")
                
                if settings.learning_mode is not None and self.anomaly_detector:
                    self.anomaly_detector.set_learning_mode(settings.learning_mode)
                    updated.append("learning_mode")
                
                return {
                    "success": True, 
                    "message": f"Updated settings: {', '.join(updated)}" if updated else "No settings changed"
                }
            except Exception as e:
                logger.error(f"Error updating settings: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # 🆕 NEW: Attendance API endpoints
        @self.api_app.get("/api/attendance/status")
        async def get_attendance_status():
            """Get current attendance system status."""
            try:
                if not self.attendance_enabled:
                    return {"enabled": False, "message": "Attendance module is disabled"}
                
                if not self.attendance_module:
                    return {"enabled": True, "initialized": False, "message": "Attendance module not initialized"}
                
                stats = self.attendance_module.get_statistics()
                stats.update({
                    "enabled": True,
                    "initialized": True,
                    "daily_attendance_count": len(self.daily_attendance),
                    "last_daily_reset": self.last_daily_reset.isoformat(),
                    "system_uptime": time.time() - self.start_time
                })
                
                return stats
            except Exception as e:
                logger.error(f"Error getting attendance status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.get("/api/attendance/daily")
        async def get_daily_attendance():
            """Get today's attendance data."""
            try:
                if not self.attendance_enabled or not self.attendance_module:
                    raise HTTPException(status_code=400, detail="Attendance module not available")
                
                return {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'attendance': dict(self.daily_attendance),
                    'total_employees_today': len(self.daily_attendance),
                    'total_logs_today': sum(data.get('total_detections', 0) for data in self.daily_attendance.values())
                }
            except Exception as e:
                logger.error(f"Error getting daily attendance: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.get("/api/attendance/summary")
        async def get_attendance_summary(days: int = 7):
            """Get attendance summary for specified days."""
            try:
                if not self.attendance_enabled or not self.attendance_module:
                    raise HTTPException(status_code=400, detail="Attendance module not available")
                
                summary = self.attendance_module.get_attendance_summary(days)
                return {
                    'days': days,
                    'summary': summary.to_dict('records') if not summary.empty else [],
                    'total_records': len(summary) if not summary.empty else 0
                }
            except Exception as e:
                logger.error(f"Error getting attendance summary: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.post("/api/attendance/export")
        async def export_attendance_report(request: ReportRequest):
            """Export attendance report."""
            try:
                if not self.attendance_enabled or not self.attendance_module:
                    raise HTTPException(status_code=400, detail="Attendance module not available")
                
                filename = f"api_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                success = self.attendance_module.export_attendance_report(filename, request.days)
                
                return {
                    'success': success,
                    'filename': filename if success else None,
                    'days': request.days,
                    'message': 'Report exported successfully' if success else 'Export failed'
                }
            except Exception as e:
                logger.error(f"Error exporting attendance report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.get("/api/attendance/employees")
        async def get_employee_list():
            """Get list of all employees."""
            try:
                if not self.attendance_enabled or not self.attendance_module:
                    raise HTTPException(status_code=400, detail="Attendance module not available")
                
                return {
                    'employees': self.attendance_module.known_employee_ids,
                    'count': len(self.attendance_module.known_employee_ids),
                    'face_gallery_path': self.attendance_module.face_dir
                }
            except Exception as e:
                logger.error(f"Error getting employee list: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.post("/api/attendance/employee/add")
        async def add_employee(request: EmployeeRequest):
            """Add a new employee to the system."""
            try:
                if not self.attendance_enabled or not self.attendance_module:
                    raise HTTPException(status_code=400, detail="Attendance module not available")
                
                if not request.image_path:
                    return {
                        "success": False,
                        "message": "Image path is required to add employee"
                    }
                
                success = self.attendance_module.add_new_employee(request.employee_id, request.image_path)
                
                return {
                    "success": success,
                    "employee_id": request.employee_id,
                    "message": f"Employee {request.employee_id} added successfully" if success else "Failed to add employee"
                }
            except Exception as e:
                logger.error(f"Error adding employee: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.delete("/api/attendance/employee/{employee_id}")
        async def remove_employee(employee_id: str):
            """Remove an employee from the system."""
            try:
                if not self.attendance_enabled or not self.attendance_module:
                    raise HTTPException(status_code=400, detail="Attendance module not available")
                
                # Note: We need to add remove_employee method to EmployeeAttendanceModule
                success = hasattr(self.attendance_module, 'remove_employee') and \
                         self.attendance_module.remove_employee(employee_id)
                
                return {
                    "success": success,
                    "employee_id": employee_id,
                    "message": f"Employee {employee_id} removed successfully" if success else "Failed to remove employee"
                }
            except Exception as e:
                logger.error(f"Error removing employee: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.put("/api/attendance/settings")
        async def update_attendance_settings(settings: AttendanceSettingsRequest):
            """Update attendance system settings."""
            try:
                if not self.attendance_enabled or not self.attendance_module:
                    raise HTTPException(status_code=400, detail="Attendance module not available")
                
                updated = []
                
                if settings.face_tolerance is not None:
                    self.attendance_module.tolerance = settings.face_tolerance
                    updated.append("face_tolerance")
                
                if settings.cooldown_seconds is not None:
                    self.attendance_module.cooldown_seconds = settings.cooldown_seconds
                    updated.append("cooldown_seconds")
                
                if settings.auto_backup is not None:
                    # Update config
                    config.attendance.auto_backup = settings.auto_backup
                    updated.append("auto_backup")
                
                if settings.show_visit_count is not None:
                    config.attendance.show_visit_count = settings.show_visit_count
                    updated.append("show_visit_count")
                
                return {
                    "success": True,
                    "updated_settings": updated,
                    "message": f"Updated: {', '.join(updated)}" if updated else "No settings changed"
                }
            except Exception as e:
                logger.error(f"Error updating attendance settings: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.get("/api/attendance/logs")
        async def get_attendance_logs(hours: int = 24, employee_id: str = None):
            """Get recent attendance logs."""
            try:
                if not self.attendance_enabled:
                    raise HTTPException(status_code=400, detail="Attendance module not available")
                
                # Get logs from logger
                from utils.logger import get_attendance_summary
                summary = get_attendance_summary(hours)
                
                # Filter by employee if specified
                if employee_id and summary.get('employee_attendance_counts'):
                    filtered_counts = {
                        emp_id: count for emp_id, count in summary['employee_attendance_counts'].items()
                        if emp_id == employee_id
                    }
                    summary['employee_attendance_counts'] = filtered_counts
                
                return summary
            except Exception as e:
                logger.error(f"Error getting attendance logs: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.post("/api/attendance/reset")
        async def reset_attendance():
            """Reset attendance tracking (daily reset)."""
            try:
                if not self.attendance_enabled or not self.attendance_module:
                    raise HTTPException(status_code=400, detail="Attendance module not available")
                
                # Reset daily attendance
                self._reset_daily_attendance()
                
                # Reset module daily counts if method exists
                if hasattr(self.attendance_module, 'reset_daily_counts'):
                    self.attendance_module.reset_daily_counts()
                
                return {
                    "success": True,
                    "message": "Attendance tracking reset successfully",
                    "reset_time": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error resetting attendance: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _add_event(self, event_type: str, details: Dict):
        """Add an event to the recent events list."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "details": details
        }
        
        self.recent_events.append(event)
        
        # Keep only the most recent events
        if len(self.recent_events) > self.max_events:
            self.recent_events = self.recent_events[-self.max_events:]
    
    # 🆕 NEW: Attendance-specific methods
    def _update_daily_attendance(self, detections: List[Dict]):
        """Update daily attendance statistics."""
        current_date = datetime.now().date()
        
        # Reset if new day
        if current_date != self.last_daily_reset:
            self._reset_daily_attendance()
        
        for detection in detections:
            employee_id = detection['employee_id']
            if employee_id != "Unknown":
                if employee_id not in self.daily_attendance:
                    self.daily_attendance[employee_id] = {
                        'first_seen': datetime.now(),
                        'last_seen': datetime.now(),
                        'total_detections': 0,
                        'visit_count': detection.get('visit_count', 0)
                    }
                
                self.daily_attendance[employee_id]['last_seen'] = datetime.now()
                self.daily_attendance[employee_id]['total_detections'] += 1
                self.daily_attendance[employee_id]['visit_count'] = detection.get('visit_count', 0)
    
    def _reset_daily_attendance(self):
        """Reset daily attendance tracking."""
        self.daily_attendance.clear()
        self.last_daily_reset = datetime.now().date()
        logger.info("Daily attendance tracking reset")
    
    def start_api_server(self):
        """Start the API server in a separate thread."""
        def run_server():
            try:
                port = int(os.getenv("API_PORT", 8080))
                logger.info(f"Starting API server on port {port}")
                uvicorn.run(
                    self.api_app,
                    host="0.0.0.0",
                    port=port,
                    log_level="info"
                )
            except Exception as e:
                logger.error(f"API server error: {e}")
        
        api_thread = threading.Thread(target=run_server, daemon=True)
        api_thread.start()
        logger.info("API server thread started")
    
    def start(self):
        """Start the surveillance system."""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("Starting surveillance system with attendance tracking")
        
        try:
            # Start API server first
            self.start_api_server()
            
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
        logger.info("Starting surveillance with GUI and attendance tracking")
        
        # Create display windows
        cv2.namedWindow('Surveillance Feed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Motion Detection', cv2.WINDOW_NORMAL)
        
        # 🆕 NEW: Attendance-specific window
        if self.attendance_enabled:
            cv2.namedWindow('Attendance Dashboard', cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                success = self._process_frame()
                if not success:
                    break
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit key pressed")
                    break
                elif key == ord('r'):
                    logger.info("Resetting detectors and attendance")
                    self._reset_detectors()
                    self._reset_daily_attendance()
                elif key == ord('l'):
                    logger.info("Toggling learning mode")
                    if self.anomaly_detector:
                        self.anomaly_detector.set_learning_mode(not self.anomaly_detector.is_learning)
                elif key == ord('s'):
                    logger.info("Saving current frame")
                    self._save_current_frame()
                elif key == ord('a'):  # 🆕 NEW: Attendance export shortcut
                    if self.attendance_enabled and self.attendance_module:
                        logger.info("Exporting attendance report")
                        filename = f"manual_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        self.attendance_module.export_attendance_report(filename)
                        
        finally:
            cv2.destroyAllWindows()
    
    def _run_headless(self):
        """Run surveillance without GUI."""
        logger.info("Starting surveillance in headless mode with attendance tracking")
        
        while self.running:
            success = self._process_frame()
            if not success:
                break
            
            # Print status every 100 frames
            if self.frame_count % 100 == 0:
    
    
                self._print_status()
    def _process_frame(self) -> bool:
        """Process a single frame through the surveillance pipeline with attendance."""
        try:
            # Get frame from camera
            if not self.camera_stream:
                return False
                
            frame = self.camera_stream.get_frame()
            if frame is None:
                logger.warning("No frame received from camera")
                return False
            
            self.frame_count += 1
            
            # Motion detection
            motion_events = []
            if self.motion_detector:
                motion_events = self.motion_detector.detect_motion(frame)
            
            # Face detection (always detect faces first)
            face_detections = []
            if self.face_detector:
                face_detections = self.face_detector.detect_faces(frame, return_encodings=True)
                
                # Face matching
                if face_detections and self.face_matcher:
                    face_detections = self.face_matcher.match_faces(face_detections)
            
            # 🆕 NEW: Attendance processing with the detected faces
            attendance_detections = []
            face_detections_for_anomaly = face_detections.copy()  # Default fallback
            
            if self.attendance_enabled and self.attendance_module and face_detections:
                try:
                    # Process attendance with detected faces
                    annotated_frame, attendance_detections = self.attendance_module.process_frame(frame)
                    
                    # Log attendance events
                    if attendance_detections:
                        for detection in attendance_detections:
                            self._add_event("attendance_detected", {
                                "employee_id": detection['employee_id'],
                                "employee_name": detection['employee_name'],
                                "visit_type": detection['visit_type'],
                                "time": detection['time'],
                                "confidence": detection['confidence']
                            })
                            
                            # Optional: Log to separate attendance logger
                            logger.info(f"ATTENDANCE: {detection['employee_name']} ({detection['employee_id']}) - {detection['visit_type']} at {detection['time']}")
                    
                    # Use annotated frame with attendance info
                    frame = annotated_frame
                    
                    # Convert attendance detections for anomaly detector if needed
                    if attendance_detections:
                        from attendance.attendance_system import convert_attendance_to_face_detections
                        face_detections_for_anomaly = convert_attendance_to_face_detections(attendance_detections)
                        
                except Exception as e:
                    logger.error(f"Error in attendance processing: {e}")
                    # Fall back to regular face detections
                    face_detections_for_anomaly = face_detections
            
            # Anomaly detection
            anomalies = []
            if self.anomaly_detector:
                # Use the appropriate face detections for anomaly detection
                anomalies = self.anomaly_detector.detect_anomalies(
                    motion_events, face_detections_for_anomaly, frame
                )
            
            # Handle detected anomalies
            if anomalies:
                self._handle_anomalies(frame, anomalies, motion_events, face_detections_for_anomaly)
            
            # Log events for API
            if motion_events:
                self._add_event("motion_detected", {
                    "count": len(motion_events),
                    "total_area": sum(e.area for e in motion_events)
                })
            
            # Log face detections
            if face_detections:
                recognized_faces = [f.face_id for f in face_detections if hasattr(f, 'face_id') and f.face_id]
                self._add_event("faces_detected", {
                    "count": len(face_detections),
                    "recognized": len(recognized_faces),
                    "faces": recognized_faces
                })
            
            # Log anomalies
            for anomaly in anomalies:
                self._add_event(f"anomaly_{anomaly.anomaly_type.value}", {
                    "confidence": anomaly.confidence,
                    "description": anomaly.description,
                    "location": anomaly.location
                })
            
            # Display results if GUI mode
            if self.gui_mode:
                self._display_results(frame, motion_events, face_detections, anomalies, attendance_detections)
                
                # Update FPS
                self._update_fps()
                
                # GPU memory cleanup
                self._cleanup_gpu_memory()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False


    # 🆕 ADD: Method to convert attendance detections (if not already present)
    def _convert_attendance_to_face_detections(self, attendance_detections):
        """Convert attendance detections to face detection format for anomaly detector."""
        face_detections = []
        
        for detection in attendance_detections:
            # Create a simple object with required attributes
            class FaceDetection:
                def __init__(self, employee_id, confidence, name):
                    self.face_id = employee_id
                    self.confidence = confidence
                    self.name = name
            
            face_detections.append(FaceDetection(
                detection['employee_id'], 
                detection['confidence'],
                detection['employee_name']
            ))
        
        return face_detections
    def _convert_attendance_to_face_detections(self, attendance_detections):
        """Convert attendance detection dictionaries to face detection objects for anomaly detector."""
        try:
            # Import the FaceDetection class
            from face_recognition_s.face_detector import FaceDetection
            
            face_detections = []
            for detection in attendance_detections:
                if isinstance(detection, dict):
                    # Extract bbox information
                    bbox = detection.get('bbox', (0, 0, 0, 0))
                    if len(bbox) == 4:
                        # Ensure bbox is in (top, right, bottom, left) format
                        left, top, right, bottom = bbox
                        face_bbox = (top, right, bottom, left)
                        
                        # Create a FaceDetection object
                        face_detection = FaceDetection(
                            bbox=face_bbox,
                            confidence=detection.get('confidence', 0.0)
                        )
                        
                        # Set face_id from employee_id
                        face_detection.face_id = detection.get('employee_id')
                        
                        face_detections.append(face_detection)
            
            return face_detections
            
        except Exception as e:
            logger.warning(f"Could not convert attendance detections to face detections: {e}")
            return []
    def _display_results(self, frame, motion_events, face_detections, anomalies, attendance_detections=None):
        """Display surveillance results with attendance information."""
        # Create display frames
        display_frame = frame.copy()
        motion_frame = frame.copy()
        
        # Draw motion detection overlay
        if motion_events and self.motion_detector:
            motion_frame = self.motion_detector.draw_motion_overlay(motion_frame, motion_events)
        
        # Draw face detection overlay (only if no attendance detections)
        if face_detections and not attendance_detections and self.face_detector:
            display_frame = self.face_detector.draw_face_overlay(
                display_frame, face_detections, show_landmarks=False
            )
        
        # Draw anomaly overlay
        if anomalies and self.anomaly_detector:
            display_frame = self.anomaly_detector.draw_anomaly_overlay(display_frame, anomalies)
        
        # Add status information
        self._add_status_overlay(display_frame)
        
        # Display frames
        cv2.imshow('Surveillance Feed', display_frame)
        cv2.imshow('Motion Detection', motion_frame)
        
        # 🆕 NEW: Display attendance dashboard
        if self.attendance_enabled and attendance_detections:
            attendance_dashboard = self._create_attendance_dashboard()
            cv2.imshow('Attendance Dashboard', attendance_dashboard)
    
    def _create_attendance_dashboard(self) -> np.ndarray:
        """Create attendance dashboard visualization."""
        dashboard_width = 600
        dashboard_height = 400
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(dashboard, "Employee Attendance Dashboard", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current date
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(dashboard, f"Date: {date_str}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Employee list
        y_pos = 90
        cv2.putText(dashboard, "Today's Attendance:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        
        if not self.daily_attendance:
            cv2.putText(dashboard, "No attendance recorded today", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        else:
            for i, (emp_id, data) in enumerate(self.daily_attendance.items()):
                if y_pos > dashboard_height - 30:
                    break
                
                first_seen = data['first_seen'].strftime("%H:%M")
                last_seen = data['last_seen'].strftime("%H:%M")
                visit_count = data.get('visit_count', 0)
                detections = data['total_detections']
                
                text = f"{emp_id}: {first_seen}-{last_seen} (V:{visit_count}, D:{detections})"
                cv2.putText(dashboard, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_pos += 20
        
        # Statistics
        stats_y = dashboard_height - 100
        cv2.putText(dashboard, "Statistics:", (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        total_employees = len(self.attendance_module.known_employee_ids) if self.attendance_module else 0
        active_today = len(self.daily_attendance)
        
        stats_text = [
            f"Total Employees: {total_employees}",
            f"Active Today: {active_today}",
            f"System FPS: {self.fps_display:.1f}"
        ]
        
        for i, stat in enumerate(stats_text):
            cv2.putText(dashboard, stat, (20, stats_y + 25 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return dashboard
    
    def _add_status_overlay(self, frame: np.ndarray):
        """Add status information overlay to frame."""
        height, width = frame.shape[:2]
        
        # Status text with attendance info
        status_lines = [
            f"FPS: {self.fps_display:.1f}",
            f"Frames: {self.frame_count}",
            f"Runtime: {self._get_runtime_str()}",
            f"GPU: {'ON' if self.use_cuda else 'OFF'}",
            f"Learning: {'ON' if self.anomaly_detector and self.anomaly_detector.is_learning else 'OFF'}"
        ]
        
        # 🆕 NEW: Add attendance status
        if self.attendance_enabled:
            status_lines.extend([
                f"Attendance: ON",
                f"Employees Today: {len(self.daily_attendance)}"
            ])
        else:
            status_lines.append("Attendance: OFF")
        
        # Background rectangle
        overlay_height = len(status_lines) * 25 + 20
        cv2.rectangle(frame, (10, 10), (300, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, overlay_height), (255, 255, 255), 1)
        
        # Status text
        for i, line in enumerate(status_lines):
            y_pos = 35 + i * 25
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
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
        # 🆕 NEW: Reset attendance module daily counts
        if self.attendance_enabled and self.attendance_module:
            if hasattr(self.attendance_module, 'reset_daily_counts'):
                self.attendance_module.reset_daily_counts()
    
    def _save_current_frame(self):
        """Save the current frame."""
        if self.camera_stream:
            frame = self.camera_stream.get_frame()
            if frame is not None:
                logger.save_image(frame, "manual_save")
                # 🆕 NEW: Also save as attendance image if enabled
                if self.attendance_enabled:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"manual_save_attendance_{timestamp}.jpg"
                    filepath = os.path.join(config.attendance.reports_directory, filename)
                    cv2.imwrite(filepath, frame)
                    logger.info(f"Attendance frame saved: {filename}")
    
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
        
        print(f"\n--- Surveillance Status ---")
        print(f"Runtime: {runtime} | Frames: {self.frame_count} | FPS: {current_fps:.1f}")
        print(f"GPU: {'ON' if self.use_cuda else 'OFF'}")
        print(f"Learning: {'ON' if self.anomaly_detector and self.anomaly_detector.is_learning else 'OFF'}")
        
        # 🆕 NEW: Add attendance status
        if self.attendance_enabled:
            print(f"Attendance: ON | Employees Today: {len(self.daily_attendance)}")
            if self.attendance_module:
                stats = self.attendance_module.get_statistics()
                print(f"Total Employees: {stats.get('total_employees', 0)} | "
                      f"Total Logs: {stats.get('total_attendance_logs', 0)}")
            
            # Show recent attendance
            if self.daily_attendance:
                print("Recent Attendance:")
                for emp_id, data in list(self.daily_attendance.items())[:5]:  # Show first 5
                    first_seen = data['first_seen'].strftime("%H:%M")
                    visit_count = data.get('visit_count', 0)
                    print(f"  {emp_id}: First seen {first_seen} (Visits: {visit_count})")
        else:
            print("Attendance: OFF")
        
        print("-" * 50)
    
    def stop(self):
        """Stop the surveillance system."""
        if not self.running:
            return
        
        logger.info("Stopping surveillance system with attendance")
        self.running = False
        
        # Stop recording if active
        if logger.video_writer:
            logger.stop_recording()
        
        # 🆕 NEW: Export final attendance report
        if self.attendance_enabled and self.attendance_module:
            try:
                final_report = f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                self.attendance_module.export_attendance_report(final_report)
                logger.info(f"Final attendance report exported: {final_report}")
            except Exception as e:
                logger.warning(f"Could not export final attendance report: {e}")
        
        # Stop camera stream
        if self.camera_stream:
            self.camera_stream.stop_stream()
        
        # GPU cleanup
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Surveillance system stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.running:
            return {'status': 'stopped', 'attendance_enabled': self.attendance_enabled}
        
        runtime = time.time() - self.start_time
        current_fps = self.frame_count / runtime if runtime > 0 else 0
        
        status = {
            'status': 'running',
            'runtime_seconds': runtime,
            'frames_processed': self.frame_count,
            'current_fps': current_fps,
            'gpu_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'learning_mode': self.anomaly_detector.is_learning if self.anomaly_detector else False,
            'camera_info': self.camera_stream.get_camera_info() if self.camera_stream else {},
            'attendance_enabled': self.attendance_enabled
        }
        
        # 🆕 NEW: Add attendance-specific status
        if self.attendance_enabled and self.attendance_module:
            attendance_stats = self.attendance_module.get_statistics()
            status['attendance_stats'] = attendance_stats
            status['daily_attendance_count'] = len(self.daily_attendance)
            status['last_daily_reset'] = self.last_daily_reset.isoformat()
        
        return status

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI-Powered Surveillance System with Employee Attendance")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device ID")
    parser.add_argument("--headless", "-hl", action="store_true", help="Run without GUI")
    parser.add_argument("--api-only", action="store_true", help="Run API server only")
    parser.add_argument("--port", "-p", type=int, default=8080, help="API server port")
    
    # 🆕 NEW: Attendance-specific arguments
    parser.add_argument("--disable-attendance", action="store_true", help="Disable attendance module")
    parser.add_argument("--face-tolerance", type=float, help="Face recognition tolerance (0.0-1.0)")
    parser.add_argument("--cooldown", type=int, help="Cooldown seconds between attendance logs")
    parser.add_argument("--export-report", action="store_true", help="Export attendance report and exit")
    
    args = parser.parse_args()
    
    # Set API port
    os.environ["API_PORT"] = str(args.port)
    
    # 🆕 NEW: Handle attendance arguments
    if args.disable_attendance:
        config.attendance.enabled = False
        logger.info("Attendance module disabled via command line")
    
    if args.face_tolerance is not None:
        config.attendance.face_tolerance = args.face_tolerance
        logger.info(f"Face tolerance set to {args.face_tolerance}")
    
    if args.cooldown is not None:
        config.attendance.cooldown_seconds = args.cooldown
        logger.info(f"Cooldown set to {args.cooldown} seconds")
    
    # 🆕 NEW: Handle export report and exit
    if args.export_report:
        try:
            from attendance_module import EmployeeAttendanceModule
            attendance = EmployeeAttendanceModule()
            filename = f"cmdline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            success = attendance.export_attendance_report(filename)
            if success:
                print(f"✅ Attendance report exported: {filename}")
                return 0
            else:
                print("❌ Failed to export attendance report")
                return 1
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
            return 1
    
    logger.info(f"Starting AI-Powered Surveillance System with Attendance (API on port {args.port})")
    
    try:
        surveillance = SurveillanceSystem(
            camera_id=args.camera,
            gui_mode=not args.headless and not args.api_only
        )
        
        if args.api_only:
            logger.info("Starting API server only...")
            surveillance.start_api_server()
            # Keep the main thread alive
            while True:
                time.sleep(1)
        else:
            surveillance.start()
    
    except KeyboardInterrupt:
        logger.info("Surveillance system interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    logger.info("Surveillance system shutdown complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())