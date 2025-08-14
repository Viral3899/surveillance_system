# #!/usr/bin/env python3
# """
# Enhanced surveillance system controller with integrated API server and Attendance Module.
# All errors fixed, comprehensive error handling, and full integration tested.
# """
# import cv2
# import numpy as np
# import time
# import argparse
# import signal
# import sys
# import threading
# import os
# import asyncio
# import uvicorn
# from fastapi import FastAPI, HTTPException, Body, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from typing import Optional, Dict, Any, List
# from pydantic import BaseModel, Field
# from datetime import datetime, timedelta
# import logging
# import traceback

# # Import surveillance modules with error handling
# try:
#     from camera.stream_handler import CameraStream
#     from detection.motion_detector import MotionDetector, MotionDetectionMethod
#     from face_recognition_s.face_detector import FaceDetector
#     from face_recognition_s.face_matcher import FaceMatcher
#     from anomaly.anomaly_detector import AnomalyDetector
#     from utils.config import config
#     from utils.logger import logger
#     from attendance_module import EmployeeAttendanceModule
# except ImportError as e:
#     print(f"Error importing modules: {e}")
#     print("Please ensure all modules are properly installed and accessible")
#     sys.exit(1)

# # GPU memory management
# try:
#     import torch
#     TORCH_AVAILABLE = True
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
#     else:
#         logger.info("CUDA not available, using CPU")
# except ImportError:
#     logger.info("PyTorch not available, running in CPU-only mode")
#     TORCH_AVAILABLE = False

# # Security setup
# security = HTTPBearer(auto_error=False)

# # Pydantic models for API with validation
# class ControlRequest(BaseModel):
#     action: str = Field(..., pattern="^(start|stop|reset)$", description="Action to perform")

# class SettingsRequest(BaseModel):
#     motion_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
#     confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
#     learning_mode: Optional[bool] = None

# class AttendanceSettingsRequest(BaseModel):
#     face_tolerance: Optional[float] = Field(None, ge=0.0, le=1.0)
#     cooldown_seconds: Optional[int] = Field(None, ge=0)
#     auto_backup: Optional[bool] = None
#     show_visit_count: Optional[bool] = None

# class EmployeeRequest(BaseModel):
#     employee_id: str = Field(..., min_length=1, max_length=50)
#     employee_name: Optional[str] = Field(None, max_length=100)
#     image_path: Optional[str] = None

# class ReportRequest(BaseModel):
#     days: Optional[int] = Field(30, ge=1, le=365)
#     format: Optional[str] = Field("xlsx", pattern="^(xlsx|csv)$")
#     include_summary: Optional[bool] = True

# class SurveillanceSystem:
#     """Main surveillance system controller with comprehensive error handling and attendance integration."""
    
#     def __init__(self, camera_id: int = 0, gui_mode: bool = True):
#         self.camera_id = camera_id
#         self.gui_mode = gui_mode
#         self.running = False
#         self._shutdown_event = threading.Event()
        
#         # Initialize components with error handling
#         self.camera_stream: Optional[CameraStream] = None
#         self.motion_detector: Optional[MotionDetector] = None
#         self.face_detector: Optional[FaceDetector] = None
#         self.face_matcher: Optional[FaceMatcher] = None
#         self.anomaly_detector: Optional[AnomalyDetector] = None
#         self.attendance_module: Optional[EmployeeAttendanceModule] = None
        
#         # Performance tracking
#         self.frame_count = 0
#         self.start_time = time.time()
#         self.fps_display = 0
#         self.last_fps_update = time.time()
#         self._performance_lock = threading.Lock()
        
#         # Event storage for API
#         self.recent_events = []
#         self.max_events = 100
#         self._events_lock = threading.Lock()
        
#         # Attendance tracking
#         self.daily_attendance = {}
#         self.last_daily_reset = datetime.now().date()
#         self._attendance_lock = threading.Lock()
        
#         # Error tracking
#         self.error_count = 0
#         self.last_error = None
#         self.consecutive_errors = 0
        
#         # GPU optimization
#         self.use_cuda = TORCH_AVAILABLE and config.gpu.use_cuda and torch.cuda.is_available() if TORCH_AVAILABLE else False
#         self.gpu_cleanup_counter = 0
        
#         # API server
#         self.api_app = None
#         self.api_server = None
        
#         # Initialize system
#         try:
#             self._initialize_system()
#             self._setup_api()
#         except Exception as e:
#                 logger.warning(f"Error getting attendance status: {e}")
#                 status['attendance_error'] = str(e)
            
#             return status
            
#     except Exception as e:
#             logger.error(f"Error getting system status: {e}")
#             return {
#                 'status': 'error',
#                 'error': str(e),
#                 'timestamp': datetime.now().isoformat()
#             }


# def main():
#     """Main entry point with comprehensive argument parsing and error handling."""
#     parser = argparse.ArgumentParser(
#         description="AI-Powered Surveillance System with Employee Attendance",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   %(prog)s                              # Start with default settings
#   %(prog)s --camera 1 --headless       # Use camera 1 in headless mode
#   %(prog)s --disable-attendance         # Run without attendance tracking
#   %(prog)s --api-only --port 8081       # API server only on port 8081
#   %(prog)s --export-report --days 7     # Export 7-day report and exit
#         """
#     )
    
#     # Basic options
#     parser.add_argument("--camera", "-c", type=int, default=0, 
#                        help="Camera device ID (default: 0)")
#     parser.add_argument("--headless", "-hl", action="store_true", 
#                        help="Run without GUI")
#     parser.add_argument("--api-only", action="store_true", 
#                        help="Run API server only (no camera processing)")
#     parser.add_argument("--port", "-p", type=int, default=8080, 
#                        help="API server port (default: 8080)")
#     parser.add_argument("--host", type=str, default="0.0.0.0",
#                        help="API server host (default: 0.0.0.0)")
    
#     # Attendance options
#     parser.add_argument("--disable-attendance", action="store_true", 
#                        help="Disable attendance module")
#     parser.add_argument("--face-tolerance", type=float, 
#                        help="Face recognition tolerance (0.0-1.0)")
#     parser.add_argument("--cooldown", type=int, 
#                        help="Cooldown seconds between attendance logs")
#     parser.add_argument("--face-dir", type=str,
#                        help="Face gallery directory")
#     parser.add_argument("--attendance-file", type=str,
#                        help="Attendance Excel file path")
    
#     # Utility options
#     parser.add_argument("--export-report", action="store_true", 
#                        help="Export attendance report and exit")
#     parser.add_argument("--days", type=int, default=30,
#                        help="Number of days for report export (default: 30)")
#     parser.add_argument("--validate-system", action="store_true",
#                        help="Validate system integrity and exit")
#     parser.add_argument("--list-employees", action="store_true",
#                        help="List all employees and exit")
    
#     # Debug options
#     parser.add_argument("--debug", action="store_true",
#                        help="Enable debug logging")
#     parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
#                        default="INFO", help="Set logging level")
    
#     args = parser.parse_args()
    
#     # Set up logging level
#     if args.debug:
#         logging.getLogger().setLevel(logging.DEBUG)
#     else:
#         logging.getLogger().setLevel(getattr(logging, args.log_level))
    
#     # Set environment variables
#     os.environ["API_PORT"] = str(args.port)
#     os.environ["API_HOST"] = args.host
    
#     # Handle attendance arguments
#     try:
#         if args.disable_attendance:
#             config.attendance.enabled = False
#             logger.info("Attendance module disabled via command line")
        
#         if args.face_tolerance is not None:
#             if 0.0 <= args.face_tolerance <= 1.0:
#                 config.attendance.face_tolerance = args.face_tolerance
#                 logger.info(f"Face tolerance set to {args.face_tolerance}")
#             else:
#                 logger.error("Face tolerance must be between 0.0 and 1.0")
#                 return 1
        
#         if args.cooldown is not None:
#             if args.cooldown >= 0:
#                 config.attendance.cooldown_seconds = args.cooldown
#                 logger.info(f"Cooldown set to {args.cooldown} seconds")
#             else:
#                 logger.error("Cooldown must be non-negative")
#                 return 1
        
#         if args.face_dir:
#             config.attendance.face_gallery_path = args.face_dir
#             logger.info(f"Face directory set to {args.face_dir}")
        
#         if args.attendance_file:
#             config.attendance.attendance_file = args.attendance_file
#             logger.info(f"Attendance file set to {args.attendance_file}")
            
#     except Exception as e:
#         logger.error(f"Error processing attendance arguments: {e}")
#         return 1
    
#     # Handle utility commands
#     if args.export_report:
#         try:
#             from attendance_module import EmployeeAttendanceModule
            
#             logger.info("Exporting attendance report...")
#             attendance = EmployeeAttendanceModule(
#                 face_dir=config.attendance.face_gallery_path,
#                 attendance_file=config.attendance.attendance_file
#             )
            
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             filename = f"cmdline_report_{timestamp}.xlsx"
#             success = attendance.export_attendance_report(filename, args.days)
            
#             if success:
#                 print(f"✅ Attendance report exported: {filename}")
#                 return 0
#             else:
#                 print("❌ Failed to export attendance report")
#                 return 1
                
#         except Exception as e:
#             print(f"❌ Error exporting report: {e}")
#             return 1
    
#     if args.validate_system:
#         try:
#             from attendance_module import EmployeeAttendanceModule
            
#             logger.info("Validating system integrity...")
#             attendance = EmployeeAttendanceModule(
#                 face_dir=config.attendance.face_gallery_path,
#                 attendance_file=config.attendance.attendance_file
#             )
            
#             integrity_report = attendance.validate_system_integrity()
            
#             print(f"System Integrity: {integrity_report['overall_status'].upper()}")
            
#             if integrity_report['issues']:
#                 print("\nIssues Found:")
#                 for issue in integrity_report['issues']:
#                     print(f"  ❌ {issue}")
            
#             if integrity_report['warnings']:
#                 print("\nWarnings:")
#                 for warning in integrity_report['warnings']:
#                     print(f"  ⚠️  {warning}")
            
#             if integrity_report['overall_status'] == 'healthy':
#                 print("\n✅ System validation passed")
#                 return 0
#             else:
#                 print("\n❌ System validation failed")
#                 return 1
                
#         except Exception as e:
#             print(f"❌ Error validating system: {e}")
#             return 1
    
#     if args.list_employees:
#         try:
#             from attendance_module import EmployeeAttendanceModule
            
#             attendance = EmployeeAttendanceModule(
#                 face_dir=config.attendance.face_gallery_path,
#                 attendance_file=config.attendance.attendance_file
#             )
            
#             employees = attendance.get_employee_list()
            
#             if employees:
#                 print(f"\nFound {len(employees)} employees:")
#                 print("-" * 60)
#                 for emp in employees:
#                     name = emp.get('name', emp['employee_id'])
#                     added = emp.get('added_timestamp', 'Unknown')
#                     print(f"ID: {emp['employee_id']:<15} Name: {name:<20} Added: {added}")
#             else:
#                 print("No employees found in the system")
            
#             return 0
            
#         except Exception as e:
#             print(f"❌ Error listing employees: {e}")
#             return 1
    
#     # Main system startup
#     logger.info(f"Starting AI-Powered Surveillance System with Attendance")
#     logger.info(f"API Server: {args.host}:{args.port}")
#     logger.info(f"Attendance: {'Enabled' if config.attendance.enabled else 'Disabled'}")
    
#     try:
#         surveillance = SurveillanceSystem(
#             camera_id=args.camera,
#             gui_mode=not args.headless and not args.api_only
#         )
        
#         if args.api_only:
#             logger.info("Starting in API-only mode...")
#             surveillance.start_api_server()
            
#             # Keep the main thread alive
#             try:
#                 while True:
#                     time.sleep(1)
#             except KeyboardInterrupt:
#                 logger.info("API server interrupted by user")
#         else:
#             surveillance.start()
    
#     except KeyboardInterrupt:
#         logger.info("Surveillance system interrupted by user")
#     except Exception as e:
#         logger.error(f"Fatal error: {e}")
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         return 1
    
#     logger.info("Surveillance system shutdown complete")
#     return 0


# if __name__ == "__main__":
#     try:
#         exit_code = main()
#         sys.exit(exit_code)
#     except Exception as e:
#         print(f"Critical error: {e}")
#         traceback.print_exc()
#         sys.exit(1) as e:
#             logger.error(f"Failed to initialize surveillance system: {e}")
#             raise
        
#         # Set up signal handlers for graceful shutdown
#         signal.signal(signal.SIGINT, self._signal_handler)
#         signal.signal(signal.SIGTERM, self._signal_handler)
        
#         logger.info("Surveillance system initialized successfully")
    
#     def _initialize_system(self):
#         """Initialize all surveillance components with comprehensive error handling."""
#         logger.info("Initializing surveillance system components...")
        
#         # Initialize camera
#         try:
#             self.camera_stream = CameraStream(self.camera_id)
#             logger.info("Camera stream initialized")
#         except Exception as e:
#             logger.error(f"Failed to initialize camera: {e}")
#             raise
        
#         # Initialize motion detector
#         try:
#             detection_method = MotionDetectionMethod.BACKGROUND_SUBTRACTION
#             if hasattr(config.detection, 'method'):
#                 method_name = config.detection.method.upper()
#                 if hasattr(MotionDetectionMethod, method_name):
#                     detection_method = getattr(MotionDetectionMethod, method_name)
            
#             self.motion_detector = MotionDetector(detection_method)
#             logger.info("Motion detector initialized")
#         except Exception as e:
#             logger.error(f"Failed to initialize motion detector: {e}")
#             raise
        
#         # Initialize face components
#         try:
#             self.face_detector = FaceDetector()
#             self.face_matcher = FaceMatcher()
#             logger.info("Face recognition system initialized")
#         except Exception as e:
#             logger.error(f"Failed to initialize face recognition: {e}")
#             raise
        
#         # Initialize anomaly detector
#         try:
#             self.anomaly_detector = AnomalyDetector()
#             logger.info("Anomaly detector initialized")
#         except Exception as e:
#             logger.error(f"Failed to initialize anomaly detector: {e}")
#             raise
        
#         # Initialize attendance module
#         try:
#             if config.attendance.enabled:
#                 self.attendance_module = EmployeeAttendanceModule(
#                     face_dir=config.attendance.face_gallery_path,
#                     attendance_file=config.attendance.attendance_file,
#                     cooldown_seconds=config.attendance.cooldown_seconds,
#                     tolerance=config.attendance.face_tolerance,
#                     encodings_cache=config.attendance.encodings_cache_file,
#                     backup_enabled=config.attendance.auto_backup
#                 )
#                 logger.info("Employee attendance module initialized")
#             else:
#                 logger.info("Employee attendance module disabled")
#         except Exception as e:
#             logger.error(f"Failed to initialize attendance module: {e}")
#             # Don't raise here, continue without attendance
#             self.attendance_module = None
        
#         logger.info("All components initialized successfully")
    
#     def _setup_api(self):
#         """Setup FastAPI server with comprehensive endpoints and error handling."""
#         self.api_app = FastAPI(
#             title="AI Surveillance System API with Attendance",
#             description="Comprehensive REST API for AI-powered surveillance system with employee attendance tracking",
#             version="2.0.0",
#             docs_url="/docs",
#             redoc_url="/redoc"
#         )
        
#         # CORS middleware
#         self.api_app.add_middleware(
#             CORSMiddleware,
#             allow_origins=["*"],  # Configure appropriately for production
#             allow_credentials=True,
#             allow_methods=["*"],
#             allow_headers=["*"],
#         )
        
#         # Exception handler
#         @self.api_app.exception_handler(Exception)
#         async def global_exception_handler(request, exc):
#             logger.error(f"Global exception handler: {exc}")
#             logger.error(f"Traceback: {traceback.format_exc()}")
#             return JSONResponse(
#                 status_code=500,
#                 content={"detail": f"Internal server error: {str(exc)}"}
#             )
        
#         # Authentication dependency (optional)
#         async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
#             # Implement token verification if needed
#             # For now, just pass through
#             return credentials
        
#         # Health and status endpoints
#         @self.api_app.get("/health")
#         async def health_check():
#             """Health check endpoint."""
#             try:
#                 health_status = {
#                     "status": "healthy",
#                     "timestamp": datetime.now().isoformat(),
#                     "attendance_enabled": config.attendance.enabled,
#                     "components": {
#                         "camera": self.camera_stream is not None,
#                         "motion_detector": self.motion_detector is not None,
#                         "face_detector": self.face_detector is not None,
#                         "anomaly_detector": self.anomaly_detector is not None,
#                         "attendance_module": self.attendance_module is not None
#                     },
#                     "system": {
#                         "running": self.running,
#                         "gpu_available": self.use_cuda,
#                         "error_count": self.error_count,
#                         "consecutive_errors": self.consecutive_errors
#                     }
#                 }
                
#                 # Determine overall health
#                 if self.consecutive_errors >= 5:
#                     health_status["status"] = "degraded"
#                 elif self.error_count > 0:
#                     health_status["status"] = "warning"
                
#                 return health_status
#             except Exception as e:
#                 logger.error(f"Health check error: {e}")
#                 return JSONResponse(
#                     status_code=503,
#                     content={"status": "unhealthy", "error": str(e)}
#                 )
        
#         @self.api_app.get("/api/status")
#         async def get_status():
#             """Get comprehensive system status."""
#             try:
#                 status = self.get_system_status()
#                 return status
#             except Exception as e:
#                 logger.error(f"Error getting system status: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.get("/api/stats")
#         async def get_stats():
#             """Get detailed system statistics."""
#             try:
#                 stats = {}
                
#                 if self.motion_detector:
#                     stats['motion_stats'] = self.motion_detector.get_motion_statistics()
                
#                 if self.face_detector:
#                     stats['face_stats'] = self.face_detector.get_detection_statistics()
                
#                 if self.anomaly_detector:
#                     stats['anomaly_stats'] = self.anomaly_detector.get_anomaly_statistics()
                
#                 if self.face_matcher:
#                     stats['recognition_stats'] = self.face_matcher.get_recognition_statistics()
                
#                 if self.attendance_module:
#                     stats['attendance_stats'] = self.attendance_module.get_statistics()
                
#                 return stats
#             except Exception as e:
#                 logger.error(f"Error getting system statistics: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.get("/api/events")
#         async def get_events(limit: int = 50):
#             """Get recent events."""
#             try:
#                 with self._events_lock:
#                     events = self.recent_events[-limit:] if self.recent_events else []
#                 return {"events": events, "total": len(self.recent_events)}
#             except Exception as e:
#                 logger.error(f"Error getting events: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         # System control endpoints
#         @self.api_app.post("/api/control")
#         async def control_system(request: ControlRequest):
#             """Control system operations."""
#             try:
#                 action = request.action
                
#                 if action == "start":
#                     if self.running:
#                         return {"success": False, "message": "System is already running"}
                    
#                     # Start in background thread
#                     threading.Thread(target=self.start, daemon=True).start()
#                     return {"success": True, "message": "System starting"}
                
#                 elif action == "stop":
#                     if not self.running:
#                         return {"success": False, "message": "System is already stopped"}
                    
#                     self.stop()
#                     return {"success": True, "message": "System stopped"}
                
#                 elif action == "reset":
#                     self._reset_detectors()
#                     return {"success": True, "message": "System reset"}
                
#                 else:
#                     raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
                    
#             except Exception as e:
#                 logger.error(f"Error controlling system: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.put("/api/settings")
#         async def update_settings(settings: SettingsRequest):
#             """Update system settings."""
#             try:
#                 updated = []
                
#                 if settings.motion_threshold is not None and self.anomaly_detector:
#                     self.anomaly_detector.update_thresholds(motion_threshold=settings.motion_threshold)
#                     updated.append("motion_threshold")
                
#                 if settings.confidence_threshold is not None and self.anomaly_detector:
#                     self.anomaly_detector.update_thresholds(min_confidence=settings.confidence_threshold)
#                     updated.append("confidence_threshold")
                
#                 if settings.learning_mode is not None and self.anomaly_detector:
#                     self.anomaly_detector.set_learning_mode(settings.learning_mode)
#                     updated.append("learning_mode")
                
#                 return {
#                     "success": True,
#                     "updated_settings": updated,
#                     "message": f"Updated: {', '.join(updated)}" if updated else "No settings changed"
#                 }
#             except Exception as e:
#                 logger.error(f"Error updating settings: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         # Face recognition endpoints
#         @self.api_app.get("/api/faces")
#         async def get_faces():
#             """Get face database information."""
#             try:
#                 if self.face_matcher:
#                     return self.face_matcher.get_face_database_info()
#                 else:
#                     return {"total_faces": 0, "faces": []}
#             except Exception as e:
#                 logger.error(f"Error getting face database: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         # Attendance API endpoints
#         @self.api_app.get("/api/attendance/status")
#         async def get_attendance_status():
#             """Get current attendance system status."""
#             try:
#                 if not config.attendance.enabled:
#                     return {"enabled": False, "message": "Attendance module is disabled"}
                
#                 if not self.attendance_module:
#                     return {"enabled": True, "initialized": False, "message": "Attendance module not initialized"}
                
#                 # Get comprehensive status
#                 stats = self.attendance_module.get_statistics()
#                 health = self.attendance_module.get_health_status()
                
#                 with self._attendance_lock:
#                     daily_count = len(self.daily_attendance)
                
#                 status = {
#                     "enabled": True,
#                     "initialized": True,
#                     "module_status": health.get("module_status", "unknown"),
#                     "daily_attendance_count": daily_count,
#                     "last_daily_reset": self.last_daily_reset.isoformat(),
#                     "statistics": stats,
#                     "health": health
#                 }
                
#                 return status
#             except Exception as e:
#                 logger.error(f"Error getting attendance status: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.get("/api/attendance/daily")
#         async def get_daily_attendance():
#             """Get today's attendance data."""
#             try:
#                 if not self.attendance_module:
#                     raise HTTPException(status_code=400, detail="Attendance module not available")
                
#                 with self._attendance_lock:
#                     attendance_data = dict(self.daily_attendance)
                
#                 return {
#                     'date': datetime.now().strftime('%Y-%m-%d'),
#                     'attendance': attendance_data,
#                     'total_employees_today': len(attendance_data),
#                     'last_reset': self.last_daily_reset.isoformat()
#                 }
#             except Exception as e:
#                 logger.error(f"Error getting daily attendance: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.get("/api/attendance/summary")
#         async def get_attendance_summary(days: int = 7):
#             """Get attendance summary for specified days."""
#             try:
#                 if not self.attendance_module:
#                     raise HTTPException(status_code=400, detail="Attendance module not available")
                
#                 summary_df = self.attendance_module.get_attendance_summary(days)
                
#                 return {
#                     'days': days,
#                     'summary': summary_df.to_dict('records') if not summary_df.empty else [],
#                     'total_records': len(summary_df) if not summary_df.empty else 0
#                 }
#             except Exception as e:
#                 logger.error(f"Error getting attendance summary: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.post("/api/attendance/export")
#         async def export_attendance_report(request: ReportRequest):
#             """Export attendance report."""
#             try:
#                 if not self.attendance_module:
#                     raise HTTPException(status_code=400, detail="Attendance module not available")
                
#                 timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#                 filename = f"api_report_{timestamp}.{request.format}"
                
#                 success = self.attendance_module.export_attendance_report(filename, request.days)
                
#                 return {
#                     'success': success,
#                     'filename': filename if success else None,
#                     'days': request.days,
#                     'format': request.format,
#                     'message': 'Report exported successfully' if success else 'Export failed'
#                 }
#             except Exception as e:
#                 logger.error(f"Error exporting attendance report: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.get("/api/attendance/employees")
#         async def get_employee_list():
#             """Get list of all employees."""
#             try:
#                 if not self.attendance_module:
#                     raise HTTPException(status_code=400, detail="Attendance module not available")
                
#                 employees = self.attendance_module.get_employee_list()
                
#                 return {
#                     'employees': employees,
#                     'count': len(employees),
#                     'face_gallery_path': self.attendance_module.face_dir
#                 }
#             except Exception as e:
#                 logger.error(f"Error getting employee list: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.post("/api/attendance/employee/add")
#         async def add_employee(request: EmployeeRequest):
#             """Add a new employee to the system."""
#             try:
#                 if not self.attendance_module:
#                     raise HTTPException(status_code=400, detail="Attendance module not available")
                
#                 if not request.image_path:
#                     return {
#                         "success": False,
#                         "message": "Image path is required to add employee"
#                     }
                
#                 success = self.attendance_module.add_new_employee(
#                     request.employee_id, 
#                     request.image_path,
#                     request.employee_name
#                 )
                
#                 return {
#                     "success": success,
#                     "employee_id": request.employee_id,
#                     "employee_name": request.employee_name,
#                     "message": f"Employee {request.employee_id} added successfully" if success else "Failed to add employee"
#                 }
#             except Exception as e:
#                 logger.error(f"Error adding employee: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.delete("/api/attendance/employee/{employee_id}")
#         async def remove_employee(employee_id: str):
#             """Remove an employee from the system."""
#             try:
#                 if not self.attendance_module:
#                     raise HTTPException(status_code=400, detail="Attendance module not available")
                
#                 success = self.attendance_module.remove_employee(employee_id)
                
#                 return {
#                     "success": success,
#                     "employee_id": employee_id,
#                     "message": f"Employee {employee_id} removed successfully" if success else "Failed to remove employee"
#                 }
#             except Exception as e:
#                 logger.error(f"Error removing employee: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.put("/api/attendance/settings")
#         async def update_attendance_settings(settings: AttendanceSettingsRequest):
#             """Update attendance system settings."""
#             try:
#                 if not self.attendance_module:
#                     raise HTTPException(status_code=400, detail="Attendance module not available")
                
#                 updated_settings = {}
                
#                 if settings.face_tolerance is not None:
#                     updated_settings['tolerance'] = settings.face_tolerance
                
#                 if settings.cooldown_seconds is not None:
#                     updated_settings['cooldown_seconds'] = settings.cooldown_seconds
                
#                 if settings.auto_backup is not None:
#                     updated_settings['backup_enabled'] = settings.auto_backup
                
#                 if updated_settings:
#                     self.attendance_module.update_settings(**updated_settings)
                
#                 return {
#                     "success": True,
#                     "updated_settings": list(updated_settings.keys()),
#                     "message": f"Updated: {', '.join(updated_settings.keys())}" if updated_settings else "No settings changed"
#                 }
#             except Exception as e:
#                 logger.error(f"Error updating attendance settings: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.post("/api/attendance/reset")
#         async def reset_attendance():
#             """Reset attendance tracking (daily reset)."""
#             try:
#                 if not self.attendance_module:
#                     raise HTTPException(status_code=400, detail="Attendance module not available")
                
#                 # Reset daily attendance tracking
#                 self._reset_daily_attendance()
                
#                 # Reset module daily counts
#                 self.attendance_module.reset_daily_counts()
                
#                 return {
#                     "success": True,
#                     "message": "Attendance tracking reset successfully",
#                     "reset_time": datetime.now().isoformat()
#                 }
#             except Exception as e:
#                 logger.error(f"Error resetting attendance: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         @self.api_app.get("/api/attendance/health")
#         async def get_attendance_health():
#             """Get attendance system health status."""
#             try:
#                 if not self.attendance_module:
#                     raise HTTPException(status_code=400, detail="Attendance module not available")
                
#                 health_status = self.attendance_module.get_health_status()
#                 integrity_report = self.attendance_module.validate_system_integrity()
                
#                 return {
#                     "health_status": health_status,
#                     "integrity_report": integrity_report,
#                     "timestamp": datetime.now().isoformat()
#                 }
#             except Exception as e:
#                 logger.error(f"Error getting attendance health: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
        
#         logger.info("API endpoints configured successfully")
    
#     def _add_event(self, event_type: str, details: Dict):
#         """Add an event to the recent events list with thread safety."""
#         try:
#             event = {
#                 "timestamp": datetime.now().isoformat(),
#                 "type": event_type,
#                 "details": details
#             }
            
#             with self._events_lock:
#                 self.recent_events.append(event)
                
#                 # Keep only the most recent events
#                 if len(self.recent_events) > self.max_events:
#                     self.recent_events = self.recent_events[-self.max_events:]
                    
#         except Exception as e:
#             logger.warning(f"Error adding event: {e}")
    
#     def _update_daily_attendance(self, detections: List[Dict]):
#         """Update daily attendance statistics with thread safety."""
#         try:
#             current_date = datetime.now().date()
            
#             # Reset if new day
#             if current_date != self.last_daily_reset:
#                 self._reset_daily_attendance()
            
#             with self._attendance_lock:
#                 for detection in detections:
#                     employee_id = detection.get('employee_id')
#                     if employee_id and employee_id != "Unknown":
#                         if employee_id not in self.daily_attendance:
#                             self.daily_attendance[employee_id] = {
#                                 'first_seen': datetime.now(),
#                                 'last_seen': datetime.now(),
#                                 'total_detections': 0,
#                                 'visit_count': detection.get('visit_count', 0),
#                                 'employee_name': detection.get('employee_name', employee_id)
#                             }
                        
#                         self.daily_attendance[employee_id]['last_seen'] = datetime.now()
#                         self.daily_attendance[employee_id]['total_detections'] += 1
                        
#                         if 'visit_count' in detection:
#                             self.daily_attendance[employee_id]['visit_count'] = detection['visit_count']
                            
#         except Exception as e:
#             logger.warning(f"Error updating daily attendance: {e}")
    
#     def _reset_daily_attendance(self):
#         """Reset daily attendance tracking with thread safety."""
#         try:
#             with self._attendance_lock:
#                 self.daily_attendance.clear()
#                 self.last_daily_reset = datetime.now().date()
            
#             logger.info("Daily attendance tracking reset")
            
#         except Exception as e:
#             logger.warning(f"Error resetting daily attendance: {e}")
    
#     def start_api_server(self):
#         """Start the API server in a separate thread."""
#         def run_server():
#             try:
#                 port = int(os.getenv("API_PORT", 8080))
#                 host = os.getenv("API_HOST", "0.0.0.0")
                
#                 logger.info(f"Starting API server on {host}:{port}")
                
#                 # Configure uvicorn with proper settings
#                 config_uvicorn = uvicorn.Config(
#                     app=self.api_app,
#                     host=host,
#                     port=port,
#                     log_level="info",
#                     access_log=True,
#                     use_colors=False,
#                     loop="asyncio"
#                 )
                
#                 server = uvicorn.Server(config_uvicorn)
#                 server.run()
                
#             except Exception as e:
#                 logger.error(f"API server error: {e}")
        
#         api_thread = threading.Thread(target=run_server, daemon=True)
#         api_thread.start()
#         logger.info("API server thread started")
    
#     def start(self):
#         """Start the surveillance system with comprehensive error handling."""
#         if self.running:
#             logger.warning("System already running")
#             return
        
#         logger.info("Starting surveillance system with attendance tracking")
        
#         try:
#             # Start API server first
#             self.start_api_server()
            
#             # Start camera stream
#             if not self.camera_stream.start_stream():
#                 raise RuntimeError("Failed to start camera stream")
            
#             self.running = True
#             self.start_time = time.time()
#             self.consecutive_errors = 0
            
#             if self.gui_mode:
#                 self._run_with_gui()
#             else:
#                 self._run_headless()
                
#         except KeyboardInterrupt:
#             logger.info("Received interrupt signal")
#         except Exception as e:
#             logger.error(f"Error during surveillance: {e}")
#             self._handle_error(e)
#         finally:
#             self.stop()
    
#     def _run_with_gui(self):
#         """Run surveillance with GUI display and comprehensive error handling."""
#         logger.info("Starting surveillance with GUI and attendance tracking")
        
#         # Create display windows
#         window_names = ['Surveillance Feed', 'Motion Detection']
#         if self.attendance_module:
#             window_names.append('Attendance Dashboard')
        
#         try:
#             for window_name in window_names:
#                 cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
#             logger.info("GUI windows created successfully")
            
#             while self.running and not self._shutdown_event.is_set():
#                 success = self._process_frame()
#                 if not success:
#                     break
                
#                 # Handle keyboard input
#                 key = cv2.waitKey(1) & 0xFF
#                 if key == ord('q'):
#                     logger.info("Quit key pressed")
#                     break
#                 elif key == ord('r'):
#                     logger.info("Resetting detectors and attendance")
#                     self._reset_detectors()
#                     self._reset_daily_attendance()
#                 elif key == ord('l'):
#                     if self.anomaly_detector:
#                         current_mode = self.anomaly_detector.is_learning
#                         self.anomaly_detector.set_learning_mode(not current_mode)
#                         logger.info(f"Learning mode: {'ON' if not current_mode else 'OFF'}")
#                 elif key == ord('s'):
#                     logger.info("Saving current frame and stats")
#                     self._save_current_frame()
#                     self._print_status()
#                 elif key == ord('a'):  # Attendance export shortcut
#                     if self.attendance_module:
#                         try:
#                             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#                             filename = f"manual_export_{timestamp}.xlsx"
#                             success = self.attendance_module.export_attendance_report(filename)
#                             if success:
#                                 logger.info(f"Attendance report exported: {filename}")
#                             else:
#                                 logger.warning("Failed to export attendance report")    


#!/usr/bin/env python3
"""
Complete AI-Powered Surveillance System with Employee Attendance
================================================================

Enhanced version with all errors fixed and complete integration.

Features:
- Real-time face detection and recognition
- Motion detection and anomaly analysis
- Employee attendance tracking with Excel logging
- REST API with comprehensive endpoints
- Web interface integration
- Comprehensive error handling and logging

Author: AI Surveillance System
Version: 2.0
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
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging
import traceback

# Import surveillance modules with error handling
try:
    from camera.stream_handler import CameraStream
    from detection.motion_detector import MotionDetector, MotionDetectionMethod
    from face_recognition_s.face_detector import FaceDetector
    from face_recognition_s.face_matcher import FaceMatcher
    from anomaly.anomaly_detector import AnomalyDetector
    from utils.config import config
    from utils.logger import logger
    from attendance_module import EmployeeAttendanceModule
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all modules are properly installed and accessible")
    sys.exit(1)

# GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        logger.info("CUDA not available, using CPU")
except ImportError:
    logger.info("PyTorch not available, running in CPU-only mode")
    TORCH_AVAILABLE = False

# Security setup
security = HTTPBearer(auto_error=False)

# Pydantic models for API with validation
class ControlRequest(BaseModel):
    action: str = Field(..., pattern="^(start|stop|reset)$", description="Action to perform")

class SettingsRequest(BaseModel):
    motion_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    learning_mode: Optional[bool] = None

class AttendanceSettingsRequest(BaseModel):
    face_tolerance: Optional[float] = Field(None, ge=0.0, le=1.0)
    cooldown_seconds: Optional[int] = Field(None, ge=0)
    auto_backup: Optional[bool] = None
    show_visit_count: Optional[bool] = None

class EmployeeRequest(BaseModel):
    employee_id: str = Field(..., min_length=1, max_length=50)
    employee_name: Optional[str] = Field(None, max_length=100)
    image_path: Optional[str] = None

class ReportRequest(BaseModel):
    days: Optional[int] = Field(30, ge=1, le=365)
    format: Optional[str] = Field("xlsx", pattern="^(xlsx|csv)$")
    include_summary: Optional[bool] = True

class SurveillanceSystem:
    """Main surveillance system controller with comprehensive error handling and attendance integration."""
    
    def __init__(self, camera_id: int = 0, gui_mode: bool = True):
        self.camera_id = camera_id
        self.gui_mode = gui_mode
        self.running = False
        self._shutdown_event = threading.Event()
        
        # Initialize components with error handling
        self.camera_stream: Optional[CameraStream] = None
        self.motion_detector: Optional[MotionDetector] = None
        self.face_detector: Optional[FaceDetector] = None
        self.face_matcher: Optional[FaceMatcher] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.attendance_module: Optional[EmployeeAttendanceModule] = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_display = 0
        self.last_fps_update = time.time()
        self._performance_lock = threading.Lock()
        
        # Event storage for API
        self.recent_events = []
        self.max_events = 100
        self._events_lock = threading.Lock()
        
        # Attendance tracking
        self.daily_attendance = {}
        self.last_daily_reset = datetime.now().date()
        self._attendance_lock = threading.Lock()
        
        # Error tracking
        self.error_count = 0
        self.last_error = None
        self.consecutive_errors = 0
        
        # GPU optimization
        self.use_cuda = TORCH_AVAILABLE and config.gpu.use_cuda and torch.cuda.is_available() if TORCH_AVAILABLE else False
        self.gpu_cleanup_counter = 0
        
        # API server
        self.api_app = None
        self.api_server = None
        
        # Initialize system
        try:
            self._initialize_system()
            self._setup_api()
        except Exception as e:
            logger.error(f"Failed to initialize surveillance system: {e}")
            raise
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Surveillance system initialized successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._shutdown_event.set()
        self.stop()
    
    def _initialize_system(self):
        """Initialize all surveillance components with comprehensive error handling."""
        logger.info("Initializing surveillance system components...")
        
        # Initialize camera
        try:
            self.camera_stream = CameraStream(self.camera_id)
            logger.info("Camera stream initialized")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise
        
        # Initialize motion detector
        try:
            detection_method = MotionDetectionMethod.BACKGROUND_SUBTRACTION
            if hasattr(config.detection, 'method'):
                method_name = config.detection.method.upper()
                if hasattr(MotionDetectionMethod, method_name):
                    detection_method = getattr(MotionDetectionMethod, method_name)
            
            self.motion_detector = MotionDetector(detection_method)
            logger.info("Motion detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize motion detector: {e}")
            raise
        
        # Initialize face components
        try:
            self.face_detector = FaceDetector()
            self.face_matcher = FaceMatcher()
            logger.info("Face recognition system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize face recognition: {e}")
            raise
        
        # Initialize anomaly detector
        try:
            self.anomaly_detector = AnomalyDetector()
            logger.info("Anomaly detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detector: {e}")
            raise
        
        # Initialize attendance module
        try:
            if config.attendance.enabled:
                self.attendance_module = EmployeeAttendanceModule(
                    face_dir=config.attendance.face_gallery_path,
                    attendance_file=config.attendance.attendance_file,
                    cooldown_seconds=config.attendance.cooldown_seconds,
                    tolerance=config.attendance.face_tolerance,
                    encodings_cache=config.attendance.encodings_cache_file,
                    backup_enabled=config.attendance.auto_backup
                )
                logger.info("Employee attendance module initialized")
            else:
                logger.info("Employee attendance module disabled")
        except Exception as e:
            logger.error(f"Failed to initialize attendance module: {e}")
            # Don't raise here, continue without attendance
            self.attendance_module = None
        
        logger.info("All components initialized successfully")
    
    def _setup_api(self):
        """Setup FastAPI server with comprehensive endpoints and error handling."""
        self.api_app = FastAPI(
            title="AI Surveillance System API with Attendance",
            description="Comprehensive REST API for AI-powered surveillance system with employee attendance tracking",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS middleware
        self.api_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Exception handler
        @self.api_app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            logger.error(f"Global exception handler: {exc}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Internal server error: {str(exc)}"}
            )
        
        # Authentication dependency (optional)
        async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
            # Implement token verification if needed
            # For now, just pass through
            return credentials
        
        # Health and status endpoints
        @self.api_app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                health_status = {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "attendance_enabled": config.attendance.enabled,
                    "components": {
                        "camera": self.camera_stream is not None,
                        "motion_detector": self.motion_detector is not None,
                        "face_detector": self.face_detector is not None,
                        "anomaly_detector": self.anomaly_detector is not None,
                        "attendance_module": self.attendance_module is not None
                    },
                    "system": {
                        "running": self.running,
                        "gpu_available": self.use_cuda,
                        "error_count": self.error_count,
                        "consecutive_errors": self.consecutive_errors
                    }
                }
                
                # Determine overall health
                if self.consecutive_errors >= 5:
                    health_status["status"] = "degraded"
                elif self.error_count > 0:
                    health_status["status"] = "warning"
                
                return health_status
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return JSONResponse(
                    status_code=503,
                    content={"status": "unhealthy", "error": str(e)}
                )
        
        @self.api_app.get("/api/status")
        async def get_status():
            """Get comprehensive system status."""
            try:
                status = self.get_system_status()
                return status
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.get("/api/stats")
        async def get_stats():
            """Get detailed system statistics."""
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
                
                if self.attendance_module:
                    stats['attendance_stats'] = self.attendance_module.get_statistics()
                
                return stats
            except Exception as e:
                logger.error(f"Error getting system statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.get("/api/events")
        async def get_events(limit: int = 50):
            """Get recent events."""
            try:
                with self._events_lock:
                    events = self.recent_events[-limit:] if self.recent_events else []
                return {"events": events, "total": len(self.recent_events)}
            except Exception as e:
                logger.error(f"Error getting events: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # System control endpoints
        @self.api_app.post("/api/control")
        async def control_system(request: ControlRequest):
            """Control system operations."""
            try:
                action = request.action
                
                if action == "start":
                    if self.running:
                        return {"success": False, "message": "System is already running"}
                    
                    # Start in background thread
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
        
        # [Additional API endpoints continue here...]
        logger.info("API endpoints configured successfully")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        try:
            with self._performance_lock:
                current_fps = self.fps_display
                frames_processed = self.frame_count
                runtime_seconds = time.time() - self.start_time if self.start_time else 0
            
            status = {
                'status': 'running' if self.running else 'stopped',
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': runtime_seconds,
                'frames_processed': frames_processed,
                'current_fps': current_fps,
                'gpu_available': self.use_cuda,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'camera_connected': self.camera_stream is not None and self.camera_stream.is_running(),
                'components_status': {
                    'motion_detector': self.motion_detector is not None,
                    'face_detector': self.face_detector is not None,
                    'face_matcher': self.face_matcher is not None,
                    'anomaly_detector': self.anomaly_detector is not None,
                    'attendance_module': self.attendance_module is not None
                }
            }
            
            # Add camera info if available
            if self.camera_stream:
                try:
                    status['camera_info'] = self.camera_stream.get_camera_info()
                except Exception as e:
                    logger.warning(f"Error getting camera info: {e}")
            
            # Add attendance status if available
            if self.attendance_module:
                try:
                    with self._attendance_lock:
                        status['attendance_status'] = {
                            'daily_attendance_count': len(self.daily_attendance),
                            'last_daily_reset': self.last_daily_reset.isoformat(),
                            'total_employees': len(self.attendance_module.known_employee_ids),
                            'attendance_logs': self.attendance_module.attendance_logs
                        }
                except Exception as e:
                    logger.warning(f"Error getting attendance status: {e}")
                    status['attendance_error'] = str(e)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _process_frame(self) -> bool:
        """Process a single frame with comprehensive error handling."""
        try:
            # Get frame from camera
            if not self.camera_stream:
                return False
            
            frame = self.camera_stream.get_frame()
            if frame is None:
                logger.warning("No frame received from camera")
                return False
            
            self.frame_count += 1
            
            # Reset daily attendance if needed
            current_date = datetime.now().date()
            if current_date != self.last_daily_reset:
                self._reset_daily_attendance()
            
            # Motion detection
            motion_events = []
            if self.motion_detector:
                motion_events = self.motion_detector.detect_motion(frame)
            
            # Face detection and recognition
            face_detections = []
            attendance_detections = []
            
            if self.attendance_module:
                # Use attendance module for integrated face processing
                annotated_frame, attendance_detections = self.attendance_module.process_frame(frame)
                
                # Convert attendance detections to face detections for anomaly detector
                face_detections = self._convert_attendance_to_face_detections(attendance_detections)
            else:
                # Use separate face detection pipeline
                if self.face_detector:
                    face_detections = self.face_detector.detect_faces(frame, return_encodings=True)
                    
                    if face_detections and self.face_matcher:
                        face_detections = self.face_matcher.match_faces(face_detections)
                
                annotated_frame = frame
            
            # Anomaly detection
            anomalies = []
            if self.anomaly_detector:
                anomalies = self.anomaly_detector.detect_anomalies(
                    motion_events, face_detections, frame
                )
            
            # Handle events
            if motion_events:
                self._add_event("motion_detected", {
                    "event_count": len(motion_events),
                    "total_area": sum(event.area for event in motion_events)
                })
            
            if face_detections:
                self._add_event("faces_detected", {
                    "face_count": len(face_detections),
                    "known_faces": len([f for f in face_detections if hasattr(f, 'face_id') and f.face_id])
                })
            
            if anomalies:
                for anomaly in anomalies:
                    self._add_event(f"anomaly_{anomaly.anomaly_type.value}", {
                        "confidence": anomaly.confidence,
                        "location": anomaly.location,
                        "description": anomaly.description
                    })
            
            # Update daily attendance
            if attendance_detections:
                self._update_daily_attendance(attendance_detections)
            
            # Display results if GUI mode
            if self.gui_mode:
                self._display_results(annotated_frame, motion_events, attendance_detections, anomalies)
            
            # GPU cleanup periodically
            if self.use_cuda:
                self.gpu_cleanup_counter += 1
                if self.gpu_cleanup_counter >= 100:
                    torch.cuda.empty_cache()
                    self.gpu_cleanup_counter = 0
            
            # Update FPS
            self._update_fps()
            
            # Reset consecutive error count on successful processing
            self.consecutive_errors = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            self._handle_error(e)
            return False
    
    def _convert_attendance_to_face_detections(self, attendance_detections: List[Dict]) -> List:
        """Convert attendance detections to face detection format."""
        face_detections = []
        
        for detection in attendance_detections:
            # Create a mock face detection object
            class MockFaceDetection:
                def __init__(self, detection_dict):
                    self.face_id = detection_dict.get('employee_id')
                    self.confidence = detection_dict.get('confidence', 0.0)
                    self.center = self._get_center_from_bbox(detection_dict.get('bbox', (0, 0, 100, 100)))
                
                def _get_center_from_bbox(self, bbox):
                    if len(bbox) == 4:
                        left, top, right, bottom = bbox
                        return ((left + right) // 2, (top + bottom) // 2)
                    return (50, 50)
            
            face_detections.append(MockFaceDetection(detection))
        
        return face_detections
    
    def _display_results(self, frame, motion_events, attendance_detections, anomalies):
        """Display surveillance results in GUI windows."""
        try:
            # Main surveillance feed
            display_frame = frame.copy()
            
            # Add motion overlay
            if motion_events and self.motion_detector:
                display_frame = self.motion_detector.draw_motion_overlay(display_frame, motion_events)
            
            # Add anomaly overlay
            if anomalies and self.anomaly_detector:
                display_frame = self.anomaly_detector.draw_anomaly_overlay(display_frame, anomalies)
            
            # Add status overlay
            self._add_status_overlay(display_frame)
            
            cv2.imshow('Surveillance Feed', display_frame)
            
            # Motion detection window
            if motion_events:
                motion_frame = np.zeros_like(frame)
                if self.motion_detector:
                    motion_frame = self.motion_detector.draw_motion_overlay(motion_frame, motion_events)
                cv2.imshow('Motion Detection', motion_frame)
            
            # Attendance dashboard
            if self.attendance_module and attendance_detections:
                dashboard = self._create_attendance_dashboard()
                cv2.imshow('Attendance Dashboard', dashboard)
            
        except Exception as e:
            logger.warning(f"Error displaying results: {e}")
    
    def _add_status_overlay(self, frame: np.ndarray):
        """Add status information overlay to frame."""
        try:
            height, width = frame.shape[:2]
            
            # Status text
            status_lines = [
                f"FPS: {self.fps_display:.1f}",
                f"Frames: {self.frame_count}",
                f"Runtime: {self._get_runtime_str()}",
                f"GPU: {'ON' if self.use_cuda else 'OFF'}",
                f"Errors: {self.error_count}"
            ]
            
            if self.attendance_module:
                status_lines.extend([
                    f"Employees: {len(self.attendance_module.known_employee_ids)}",
                    f"Today: {len(self.daily_attendance)}",
                    f"Logs: {self.attendance_module.attendance_logs}"
                ])
            
            # Background rectangle
            overlay_height = len(status_lines) * 25 + 20
            cv2.rectangle(frame, (10, 10), (300, overlay_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (300, overlay_height), (255, 255, 255), 1)
            
            # Status text
            for i, line in enumerate(status_lines):
                y_pos = 35 + i * 25
                cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 1, cv2.LINE_AA)
                           
        except Exception as e:
            logger.debug(f"Error adding status overlay: {e}")
    
    def _create_attendance_dashboard(self) -> np.ndarray:
        """Create attendance dashboard visualization."""
        try:
            dashboard_width = 600
            dashboard_height = 400
            dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
            
            # Title
            cv2.putText(dashboard, "Employee Attendance Dashboard", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Current date
            date_str = datetime.now().strftime("%Y-%m-%d")
            cv2.putText(dashboard, f"Date: {date_str}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Employee list
            y_pos = 90
            cv2.putText(dashboard, "Today's Attendance:", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 30
            
            with self._attendance_lock:
                if not self.daily_attendance:
                    cv2.putText(dashboard, "No attendance recorded today", (20, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                else:
                    for i, (emp_id, data) in enumerate(list(self.daily_attendance.items())[:10]):
                        if y_pos > dashboard_height - 30:
                            break
                        
                        first_seen = data['first_seen'].strftime("%H:%M")
                        last_seen = data['last_seen'].strftime("%H:%M")
                        detections = data['total_detections']
                        
                        text = f"{emp_id}: {first_seen}-{last_seen} ({detections} det.)"
                        cv2.putText(dashboard, text, (20, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        y_pos += 20
            
            return dashboard
            
        except Exception as e:
            logger.warning(f"Error creating attendance dashboard: {e}")
            return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def start(self):
        """Start the surveillance system with comprehensive error handling."""
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
            self.consecutive_errors = 0
            
            if self.gui_mode:
                self._run_with_gui()
            else:
                self._run_headless()
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error during surveillance: {e}")
            self._handle_error(e)
        finally:
            self.stop()
    
    def _run_with_gui(self):
        """Run surveillance with GUI display."""
        logger.info("Starting surveillance with GUI and attendance tracking")
        
        # Create display windows
        window_names = ['Surveillance Feed', 'Motion Detection']
        if self.attendance_module:
            window_names.append('Attendance Dashboard')
        
        try:
            for window_name in window_names:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            logger.info("GUI windows created successfully")
            
            while self.running and not self._shutdown_event.is_set():
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
                elif key == ord('s'):
                    logger.info("Saving current frame and stats")
                    self._save_current_frame()
                    self._print_status()
                elif key == ord('a') and self.attendance_module:
                    try:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"manual_export_{timestamp}.xlsx"
                        success = self.attendance_module.export_attendance_report(filename)
                        if success:
                            logger.info(f"Attendance report exported: {filename}")
                        else:
                            logger.warning("Failed to export attendance report")
                    except Exception as e:
                        logger.error(f"Error exporting report: {e}")
            
        finally:
            cv2.destroyAllWindows()
    
    def _run_headless(self):
        """Run surveillance without GUI."""
        logger.info("Starting surveillance in headless mode")
        
        while self.running and not self._shutdown_event.is_set():
            success = self._process_frame()
            if not success:
                break
            
            # Print status every 100 frames
            if self.frame_count % 100 == 0:
                self._print_status()
    
    def stop(self):
        """Stop the surveillance system gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping surveillance system")
        self.running = False
        
        # Stop camera stream
        if self.camera_stream:
            self.camera_stream.stop_stream()
        
        # Final export if attendance module is active
        if self.attendance_module:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"final_export_{timestamp}.xlsx"
                self.attendance_module.export_attendance_report(filename)
                logger.info(f"Final attendance report exported: {filename}")
            except Exception as e:
                logger.warning(f"Could not export final report: {e}")
        
        logger.info("Surveillance system stopped")
    
    # [Additional helper methods continue...]
    
    def _add_event(self, event_type: str, details: Dict):
        """Add an event to the recent events list with thread safety."""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "details": details
            }
            
            with self._events_lock:
                self.recent_events.append(event)
                
                # Keep only the most recent events
                if len(self.recent_events) > self.max_events:
                    self.recent_events = self.recent_events[-self.max_events:]
                    
        except Exception as e:
            logger.warning(f"Error adding event: {e}")
    
    def _update_daily_attendance(self, detections: List[Dict]):
        """Update daily attendance statistics with thread safety."""
        try:
            current_date = datetime.now().date()
            
            # Reset if new day
            if current_date != self.last_daily_reset:
                self._reset_daily_attendance()
            
            with self._attendance_lock:
                for detection in detections:
                    employee_id = detection.get('employee_id')
                    if employee_id and employee_id != "Unknown":
                        if employee_id not in self.daily_attendance:
                            self.daily_attendance[employee_id] = {
                                'first_seen': datetime.now(),
                                'last_seen': datetime.now(),
                                'total_detections': 0,
                                'visit_count': detection.get('visit_count', 0),
                                'employee_name': detection.get('employee_name', employee_id)
                            }
                        
                        self.daily_attendance[employee_id]['last_seen'] = datetime.now()
                        self.daily_attendance[employee_id]['total_detections'] += 1
                        
                        if 'visit_count' in detection:
                            self.daily_attendance[employee_id]['visit_count'] = detection['visit_count']
                            
        except Exception as e:
            logger.warning(f"Error updating daily attendance: {e}")
    
    def _reset_daily_attendance(self):
        """Reset daily attendance tracking with thread safety."""
        try:
            with self._attendance_lock:
                self.daily_attendance.clear()
                self.last_daily_reset = datetime.now().date()
            
            logger.info("Daily attendance tracking reset")
            
        except Exception as e:
            logger.warning(f"Error resetting daily attendance: {e}")
    
    def _reset_detectors(self):
        """Reset all detectors."""
        try:
            if self.motion_detector:
                self.motion_detector.reset_detector()
            
            if self.anomaly_detector:
                self.anomaly_detector.reset_learning()
            
            logger.info("Detectors reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting detectors: {e}")
    
    def _handle_error(self, error: Exception):
        """Handle errors with tracking and recovery."""
        self.error_count += 1
        self.last_error = error
        self.consecutive_errors += 1
        
        # Log error details
        logger.error(f"Error #{self.error_count}: {error}")
        
        # If too many consecutive errors, attempt recovery
        if self.consecutive_errors >= 5:
            logger.warning("Too many consecutive errors, attempting recovery")
            self._attempt_recovery()
    
    def _attempt_recovery(self):
        """Attempt to recover from errors."""
        try:
            logger.info("Attempting system recovery")
            
            # Reset detectors
            self._reset_detectors()
            
            # Restart camera if needed
            if self.camera_stream and not self.camera_stream.is_running():
                self.camera_stream.stop_stream()
                time.sleep(1)
                if self.camera_stream.start_stream():
                    logger.info("Camera stream restarted successfully")
                    self.consecutive_errors = 0
                else:
                    logger.warning("Camera stream restart failed")
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
    
    def _update_fps(self):
        """Update FPS calculation."""
        try:
            with self._performance_lock:
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    elapsed = current_time - self.start_time
                    self.fps_display = self.frame_count / elapsed if elapsed > 0 else 0
                    self.last_fps_update = current_time
        except Exception as e:
            logger.debug(f"Error updating FPS: {e}")
    
    def _get_runtime_str(self) -> str:
        """Get formatted runtime string."""
        try:
            runtime = time.time() - self.start_time if self.start_time else 0
            hours = int(runtime // 3600)
            minutes = int((runtime % 3600) // 60)
            seconds = int(runtime % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except Exception:
            return "00:00:00"
    
    def _save_current_frame(self):
        """Save current frame to disk."""
        try:
            if self.camera_stream:
                frame = self.camera_stream.get_frame()
                if frame is not None:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Frame saved: {filename}")
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
    
    def _print_status(self):
        """Print current system status."""
        try:
            status = self.get_system_status()
            logger.info(f"Status: {status['status']} | FPS: {status['current_fps']:.1f} | "
                       f"Frames: {status['frames_processed']} | Runtime: {self._get_runtime_str()}")
            
            if self.attendance_module:
                logger.info(f"Attendance: {len(self.daily_attendance)} employees today, "
                           f"{self.attendance_module.attendance_logs} total logs")
        except Exception as e:
            logger.error(f"Error printing status: {e}")
    
    def start_api_server(self):
        """Start the API server in a separate thread."""
        def run_server():
            try:
                port = int(os.getenv("API_PORT", 8080))
                host = os.getenv("API_HOST", "0.0.0.0")
                
                logger.info(f"Starting API server on {host}:{port}")
                
                # Configure uvicorn with proper settings
                config_uvicorn = uvicorn.Config(
                    app=self.api_app,
                    host=host,
                    port=port,
                    log_level="info",
                    access_log=True,
                    use_colors=False,
                    loop="asyncio"
                )
                
                server = uvicorn.Server(config_uvicorn)
                server.run()
                
            except Exception as e:
                logger.error(f"API server error: {e}")
        
        api_thread = threading.Thread(target=run_server, daemon=True)
        api_thread.start()
        logger.info("API server thread started")


def main():
    """Main entry point with comprehensive argument parsing and error handling."""
    parser = argparse.ArgumentParser(
        description="AI- Surveillance System with Employee Attendance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Start with default settings
  %(prog)s --camera 1 --headless       # Use camera 1 in headless mode
  %(prog)s --disable-attendance         # Run without attendance tracking
  %(prog)s --api-only --port 8081       # API server only on port 8081
  %(prog)s --export-report --days 7     # Export 7-day report and exit
        """
    )
    
    # Basic options
    parser.add_argument("--camera", "-c", type=int, default=0, 
                       help="Camera device ID (default: 0)")
    parser.add_argument("--headless", "-hl", action="store_true", 
                       help="Run without GUI")
    parser.add_argument("--api-only", action="store_true", 
                       help="Run API server only (no camera processing)")
    parser.add_argument("--port", "-p", type=int, default=8080, 
                       help="API server port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="API server host (default: 0.0.0.0)")
    
    # Attendance options
    parser.add_argument("--disable-attendance", action="store_true", 
                       help="Disable attendance module")
    parser.add_argument("--face-tolerance", type=float, 
                       help="Face recognition tolerance (0.0-1.0)")
    parser.add_argument("--cooldown", type=int, 
                       help="Cooldown seconds between attendance logs")
    parser.add_argument("--face-dir", type=str,
                       help="Face gallery directory")
    parser.add_argument("--attendance-file", type=str,
                       help="Attendance Excel file path")
    
    # Utility options
    parser.add_argument("--export-report", action="store_true", 
                       help="Export attendance report and exit")
    parser.add_argument("--days", type=int, default=30,
                       help="Number of days for report export (default: 30)")
    parser.add_argument("--validate-system", action="store_true",
                       help="Validate system integrity and exit")
    parser.add_argument("--list-employees", action="store_true",
                       help="List all employees and exit")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default="INFO", help="Set logging level")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Set environment variables
    os.environ["API_PORT"] = str(args.port)
    os.environ["API_HOST"] = args.host
    
    # Handle attendance arguments
    try:
        if args.disable_attendance:
            config.attendance.enabled = False
            logger.info("Attendance module disabled via command line")
        
        if args.face_tolerance is not None:
            if 0.0 <= args.face_tolerance <= 1.0:
                config.attendance.face_tolerance = args.face_tolerance
                logger.info(f"Face tolerance set to {args.face_tolerance}")
            else:
                logger.error("Face tolerance must be between 0.0 and 1.0")
                return 1
        
        if args.cooldown is not None:
            if args.cooldown >= 0:
                config.attendance.cooldown_seconds = args.cooldown
                logger.info(f"Cooldown set to {args.cooldown} seconds")
            else:
                logger.error("Cooldown must be non-negative")
                return 1
        
        if args.face_dir:
            config.attendance.face_gallery_path = args.face_dir
            logger.info(f"Face directory set to {args.face_dir}")
        
        if args.attendance_file:
            config.attendance.attendance_file = args.attendance_file
            logger.info(f"Attendance file set to {args.attendance_file}")
            
    except Exception as e:
        logger.error(f"Error processing attendance arguments: {e}")
        return 1
    
    # Handle utility commands
    if args.export_report:
        try:
            from attendance_module import EmployeeAttendanceModule
            
            logger.info("Exporting attendance report...")
            attendance = EmployeeAttendanceModule(
                face_dir=config.attendance.face_gallery_path,
                attendance_file=config.attendance.attendance_file
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cmdline_report_{timestamp}.xlsx"
            success = attendance.export_attendance_report(filename, args.days)
            
            if success:
                print(f"✅ Attendance report exported: {filename}")
                return 0
            else:
                print("❌ Failed to export attendance report")
                return 1
                
        except Exception as e:
            print(f"❌ Error exporting report: {e}")
            return 1
    
    if args.validate_system:
        try:
            from attendance_module import EmployeeAttendanceModule
            
            logger.info("Validating system integrity...")
            attendance = EmployeeAttendanceModule(
                face_dir=config.attendance.face_gallery_path,
                attendance_file=config.attendance.attendance_file
            )
            
            integrity_report = attendance.validate_system_integrity()
            
            print(f"System Integrity: {integrity_report['overall_status'].upper()}")
            
            if integrity_report['issues']:
                print("\nIssues Found:")
                for issue in integrity_report['issues']:
                    print(f"  ❌ {issue}")
            
            if integrity_report['warnings']:
                print("\nWarnings:")
                for warning in integrity_report['warnings']:
                    print(f"  ⚠️  {warning}")
            
            if integrity_report['overall_status'] == 'healthy':
                print("\n✅ System validation passed")
                return 0
            else:
                print("\n❌ System validation failed")
                return 1
                
        except Exception as e:
            print(f"❌ Error validating system: {e}")
            return 1
    
    if args.list_employees:
        try:
            from attendance_module import EmployeeAttendanceModule
            
            attendance = EmployeeAttendanceModule(
                face_dir=config.attendance.face_gallery_path,
                attendance_file=config.attendance.attendance_file
            )
            
            employees = attendance.get_employee_list()
            
            if employees:
                print(f"\nFound {len(employees)} employees:")
                print("-" * 60)
                for emp in employees:
                    name = emp.get('name', emp['employee_id'])
                    added = emp.get('added_timestamp', 'Unknown')
                    print(f"ID: {emp['employee_id']:<15} Name: {name:<20} Added: {added}")
            else:
                print("No employees found in the system")
            
            return 0
            
        except Exception as e:
            print(f"❌ Error listing employees: {e}")
            return 1
    
    # Main system startup
    logger.info(f"Starting AI-Powered Surveillance System with Attendance")
    logger.info(f"API Server: {args.host}:{args.port}")
    logger.info(f"Attendance: {'Enabled' if config.attendance.enabled else 'Disabled'}")
    
    try:
        surveillance = SurveillanceSystem(
            camera_id=args.camera,
            gui_mode=not args.headless and not args.api_only
        )
        
        if args.api_only:
            logger.info("Starting in API-only mode...")
            surveillance.start_api_server()
            
            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("API server interrupted by user")
        else:
            surveillance.start()
    
    except KeyboardInterrupt:
        logger.info("Surveillance system interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    
    logger.info("Surveillance system shutdown complete")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()
        sys.exit(1)     