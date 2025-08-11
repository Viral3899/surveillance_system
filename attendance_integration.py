"""
Attendance Integration Script for AI Surveillance System
=======================================================

This script demonstrates how to integrate the Employee Attendance Module
with the existing AI surveillance system.

Features:
- Seamless integration with existing surveillance pipeline
- Combined face recognition and attendance tracking
- Enhanced API endpoints for attendance data
- Real-time attendance updates in the web interface

Author: AI Surveillance System
Version: 1.0
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

# Import existing surveillance modules
try:
    from camera.stream_handler import CameraStream
    from face_recognition_s.face_detector import FaceDetection
    from utils.config import config
    from utils.logger import logger
except ImportError:
    # Fallback for standalone testing
    print("Warning: Surveillance modules not found. Running in standalone mode.")

# Import the attendance module
from attendance_module import EmployeeAttendanceModule

class AttendanceIntegratedSurveillance:
    """
    Enhanced surveillance system with integrated employee attendance tracking.
    
    This class extends the existing surveillance system to include
    employee attendance functionality while maintaining all existing features.
    """
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize the integrated surveillance system.
        
        Args:
            camera_id (int): Camera device ID
        """
        self.camera_id = camera_id
        self.running = False
        
        # Initialize attendance module
        self.attendance_module = EmployeeAttendanceModule(
            face_dir="faces",
            attendance_file="attendance.xlsx",
            cooldown_seconds=5,
            tolerance=0.5
        )
        
        # Camera stream
        self.camera_stream: Optional[CameraStream] = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_display = 0
        self.last_fps_update = time.time()
        
        # Attendance statistics
        self.daily_attendance = {}
        self.last_daily_reset = datetime.now().date()
        
        logger.info("Attendance-integrated surveillance system initialized")
    
    def start(self, gui_mode: bool = True):
        """
        Start the integrated surveillance system.
        
        Args:
            gui_mode (bool): Whether to show GUI windows
        """
        try:
            # Initialize camera
            if hasattr(self, 'camera_stream') and self.camera_stream is None:
                # Use existing camera stream handler if available
                from camera.stream_handler import CameraStream
                self.camera_stream = CameraStream(self.camera_id)
                if not self.camera_stream.start_stream():
                    raise RuntimeError("Failed to start camera stream")
            else:
                # Fallback to OpenCV camera
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
            self.running = True
            self.start_time = time.time()
            
            logger.info("Starting attendance-integrated surveillance")
            
            if gui_mode:
                self._run_with_gui()
            else:
                self._run_headless()
                
        except Exception as e:
            logger.error(f"Error starting surveillance: {e}")
            self.stop()
    
    def _run_with_gui(self):
        """Run surveillance with GUI display."""
        cv2.namedWindow('Attendance Surveillance', cv2.WINDOW_NORMAL)
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
                    self._reset_daily_attendance()
                elif key == ord('s'):
                    self._export_attendance_report()
                elif key == ord('h'):
                    self._print_help()
                    
        finally:
            cv2.destroyAllWindows()
    
    def _run_headless(self):
        """Run surveillance without GUI."""
        logger.info("Starting attendance surveillance in headless mode")
        
        while self.running:
            success = self._process_frame()
            if not success:
                break
            
            # Print status every 100 frames
            if self.frame_count % 100 == 0:
                self._print_status()
    
    def _process_frame(self) -> bool:
        """Process a single frame with attendance tracking."""
        try:
            # Get frame from camera
            frame = self._get_camera_frame()
            if frame is None:
                logger.warning("No frame received from camera")
                return False
            
            self.frame_count += 1
            
            # Reset daily attendance if needed
            if datetime.now().date() != self.last_daily_reset:
                self._reset_daily_attendance()
            
            # Process frame with attendance module
            annotated_frame, detections = self.attendance_module.process_frame(frame)
            
            # Update daily attendance statistics
            self._update_daily_stats(detections)
            
            # Display results if GUI mode
            if hasattr(self, 'cap') or self.camera_stream:
                self._display_results(annotated_frame, detections)
            
            # Update FPS
            self._update_fps()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return False
    
    def _get_camera_frame(self) -> Optional[np.ndarray]:
        """Get frame from camera source."""
        if self.camera_stream:
            return self.camera_stream.get_frame()
        elif hasattr(self, 'cap'):
            ret, frame = self.cap.read()
            return frame if ret else None
        return None
    
    def _update_daily_stats(self, detections: List[Dict]):
        """Update daily attendance statistics."""
        current_date = datetime.now().date()
        
        for detection in detections:
            employee_id = detection['employee_id']
            if employee_id != "Unknown":
                if employee_id not in self.daily_attendance:
                    self.daily_attendance[employee_id] = {
                        'first_seen': datetime.now(),
                        'last_seen': datetime.now(),
                        'total_detections': 0
                    }
                
                self.daily_attendance[employee_id]['last_seen'] = datetime.now()
                self.daily_attendance[employee_id]['total_detections'] += 1
    
    def _display_results(self, annotated_frame: np.ndarray, detections: List[Dict]):
        """Display surveillance results with attendance information."""
        # Main surveillance feed
        display_frame = annotated_frame.copy()
        self._add_status_overlay(display_frame)
        cv2.imshow('Attendance Surveillance', display_frame)
        
        # Attendance dashboard
        dashboard = self._create_attendance_dashboard()
        cv2.imshow('Attendance Dashboard', dashboard)
    
    def _add_status_overlay(self, frame: np.ndarray):
        """Add status information overlay to frame."""
        height, width = frame.shape[:2]
        
        # Status text
        status_lines = [
            f"FPS: {self.fps_display:.1f}",
            f"Frames: {self.frame_count}",
            f"Runtime: {self._get_runtime_str()}",
            f"Employees Today: {len(self.daily_attendance)}",
            f"Total Employees: {len(self.attendance_module.known_employee_ids)}",
            f"Attendance Logs: {self.attendance_module.attendance_logs}"
        ]
        
        # Background rectangle
        overlay_height = len(status_lines) * 25 + 20
        cv2.rectangle(frame, (10, 10), (300, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, overlay_height), (255, 255, 255), 1)
        
        # Status text
        for i, line in enumerate(status_lines):
            y_pos = 35 + i * 25
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    def _create_attendance_dashboard(self) -> np.ndarray:
        """Create attendance dashboard visualization."""
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
        
        if not self.daily_attendance:
            cv2.putText(dashboard, "No attendance recorded today", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        else:
            for i, (emp_id, data) in enumerate(self.daily_attendance.items()):
                if y_pos > dashboard_height - 30:
                    break
                
                first_seen = data['first_seen'].strftime("%H:%M")
                last_seen = data['last_seen'].strftime("%H:%M")
                detections = data['total_detections']
                
                text = f"{emp_id}: {first_seen}-{last_seen} ({detections} det.)"
                cv2.putText(dashboard, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_pos += 20
        
        # Statistics
        stats_y = dashboard_height - 80
        cv2.putText(dashboard, "Statistics:", (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        stats = self.attendance_module.get_statistics()
        stats_text = [
            f"Total Employees: {stats['total_employees']}",
            f"Total Logs: {stats['total_attendance_logs']}",
            f"Active Today: {len(self.daily_attendance)}"
        ]
        
        for i, stat in enumerate(stats_text):
            cv2.putText(dashboard, stat, (20, stats_y + 25 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return dashboard
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            elapsed = current_time - self.start_time
            self.fps_display = self.frame_count / elapsed if elapsed > 0 else 0
            self.last_fps_update = current_time
    
    def _get_runtime_str(self) -> str:
        """Get formatted runtime string."""
        runtime = time.time() - self.start_time
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _reset_daily_attendance(self):
        """Reset daily attendance tracking."""
        self.daily_attendance.clear()
        self.last_daily_reset = datetime.now().date()
        logger.info("Daily attendance tracking reset")
    
    def _export_attendance_report(self):
        """Export attendance report."""
        try:
            filename = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            success = self.attendance_module.export_attendance_report(filename, days=30)
            if success:
                logger.info(f"Attendance report exported: {filename}")
            else:
                logger.error("Failed to export attendance report")
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
    
    def _print_status(self):
        """Print status information for headless mode."""
        runtime = self._get_runtime_str()
        current_fps = self.frame_count / (time.time() - self.start_time)
        
        print(f"\n--- Attendance Surveillance Status ---")
        print(f"Runtime: {runtime} | Frames: {self.frame_count} | FPS: {current_fps:.1f}")
        print(f"Employees Today: {len(self.daily_attendance)}")
        print(f"Total Attendance Logs: {self.attendance_module.attendance_logs}")
        
        if self.daily_attendance:
            print("Today's Attendance:")
            for emp_id, data in list(self.daily_attendance.items())[:5]:  # Show first 5
                first_seen = data['first_seen'].strftime("%H:%M")
                last_seen = data['last_seen'].strftime("%H:%M")
                print(f"  {emp_id}: {first_seen}-{last_seen}")
        
        print("-" * 50)
    
    def _print_help(self):
        """Print keyboard shortcuts help."""
        help_text = """
        Keyboard Shortcuts:
        q - Quit
        r - Reset daily attendance
        s - Export attendance report
        h - Show this help
        """
        print(help_text)
        logger.info("Help information displayed")
    
    def stop(self):
        """Stop the surveillance system."""
        if not self.running:
            return
        
        logger.info("Stopping attendance surveillance system")
        self.running = False
        
        # Stop camera
        if hasattr(self, 'cap'):
            self.cap.release()
        elif self.camera_stream:
            self.camera_stream.stop_stream()
        
        # Export final report
        try:
            self._export_attendance_report()
        except Exception as e:
            logger.warning(f"Could not export final report: {e}")
        
        logger.info("Attendance surveillance system stopped")
    
    def get_attendance_data(self) -> Dict[str, Any]:
        """Get comprehensive attendance data for API endpoints."""
        return {
            'daily_attendance': dict(self.daily_attendance),
            'module_stats': self.attendance_module.get_statistics(),
            'recent_summary': self.attendance_module.get_attendance_summary(days=7).to_dict('records'),
            'runtime_stats': {
                'frames_processed': self.frame_count,
                'fps': self.fps_display,
                'runtime_seconds': time.time() - self.start_time,
                'employees_today': len(self.daily_attendance)
            }
        }


# Enhanced API Server with Attendance Endpoints
class AttendanceAPI:
    """Enhanced API server with attendance endpoints."""
    
    def __init__(self, surveillance_system: AttendanceIntegratedSurveillance):
        self.surveillance = surveillance_system
        self.app = self._create_api()
    
    def _create_api(self):
        """Create FastAPI application with attendance endpoints."""
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import JSONResponse
        except ImportError:
            logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
            return None
        
        app = FastAPI(title="Attendance Surveillance API", version="1.0.0")
        
        @app.get("/api/attendance/status")
        async def get_attendance_status():
            """Get current attendance system status."""
            try:
                return self.surveillance.get_attendance_data()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/attendance/daily")
        async def get_daily_attendance():
            """Get today's attendance data."""
            try:
                return {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'attendance': dict(self.surveillance.daily_attendance)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/attendance/summary")
        async def get_attendance_summary(days: int = 7):
            """Get attendance summary for specified days."""
            try:
                summary = self.surveillance.attendance_module.get_attendance_summary(days)
                return {
                    'days': days,
                    'summary': summary.to_dict('records')
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/attendance/export")
        async def export_attendance_report(days: int = 30):
            """Export attendance report."""
            try:
                filename = f"api_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                success = self.surveillance.attendance_module.export_attendance_report(filename, days)
                return {
                    'success': success,
                    'filename': filename if success else None,
                    'message': 'Report exported successfully' if success else 'Export failed'
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/attendance/employees")
        async def get_employee_list():
            """Get list of all employees."""
            try:
                return {
                    'employees': self.surveillance.attendance_module.known_employee_ids,
                    'count': len(self.surveillance.attendance_module.known_employee_ids)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        return app


# Integration with existing surveillance system
def integrate_attendance_with_existing_system():
    """
    Example function showing how to integrate attendance module
    with the existing surveillance system.
    """
    try:
        # Import existing surveillance system
        from api_server import SurveillanceSystem
        
        class AttendanceEnhancedSurveillanceSystem(SurveillanceSystem):
            """Enhanced surveillance system with attendance capabilities."""
            
            def __init__(self, camera_id: int = 0, gui_mode: bool = True):
                super().__init__(camera_id, gui_mode)
                
                # Add attendance module
                self.attendance_module = EmployeeAttendanceModule(
                    face_dir="faces",
                    attendance_file="attendance.xlsx",
                    cooldown_seconds=5,
                    tolerance=0.5
                )
                
                # Enhanced API endpoints
                self._setup_attendance_api()
            
            def _setup_attendance_api(self):
                """Add attendance endpoints to existing API."""
                
                @self.api_app.get("/api/attendance/status")
                async def get_attendance_status():
                    return self.attendance_module.get_statistics()
                
                @self.api_app.get("/api/attendance/daily")
                async def get_daily_attendance():
                    summary = self.attendance_module.get_attendance_summary(days=1)
                    return summary.to_dict('records')
                
                @self.api_app.post("/api/attendance/export")
                async def export_report(days: int = 30):
                    filename = f"attendance_{datetime.now().strftime('%Y%m%d')}.xlsx"
                    success = self.attendance_module.export_attendance_report(filename, days)
                    return {'success': success, 'filename': filename}
            
            def _process_frame(self) -> bool:
                """Enhanced frame processing with attendance tracking."""
                try:
                    # Get frame from camera
                    if not self.camera_stream:
                        return False
                    
                    frame = self.camera_stream.get_frame()
                    if frame is None:
                        return False
                    
                    self.frame_count += 1
                    
                    # Original surveillance processing
                    motion_events = []
                    if self.motion_detector:
                        motion_events = self.motion_detector.detect_motion(frame)
                    
                    # Attendance processing
                    annotated_frame, attendance_detections = self.attendance_module.process_frame(frame)
                    
                    # Face detection for surveillance (if not handled by attendance)
                    face_detections = []
                    if self.face_detector and not attendance_detections:
                        face_detections = self.face_detector.detect_faces(frame, return_encodings=True)
                        if face_detections and self.face_matcher:
                            face_detections = self.face_matcher.match_faces(face_detections)
                    
                    # Anomaly detection
                    anomalies = []
                    if self.anomaly_detector:
                        anomalies = self.anomaly_detector.detect_anomalies(
                            motion_events, face_detections, frame
                        )
                    
                    # Handle anomalies
                    if anomalies:
                        self._handle_anomalies(frame, anomalies, motion_events, face_detections)
                    
                    # Display results
                    if self.gui_mode:
                        self._display_results(annotated_frame, motion_events, attendance_detections, anomalies)
                    
                    self._update_fps()
                    return True
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    return False
        
        return AttendanceEnhancedSurveillanceSystem
        
    except ImportError:
        logger.warning("Existing surveillance system not found. Using standalone integration.")
        return AttendanceIntegratedSurveillance


# Main function for standalone usage
def main():
    """Main function for standalone attendance surveillance."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Employee Attendance Surveillance System")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device ID")
    parser.add_argument("--headless", "-hl", action="store_true", help="Run without GUI")
    parser.add_argument("--face-dir", "-f", default="faces", help="Face gallery directory")
    parser.add_argument("--attendance-file", "-a", default="attendance.xlsx", help="Attendance Excel file")
    parser.add_argument("--cooldown", type=int, default=5, help="Cooldown seconds between logs")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Face recognition tolerance")
    
    args = parser.parse_args()
    
    try:
        # Create attendance module with custom parameters
        attendance_module = EmployeeAttendanceModule(
            face_dir=args.face_dir,
            attendance_file=args.attendance_file,
            cooldown_seconds=args.cooldown,
            tolerance=args.tolerance
        )
        
        # Create integrated surveillance system
        surveillance = AttendanceIntegratedSurveillance(camera_id=args.camera)
        surveillance.attendance_module = attendance_module
        
        # Start the system
        logger.info("Starting Employee Attendance Surveillance System")
        surveillance.start(gui_mode=not args.headless)
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    logger.info("Employee Attendance Surveillance System shutdown complete")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())