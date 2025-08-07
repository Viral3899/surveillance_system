#!/usr/bin/env python3
"""
Enhanced main surveillance system controller with integrated API server.
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any

# Import surveillance modules
from camera.stream_handler import CameraStream, MultiCameraManager
from detection.motion_detector import MotionDetector, MotionDetectionMethod
from face_recognition.face_detector import FaceDetector
from face_recognition.face_matcher import FaceMatcher
from anomaly.anomaly_detector import AnomalyDetector
from utils.config import config
from utils.logger import logger

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

class SurveillanceSystem:
    """Main surveillance system controller with API integration."""
    
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
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_display = 0
        self.last_fps_update = time.time()
        
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
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize surveillance system: {e}")
            raise
    
    def _setup_api(self):
        """Setup FastAPI server."""
        self.api_app = FastAPI(
            title="AI Surveillance System API",
            description="REST API for AI-powered surveillance system",
            version="1.0.0"
        )
        
        # CORS middleware
        self.api_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # API routes
        @self.api_app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": time.time()}
        
        @self.api_app.get("/api/status")
        async def get_status():
            try:
                return self.get_system_status()
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
                return stats
            except Exception as e:
                logger.error(f"Error getting system statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.api_app.post("/api/control")
        async def control_system(action: str):
            try:
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
            logger.warning("System is already running")
            return
        
        logger.info("Starting surveillance system.")
        
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
                    if self.anomaly_detector:
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
            
            # Face detection (only if motion detected or periodically)
            face_detections = []
            if (motion_events or self.frame_count % 10 == 0) and self.face_detector:
                face_detections = self.face_detector.detect_faces(frame, return_encodings=True)
                
                # Face matching
                if face_detections and self.face_matcher:
                    face_detections = self.face_matcher.match_faces(face_detections)
            
            # Anomaly detection
            anomalies = []
            if self.anomaly_detector:
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
        """Display surveillance results in GUI windows."""
        # Create display frames
        display_frame = frame.copy()
        motion_frame = frame.copy()
        
        # Draw motion detection overlay
        if motion_events and self.motion_detector:
            motion_frame = self.motion_detector.draw_motion_overlay(motion_frame, motion_events)
        
        # Draw face detection overlay
        if face_detections and self.face_detector:
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
    
    def _add_status_overlay(self, frame):
        """Add status information overlay to frame."""
        height, width = frame.shape[:2]
        
        # Status text
        status_lines = [
            f"FPS: {self.fps_display:.1f}",
            f"Frames: {self.frame_count}",
            f"Runtime: {self._get_runtime_str()}",
            f"GPU: {'ON' if self.use_cuda else 'OFF'}",
            f"Learning: {'ON' if self.anomaly_detector and self.anomaly_detector.is_learning else 'OFF'}"
        ]
        
        # Background rectangle
        overlay_height = len(status_lines) * 25 + 20
        cv2.rectangle(frame, (10, 10), (280, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (280, overlay_height), (255, 255, 255), 1)
        
        # Status text
        for i, line in enumerate(status_lines):
            y_pos = 35 + i * 25
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
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
        if self.camera_stream:
            frame = self.camera_stream.get_frame()
            if frame is not None:
                logger.save_image(frame, "manual_save")
    
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
            return {'status': 'stopped'}
        
        runtime = time.time() - self.start_time
        current_fps = self.frame_count / runtime if runtime > 0 else 0
        
        status = {
            'status': 'running',
            'runtime_seconds': runtime,
            'frames_processed': self.frame_count,
            'current_fps': current_fps,
            'gpu_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
            'learning_mode': self.anomaly_detector.is_learning if self.anomaly_detector else False,
            'camera_info': self.camera_stream.get_camera_info() if self.camera_stream else {}
        }
        
        return status

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI-Powered Surveillance System with API")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device ID")
    parser.add_argument("--headless", "-hl", action="store_true", help="Run without GUI")
    parser.add_argument("--api-only", action="store_true", help="Run API server only")
    parser.add_argument("--port", "-p", type=int, default=8080, help="API server port")
    
    args = parser.parse_args()
    
    # Set API port
    os.environ["API_PORT"] = str(args.port)
    
    logger.info(f"Starting AI-Powered Surveillance System (API on port {args.port})")
    
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