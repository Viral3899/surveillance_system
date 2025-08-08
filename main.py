#!/usr/bin/env python3
"""
Main surveillance system controller.
Orchestrates all components for real-time AI-powered surveillance.
"""
import cv2
import numpy as np
import time
import argparse
import signal
import sys
import threading
import os
import shutil
from typing import Optional, Dict, Any

# Import surveillance modules

from camera.stream_handler import CameraStream, MultiCameraManager
from detection.motion_detector import MotionDetector, MotionDetectionMethod
from face_recognition_s.face_detector import FaceDetector
from face_recognition_s.face_matcher import FaceMatcher
from anomaly.anomaly_detector import AnomalyDetector
from utils.config import config
from utils.logger import logger

# GPU memory managemepythp
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
    """Main surveillance system controller."""
    
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
        
        # Initialize system
        self._initialize_system()
        
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
        """Display surveillance results in GUI windows."""
        # Create display frames
        display_frame = frame.copy()
        motion_frame = frame.copy()
        
        # Draw motion detection overlay
        if motion_events:
            motion_frame = self.motion_detector.draw_motion_overlay(motion_frame, motion_events)
        
        # Draw face detection overlay
        if face_detections:
            display_frame = self.face_detector.draw_face_overlay(
                display_frame, face_detections, show_landmarks=False
            )
        
        # Draw anomaly overlay
        if anomalies:
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
            f"Learning: {'ON' if self.anomaly_detector.is_learning else 'OFF'}"
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
        
        # Controls help
        help_lines = [
            "Controls:",
            "Q - Quit",
            "R - Reset",
            "L - Toggle Learning",
            "S - Save Frame"
        ]
        
        help_y_start = height - len(help_lines) * 20 - 20
        cv2.rectangle(frame, (10, help_y_start - 10), (200, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, help_y_start - 10), (200, height - 10), (255, 255, 255), 1)
        
        for i, line in enumerate(help_lines):
            y_pos = help_y_start + i * 20
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
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

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Surveillance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Start with default camera and GUI
  python main.py --camera 1 --headless    # Use camera 1 in headless mode  
  python main.py --web                     # Launch with web interface
  python main.py --face-gallery ./faces   # Custom face gallery path
        """
    )
    parser.add_argument("--camera", "-c", type=int, default=0, 
                       help="Camera device ID (default: 0)")
    parser.add_argument("--headless", "-hl", action="store_true",
                       help="Run without GUI")
    parser.add_argument("--web", "-w", action="store_true",
                       help="Launch web interface (requires streamlit)")
    parser.add_argument("--config", "-cfg", type=str,
                       help="Path to custom config file (not implemented yet)")
    parser.add_argument("--face-gallery", "-fg", type=str,
                       help="Path to face gallery folder")
    parser.add_argument("--output-dir", "-o", type=str,
                       help="Output directory for recordings and logs")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(f"/dev/video{args.camera}") and os.name != 'nt':
        if args.camera != 0:  # Don't warn for default camera
            print(f"Warning: Camera device {args.camera} may not exist")
    
    # Update config if custom paths provided
    if args.face_gallery:
        if not os.path.exists(args.face_gallery):
            print(f"Error: Face gallery path does not exist: {args.face_gallery}")
            return 1
        config.face.face_gallery_path = args.face_gallery
        config._create_directories()
    
    if args.output_dir:
        config.logging.output_dir = args.output_dir
        config._create_directories()
    
    if args.verbose:
        config.logging.log_level = "DEBUG"
    
    if args.no_gpu:
        config.gpu.use_cuda = False
    
    # Print startup information
    print("=" * 60)
    print("🔍 AI-Powered Surveillance System")
    print("=" * 60)
    print(f"📹 Camera: {args.camera}")
    print(f"🖥️  Display: {'Headless' if args.headless else 'GUI'}")
    print(f"🌐 Web Interface: {'Enabled' if args.web else 'Disabled'}")
    print(f"👥 Face Gallery: {config.face.face_gallery_path}")
    print(f"📁 Output Directory: {config.logging.output_dir}")
    print(f"🖥️  GPU Acceleration: {'Enabled' if config.gpu.use_cuda else 'Disabled'}")
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"🎮 GPU Device: {torch.cuda.get_device_name()}")
    print("=" * 60)
    
    logger.info("Starting AI-Powered Surveillance System")
    logger.info(f"Configuration: Camera={args.camera}, Headless={args.headless}, Web={args.web}")
    
    try:
        # Create surveillance system
        surveillance = SurveillanceSystem(
            camera_id=args.camera,
            gui_mode=not args.headless
        )
        
        if args.web:
            # Launch web interface
            print("\n🌐 Starting web interface.")
            print("💡 Tip: Run 'streamlit run main.py --web' for better web experience")
            web_ui = SurveillanceWebUI(surveillance)
            web_ui.create_streamlit_app()
        else:
            # Print controls information
            if not args.headless:
                print("\n⌨️  Controls:")
                print("   Q - Quit system")
                print("   R - Reset detectors") 
                print("   L - Toggle learning mode")
                print("   S - Save current frame")
                print("\n🚀 Starting surveillance. Press Q to quit")
            else:
                print("\n🚀 Starting headless surveillance. Press Ctrl+C to quit")
            
            # Start surveillance
            surveillance.start()
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Surveillance system interrupted by user")
        logger.info("Surveillance system interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")
        return 1
    
    print("✅ Surveillance system shutdown complete")
    logger.info("Surveillance system shutdown complete")
    return 0

if __name__ == "__main__":
    # Ensure proper exit
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)