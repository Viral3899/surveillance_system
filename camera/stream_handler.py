"""
Camera stream handler for the surveillance system.
Optimized for real-time processing with GPU acceleration.
"""
import cv2
import time
import numpy as np
from typing import Optional, Generator, Tuple
import threading
from queue import Queue, Empty
import torch

from utils.config import config
from utils.logger import logger

class CameraStream:
    """Optimized camera stream handler with threading and GPU support."""
    
    def __init__(self, device_id: int = None, resolution: Tuple[int, int] = None):
        self.device_id = device_id or config.camera.device_id
        self.resolution = resolution or config.camera.resolution
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue = Queue(maxsize=config.camera.buffer_size)
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        
        self._fps_counter = 0
        self._fps_start_time = time.time()
        self._current_fps = 0
        
        # GPU optimization
        self.use_cuda = config.gpu.use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        
        self._initialize_camera()
    
    def _initialize_camera(self) -> bool:
        """Initialize camera with optimal settings."""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.device_id}")
            
            # Set camera properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, config.camera.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.camera.buffer_size)
            
            # Additional optimizations
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def _capture_frames(self):
        """Background thread for frame capture."""
        frame_time = 1.0 / config.camera.fps
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Clear old frames if queue is full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                
                # Add new frame
                self.frame_queue.put(frame)
                
                # FPS control
                time.sleep(frame_time)
                
            except Exception as e:
                logger.error(f"Error in capture thread: {e}")
                break
    
    def start_stream(self) -> bool:
        """Start the camera stream."""
        if self.running:
            logger.warning("Stream already running")
            return True
        
        if self.cap is None or not self.cap.isOpened():
            if not self._initialize_camera():
                return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        logger.info("Camera stream started")
        return True
    
    def stop_stream(self):
        """Stop the camera stream."""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        logger.info("Camera stream stopped")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the latest frame from the stream."""
        if not self.running:
            return None
        
        try:
            frame = self.frame_queue.get(timeout=timeout)
            self._update_fps()
            return frame
        except Empty:
            logger.warning("Frame timeout")
            return None
    
    def get_frame_generator(self) -> Generator[np.ndarray, None, None]:
        """Generator for continuous frame access."""
        while self.running:
            frame = self.get_frame()
            if frame is not None:
                yield frame
            else:
                break
    
    def _update_fps(self):
        """Update FPS counter."""
        self._fps_counter += 1
        current_time = time.time()
        
        if current_time - self._fps_start_time >= 1.0:
            self._current_fps = self._fps_counter
            self._fps_counter = 0
            self._fps_start_time = current_time
    
    def get_fps(self) -> float:
        """Get current processing FPS."""
        return self._current_fps
    
    def get_camera_info(self) -> dict:
        """Get camera information."""
        if not self.cap:
            return {}
        
        return {
            "device_id": self.device_id,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS)),
            "processing_fps": self._current_fps,
            "cuda_available": self.use_cuda
        }
    
    def is_running(self) -> bool:
        """Check if stream is running."""
        return self.running and (self.cap is not None) and self.cap.isOpened()
    
    def __enter__(self):
        """Context manager entry."""
        self.start_stream()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_stream()

class MultiCameraManager:
    """Manager for multiple camera streams."""
    
    def __init__(self):
        self.cameras = {}
        self.active_camera_id = None
    
    def add_camera(self, camera_id: str, device_id: int, resolution: Tuple[int, int] = None) -> bool:
        """Add a new camera to the manager."""
        try:
            camera = CameraStream(device_id, resolution)
            if camera.start_stream():
                self.cameras[camera_id] = camera
                if self.active_camera_id is None:
                    self.active_camera_id = camera_id
                logger.info(f"Added camera: {camera_id}")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Failed to add camera {camera_id}: {e}")
            return False
    
    def remove_camera(self, camera_id: str):
        """Remove a camera from the manager."""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop_stream()
            del self.cameras[camera_id]
            
            if self.active_camera_id == camera_id:
                self.active_camera_id = next(iter(self.cameras.keys()), None)
            
            logger.info(f"Removed camera: {camera_id}")
    
    def switch_camera(self, camera_id: str) -> bool:
        """Switch to a different camera."""
        if camera_id in self.cameras:
            self.active_camera_id = camera_id
            logger.info(f"Switched to camera: {camera_id}")
            return True
        return False
    
    def get_active_frame(self) -> Optional[np.ndarray]:
        """Get frame from the active camera."""
        if self.active_camera_id and self.active_camera_id in self.cameras:
            return self.cameras[self.active_camera_id].get_frame()
        return None
    
    def get_all_camera_info(self) -> dict:
        """Get information about all cameras."""
        return {cam_id: cam.get_camera_info() for cam_id, cam in self.cameras.items()}
    
    def stop_all(self):
        """Stop all camera streams."""
        for camera in self.cameras.values():
            camera.stop_stream()
        self.cameras.clear()
        self.active_camera_id = None