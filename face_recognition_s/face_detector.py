"""
Face detection module optimized for real-time surveillance.
Uses dlib and face_recognition library with comprehensive error handling.
"""
import cv2
import os
import numpy as np
import face_recognition
import dlib
from typing import List, Tuple, Dict, Optional
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logger = logging.getLogger(__name__)

class FaceDetection:
    """Represents a detected face with comprehensive information."""
    
    def __init__(self, bbox: Tuple[int, int, int, int], landmarks: np.ndarray = None,
                 encoding: np.ndarray = None, confidence: float = 1.0):
        # Validate bbox
        if len(bbox) != 4:
            raise ValueError("bbox must have 4 elements: (top, right, bottom, left)")
        
        self.bbox = bbox  # (top, right, bottom, left)
        self.landmarks = landmarks
        self.encoding = encoding
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
        self.timestamp = time.time()
        self.face_id = None  # Will be set by face matcher
        
        # Convert bbox format for easier use
        self.top, self.right, self.bottom, self.left = bbox
        self.width = max(0, self.right - self.left)
        self.height = max(0, self.bottom - self.top)
        self.center = ((self.left + self.right) // 2, (self.top + self.bottom) // 2)
        
        # Validate dimensions
        if self.width <= 0 or self.height <= 0:
            logger.warning(f"Invalid face dimensions: {self.width}x{self.height}")

class FaceDetector:
    """High-performance face detector with multiple backends and error handling."""
    
    def __init__(self, model: str = None):
        # Import config here to avoid circular imports
        try:
            from utils.config import config
            self.model = model or config.face.model
            self.detection_scale = config.face.detection_scale
            self.use_cuda = config.gpu.use_cuda
        except ImportError:
            logger.warning("Config not available, using defaults")
            self.model = model or "hog"
            self.detection_scale = 0.5
            self.use_cuda = False
        
        # Performance tracking
        self.detection_times = []
        self.encoding_times = []
        self.total_detections = 0
        self._stats_lock = threading.Lock()
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Initialize face detector
        self.face_detector = None
        self.landmark_predictor = None
        self._detector_lock = threading.Lock()
        
        self._initialize_detector()
        
        logger.info(f"Face detector initialized with model: {self.model}")
    
    def _initialize_detector(self):
        """Initialize the face detection models with error handling."""
        try:
            # Test face_recognition library with different models
            if self.model == "cnn":
                try:
                    # Test CNN model availability
                    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                    face_recognition.face_locations(test_image, model="cnn")
                    logger.info("CNN face detection model verified")
                except Exception as e:
                    logger.warning(f"CNN model failed, falling back to HOG: {e}")
                    self.model = "hog"
            
            # Initialize dlib components for additional features
            try:
                self.face_detector = dlib.get_frontal_face_detector()
                logger.info("Dlib face detector initialized")
            except Exception as e:
                logger.warning(f"Dlib face detector initialization failed: {e}")
            
            # Try to load the shape predictor
            try:
                predictor_path = "models/shape_predictor_68_face_landmarks.dat"
                if os.path.exists(predictor_path):
                    self.landmark_predictor = dlib.shape_predictor(predictor_path)
                    logger.info("Landmark predictor loaded successfully")
                else:
                    logger.info("Landmark predictor file not found, landmarks will be unavailable")
            except Exception as e:
                logger.warning(f"Could not load landmark predictor: {e}")
                self.landmark_predictor = None
                
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            self.model = "hog"  # Fallback to most reliable method
    
    def detect_faces(self, frame: np.ndarray, return_encodings: bool = True) -> List[FaceDetection]:
        """
        Detect faces in the given frame with comprehensive error handling.
        
        Args:
            frame: Input frame as numpy array
            return_encodings: Whether to compute face encodings
            
        Returns:
            List of FaceDetection objects
        """
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided to detect_faces")
            return []
        
        start_time = time.time()
        
        try:
            # Validate frame dimensions
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                logger.warning(f"Invalid frame shape: {frame.shape}")
                return []
            
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                logger.warning(f"Invalid frame dimensions: {width}x{height}")
                return []
            
            # Scale down image for faster detection
            scale_factor = 1.0
            if self.detection_scale < 1.0:
                new_width = int(width * self.detection_scale)
                new_height = int(height * self.detection_scale)
                if new_width > 0 and new_height > 0:
                    small_frame = cv2.resize(frame, (new_width, new_height))
                    scale_factor = 1.0 / self.detection_scale
                else:
                    small_frame = frame
                    scale_factor = 1.0
            else:
                small_frame = frame
            
            # Convert BGR to RGB for face_recognition library
            try:
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                logger.error(f"Color conversion error: {e}")
                return []
            
            # Detect face locations
            try:
                face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
            except Exception as e:
                logger.error(f"Face location detection error: {e}")
                return []
            
            detection_time = time.time() - start_time
            with self._stats_lock:
                self.detection_times.append(detection_time)
            
            detections = []
            
            for face_location in face_locations:
                try:
                    # Scale back to original coordinates
                    top, right, bottom, left = face_location
                    if scale_factor != 1.0:
                        top = int(top * scale_factor)
                        right = int(right * scale_factor)
                        bottom = int(bottom * scale_factor)
                        left = int(left * scale_factor)
                        
                        # Ensure coordinates are within frame bounds
                        top = max(0, min(height, top))
                        bottom = max(0, min(height, bottom))
                        left = max(0, min(width, left))
                        right = max(0, min(width, right))
                    
                    bbox = (top, right, bottom, left)
                    
                    # Extract face encoding if requested
                    encoding = None
                    if return_encodings:
                        encoding_start = time.time()
                        
                        try:
                            # Get face encoding
                            face_encodings = face_recognition.face_encodings(
                                rgb_frame, [face_location]
                            )
                            
                            if face_encodings:
                                encoding = face_encodings[0]
                            
                            encoding_time = time.time() - encoding_start
                            with self._stats_lock:
                                self.encoding_times.append(encoding_time)
                                
                        except Exception as e:
                            logger.warning(f"Face encoding error: {e}")
                    
                    # Get landmarks if predictor is available
                    landmarks = None
                    if self.landmark_predictor:
                        try:
                            landmarks = self._get_landmarks(frame, bbox)
                        except Exception as e:
                            logger.debug(f"Landmark detection error: {e}")
                    
                    # Calculate confidence based on face size and quality
                    face_width = right - left
                    face_height = bottom - top
                    face_area = face_width * face_height
                    min_face_area = 50 * 50  # Minimum reasonable face size
                    confidence = min(1.0, face_area / min_face_area)
                    
                    # Additional quality checks
                    if face_width < 20 or face_height < 20:
                        confidence *= 0.5  # Reduce confidence for very small faces
                    
                    detection = FaceDetection(
                        bbox=bbox,
                        landmarks=landmarks,
                        encoding=encoding,
                        confidence=confidence
                    )
                    
                    detections.append(detection)
                    
                except Exception as e:
                    logger.warning(f"Error processing face detection: {e}")
                    continue
            
            with self._stats_lock:
                self.total_detections += len(detections)
                
                # Clean up old timing data
                if len(self.detection_times) > 100:
                    self.detection_times = self.detection_times[-100:]
                if len(self.encoding_times) > 100:
                    self.encoding_times = self.encoding_times[-100:]
            
            return detections
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def _get_landmarks(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Get facial landmarks using dlib with error handling."""
        if self.landmark_predictor is None:
            return None
        
        try:
            top, right, bottom, left = bbox
            
            # Validate coordinates
            height, width = frame.shape[:2]
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                logger.warning(f"Invalid bbox coordinates: {bbox}")
                return None
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Convert to dlib rectangle
            dlib_rect = dlib.rectangle(left, top, right, bottom)
            
            # Get landmarks
            landmarks = self.landmark_predictor(gray, dlib_rect)
            
            # Convert to numpy array
            landmark_points = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            return landmark_points
            
        except Exception as e:
            logger.debug(f"Landmark detection failed: {e}")
            return None
    
    def detect_faces_dlib(self, frame: np.ndarray) -> List[FaceDetection]:
        """Alternative face detection using pure dlib (faster but less accurate)."""
        if frame is None or frame.size == 0:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Scale down for faster detection
            scale_factor = 1.0
            if self.detection_scale < 1.0:
                height, width = gray.shape
                new_width = int(width * self.detection_scale)
                new_height = int(height * self.detection_scale)
                if new_width > 0 and new_height > 0:
                    small_gray = cv2.resize(gray, (new_width, new_height))
                    scale_factor = 1.0 / self.detection_scale
                else:
                    small_gray = gray
            else:
                small_gray = gray
            
            # Detect faces
            if self.face_detector is None:
                logger.warning("Dlib face detector not available")
                return []
            
            dlib_faces = self.face_detector(small_gray)
            
            detections = []
            for face in dlib_faces:
                try:
                    # Scale coordinates back
                    left = int(face.left() * scale_factor)
                    top = int(face.top() * scale_factor)
                    right = int(face.right() * scale_factor)
                    bottom = int(face.bottom() * scale_factor)
                    
                    bbox = (top, right, bottom, left)
                    
                    # Get landmarks
                    landmarks = self._get_landmarks(frame, bbox)
                    
                    detection = FaceDetection(
                        bbox=bbox,
                        landmarks=landmarks,
                        confidence=0.8  # Default confidence for dlib detections
                    )
                    
                    detections.append(detection)
                    
                except Exception as e:
                    logger.warning(f"Error processing dlib detection: {e}")
                    continue
            
            return detections
            
        except Exception as e:
            logger.error(f"Dlib face detection error: {e}")
            return []
    
    def extract_face_roi(self, frame: np.ndarray, detection: FaceDetection, 
                        padding: float = 0.2) -> Optional[np.ndarray]:
        """Extract face region of interest with padding and validation."""
        try:
            if frame is None or frame.size == 0:
                return None
            
            top, right, bottom, left = detection.bbox
            
            # Add padding
            height, width = frame.shape[:2]
            pad_h = int((bottom - top) * padding)
            pad_w = int((right - left) * padding)
            
            # Ensure coordinates are within frame bounds
            top = max(0, top - pad_h)
            bottom = min(height, bottom + pad_h)
            left = max(0, left - pad_w)
            right = min(width, right + pad_w)
            
            # Validate final coordinates
            if top >= bottom or left >= right:
                logger.warning("Invalid ROI coordinates after padding")
                return None
            
            roi = frame[top:bottom, left:right]
            
            if roi.size == 0:
                logger.warning("Empty ROI extracted")
                return None
            
            return roi
            
        except Exception as e:
            logger.error(f"Error extracting face ROI: {e}")
            return None
    
    def draw_face_overlay(self, frame: np.ndarray, detections: List[FaceDetection], 
                         show_landmarks: bool = False, show_info: bool = True) -> np.ndarray:
        """Draw face detection overlay on frame with error handling."""
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame for overlay")
            return frame
        
        try:
            overlay = frame.copy()
            
            for i, detection in enumerate(detections):
                try:
                    top, right, bottom, left = detection.bbox
                    
                    # Validate coordinates
                    height, width = frame.shape[:2]
                    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                        logger.warning(f"Invalid bbox for overlay: {detection.bbox}")
                        continue
                    
                    # Color based on confidence or recognition status
                    if hasattr(detection, 'face_id') and detection.face_id:
                        color = (0, 255, 0)  # Green for recognized faces
                        label = f"ID: {detection.face_id}"
                    else:
                        color = (0, 0, 255)  # Red for unknown faces
                        label = f"Unknown #{i}"
                    
                    # Draw bounding box
                    cv2.rectangle(overlay, (left, top), (right, bottom), color, 2)
                    
                    # Draw info text
                    if show_info:
                        info_text = f"{label} ({detection.confidence:.2f})"
                        
                        # Calculate text size and position
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        
                        try:
                            text_size = cv2.getTextSize(info_text, font, font_scale, thickness)[0]
                            
                            # Background rectangle for text
                            text_bg_top = max(0, top - text_size[1] - 10)
                            text_bg_bottom = top
                            text_bg_right = min(width, left + text_size[0])
                            
                            cv2.rectangle(overlay, (left, text_bg_top), 
                                        (text_bg_right, text_bg_bottom), color, -1)
                            
                            # Text
                            text_y = max(text_size[1], top - 5)
                            cv2.putText(overlay, info_text, (left, text_y), 
                                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                        except Exception as e:
                            logger.debug(f"Error drawing text overlay: {e}")
                    
                    # Draw landmarks if available
                    if show_landmarks and detection.landmarks is not None:
                        try:
                            for (x, y) in detection.landmarks:
                                if 0 <= x < width and 0 <= y < height:
                                    cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 0), -1)
                        except Exception as e:
                            logger.debug(f"Error drawing landmarks: {e}")
                    
                except Exception as e:
                    logger.warning(f"Error drawing detection overlay: {e}")
                    continue
            
            return overlay
            
        except Exception as e:
            logger.error(f"Error creating face overlay: {e}")
            return frame
    
    def get_detection_statistics(self) -> Dict:
        """Get face detection performance statistics."""
        try:
            with self._stats_lock:
                avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
                avg_encoding_time = np.mean(self.encoding_times) if self.encoding_times else 0
                
                return {
                    'total_detections': self.total_detections,
                    'average_detection_time_ms': avg_detection_time * 1000,
                    'average_encoding_time_ms': avg_encoding_time * 1000,
                    'detection_fps': 1.0 / avg_detection_time if avg_detection_time > 0 else 0,
                    'model': self.model,
                    'detection_scale': self.detection_scale,
                    'cuda_available': self.use_cuda,
                    'dlib_available': self.face_detector is not None,
                    'landmarks_available': self.landmark_predictor is not None
                }
        except Exception as e:
            logger.error(f"Error getting detection statistics: {e}")
            return {
                'total_detections': self.total_detections,
                'model': self.model,
                'error': str(e)
            }
    
    def optimize_for_realtime(self):
        """Optimize settings for real-time processing."""
        try:
            # Reduce detection scale for faster processing
            self.detection_scale = min(0.5, self.detection_scale)
            
            # Switch to HOG model if using CNN for better speed
            if self.model == "cnn":
                self.model = "hog"
                logger.info("Switched to HOG model for real-time optimization")
            
            logger.info(f"Optimized for real-time: scale={self.detection_scale}, model={self.model}")
            
        except Exception as e:
            logger.error(f"Error optimizing for real-time: {e}")
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory if using CUDA."""
        try:
            if self.use_cuda:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("GPU memory cache cleared")
                except ImportError:
                    logger.debug("PyTorch not available for GPU cleanup")
        except Exception as e:
            logger.debug(f"Error during GPU cleanup: {e}")
    
    def validate_detection(self, detection: FaceDetection, frame_shape: Tuple[int, int]) -> bool:
        """Validate a face detection against frame boundaries."""
        try:
            height, width = frame_shape
            top, right, bottom, left = detection.bbox
            
            # Check bounds
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                return False
            
            # Check minimum size
            if (right - left) < 20 or (bottom - top) < 20:
                return False
            
            # Check confidence
            if detection.confidence < 0.1:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating detection: {e}")
            return False
    
    def batch_detect_faces(self, frames: List[np.ndarray], 
                          return_encodings: bool = True) -> List[List[FaceDetection]]:
        """Detect faces in multiple frames efficiently."""
        if not frames:
            return []
        
        try:
            results = []
            
            # Process frames in parallel if we have multiple
            if len(frames) > 1 and self.thread_pool:
                futures = []
                for frame in frames:
                    future = self.thread_pool.submit(self.detect_faces, frame, return_encodings)
                    futures.append(future)
                
                for future in futures:
                    try:
                        result = future.result(timeout=5.0)  # 5 second timeout
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Batch detection error: {e}")
                        results.append([])
            else:
                # Process sequentially
                for frame in frames:
                    result = self.detect_faces(frame, return_encodings)
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch face detection error: {e}")
            return [[] for _ in frames]
    
    def set_detection_parameters(self, model: str = None, scale: float = None):
        """Update detection parameters at runtime."""
        try:
            if model and model in ["hog", "cnn"]:
                self.model = model
                logger.info(f"Updated detection model to: {model}")
            
            if scale and 0.1 <= scale <= 1.0:
                self.detection_scale = scale
                logger.info(f"Updated detection scale to: {scale}")
                
        except Exception as e:
            logger.error(f"Error updating detection parameters: {e}")
    
    def get_face_quality_score(self, frame: np.ndarray, detection: FaceDetection) -> float:
        """Calculate a quality score for a detected face."""
        try:
            # Extract face ROI
            face_roi = self.extract_face_roi(frame, detection, padding=0.1)
            if face_roi is None:
                return 0.0
            
            # Convert to grayscale for analysis
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)  # Normalize
            
            # Calculate brightness score
            mean_brightness = np.mean(gray_roi)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0  # Optimal around 128
            
            # Size score based on face dimensions
            face_area = detection.width * detection.height
            size_score = min(1.0, face_area / (100 * 100))  # Normalize by 100x100 pixels
            
            # Combine scores
            quality_score = (sharpness_score * 0.4 + brightness_score * 0.3 + 
                           size_score * 0.2 + detection.confidence * 0.1)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.debug(f"Error calculating face quality: {e}")
            return detection.confidence
    
    def filter_detections_by_quality(self, detections: List[FaceDetection], 
                                   frame: np.ndarray, min_quality: float = 0.3) -> List[FaceDetection]:
        """Filter detections based on quality score."""
        try:
            filtered_detections = []
            
            for detection in detections:
                quality_score = self.get_face_quality_score(frame, detection)
                if quality_score >= min_quality:
                    # Store quality score in detection for later use
                    detection.quality_score = quality_score
                    filtered_detections.append(detection)
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Error filtering detections by quality: {e}")
            return detections
    
    def __del__(self):
        """Cleanup resources when detector is destroyed."""
        try:
            if hasattr(self, 'thread_pool') and self.thread_pool:
                self.thread_pool.shutdown(wait=False)
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")