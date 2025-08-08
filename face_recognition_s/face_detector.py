# """
# Face detection module optimized for real-time surveillance.
# Uses dlib and face_recognition library for offline processing.
# """
# import cv2
# import numpy as np

# import dlib
# from typing import List, Tuple, Dict, Optional
# import time
# import torch

# from utils.config import config
# from utils.logger import logger

# class FaceDetection:
#     """Represents a detected face."""
    
#     def __init__(self, bbox: Tuple[int, int, int, int], landmarks: np.ndarray = None,
#                  encoding: np.ndarray = None, confidence: float = 1.0):
#         self.bbox = bbox  # (top, right, bottom, left)
#         self.landmarks = landmarks
#         self.encoding = encoding
#         self.confidence = confidence
#         self.timestamp = time.time()
#         self.face_id = None  # Will be set by face matcher
        
#         # Convert bbox format for easier use
#         self.top, self.right, self.bottom, self.left = bbox
#         self.width = self.right - self.left
#         self.height = self.bottom - self.top
#         self.center = ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

# class FaceDetector:
#     """High-performance face detector with multiple backends."""
    
#     def __init__(self, model: str = None):
#         self.model = model or config.face.model
#         self.detection_scale = config.face.detection_scale
        
#         # Performance tracking
#         self.detection_times = []
#         self.encoding_times = []
#         self.total_detections = 0
        
#         # GPU optimization
#         self.use_cuda = config.gpu.use_cuda and torch.cuda.is_available()
        
#         # Initialize face detector
#         self.face_detector = None
#         self.landmark_predictor = None
#         self._initialize_detector()
        
#         logger.info(f"Face detector initialized with model: {self.model}")
    
#     def _initialize_detector(self):
#         """Initialize the face detection models."""
#         try:
#             if self.model == "cnn" and self.use_cuda:
#                 # Use CNN model with GPU acceleration if available
#                 logger.info("Using CNN face detector with GPU acceleration")
#                 # Note: CNN model requires more setup for GPU, fallback to HOG if issues
#                 try:
#                     # Test CNN model
#                     test_image = np.zeros((100, 100, 3), dtype=np.uint8)
#                     face_recognition.face_locations(test_image, model="cnn")
#                     logger.info("CNN model verified")
#                 except Exception as e:
#                     logger.warning(f"CNN model failed, falling back to HOG: {e}")
#                     self.model = "hog"
            
#             # Initialize dlib components for additional features
#             self.face_detector = dlib.get_frontal_face_detector()
            
#             # Try to load the shape predictor
#             try:
#                 # This requires downloading shape_predictor_68_face_landmarks.dat
#                 # from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#                 predictor_path = "models/shape_predictor_68_face_landmarks.dat"
#                 self.landmark_predictor = dlib.shape_predictor(predictor_path)
#                 logger.info("Landmark predictor loaded successfully")
#             except Exception as e:
#                 logger.warning(f"Could not load landmark predictor: {e}")
#                 self.landmark_predictor = None
                
#         except Exception as e:
#             logger.error(f"Failed to initialize face detector: {e}")
#             self.model = "hog"  # Fallback to most reliable method
    
#     def detect_faces(self, frame: np.ndarray, return_encodings: bool = True) -> List[FaceDetection]:
#         """
#         Detect faces in the given frame.
        
#         Args:
#             frame: Input frame as numpy array
#             return_encodings: Whether to compute face encodings
            
#         Returns:
#             List of FaceDetection objects
#         """
#         start_time = time.time()
        
#         try:
#             # Scale down image for faster detection
#             if self.detection_scale < 1.0:
#                 small_frame = cv2.resize(frame, (0, 0), fx=self.detection_scale, fy=self.detection_scale)
#                 scale_factor = 1.0 / self.detection_scale
#             else:
#                 small_frame = frame
#                 scale_factor = 1.0
            
#             # Convert BGR to RGB for face_recognition library
#             rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
#             # Detect face locations
#             face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
            
#             detection_time = time.time() - start_time
#             self.detection_times.append(detection_time)
            
#             detections = []
            
#             for face_location in face_locations:
#                 # Scale back to original coordinates
#                 top, right, bottom, left = face_location
#                 if self.detection_scale < 1.0:
#                     top = int(top * scale_factor)
#                     right = int(right * scale_factor)
#                     bottom = int(bottom * scale_factor)
#                     left = int(left * scale_factor)
                
#                 bbox = (top, right, bottom, left)
                
#                 # Extract face encoding if requested
#                 encoding = None
#                 if return_encodings:
#                     encoding_start = time.time()
                    
#                     # Get face encoding
#                     face_encodings = face_recognition.face_encodings(
#                         rgb_frame, [face_location]
#                     )
                    
#                     if face_encodings:
#                         encoding = face_encodings[0]
                    
#                     encoding_time = time.time() - encoding_start
#                     self.encoding_times.append(encoding_time)
                
#                 # Get landmarks if predictor is available
#                 landmarks = None
#                 if self.landmark_predictor:
#                     landmarks = self._get_landmarks(frame, bbox)
                
#                 # Calculate confidence (simplified)
#                 face_width = right - left
#                 face_height = bottom - top
#                 confidence = min(1.0, (face_width * face_height) / (50 * 50))  # Normalize by minimum face size
                
#                 detection = FaceDetection(
#                     bbox=bbox,
#                     landmarks=landmarks,
#                     encoding=encoding,
#                     confidence=confidence
#                 )
                
#                 detections.append(detection)
            
#             self.total_detections += len(detections)
            
#             # Clean up old timing data
#             if len(self.detection_times) > 100:
#                 self.detection_times = self.detection_times[-100:]
#             if len(self.encoding_times) > 100:
#                 self.encoding_times = self.encoding_times[-100:]
            
#             return detections
            
#         except Exception as e:
#             logger.error(f"Face detection error: {e}")
#             return []
    
#     def _get_landmarks(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
#         """Get facial landmarks using dlib."""
#         if self.landmark_predictor is None:
#             return None
        
#         try:
#             top, right, bottom, left = bbox
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Convert to dlib rectangle
#             dlib_rect = dlib.rectangle(left, top, right, bottom)
            
#             # Get landmarks
#             landmarks = self.landmark_predictor(gray, dlib_rect)
            
#             # Convert to numpy array
#             landmark_points = np.array([[p.x, p.y] for p in landmarks.parts()])
            
#             return landmark_points
            
#         except Exception as e:
#             logger.warning(f"Landmark detection failed: {e}")
#             return None
    
#     def detect_faces_dlib(self, frame: np.ndarray) -> List[FaceDetection]:
#         """Alternative face detection using pure dlib (faster but less accurate)."""
#         try:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Scale down for faster detection
#             if self.detection_scale < 1.0:
#                 small_gray = cv2.resize(gray, (0, 0), fx=self.detection_scale, fy=self.detection_scale)
#                 scale_factor = 1.0 / self.detection_scale
#             else:
#                 small_gray = gray
#                 scale_factor = 1.0
            
#             # Detect faces
#             dlib_faces = self.face_detector(small_gray)
            
#             detections = []
#             for face in dlib_faces:
#                 # Scale coordinates back
#                 left = int(face.left() * scale_factor)
#                 top = int(face.top() * scale_factor)
#                 right = int(face.right() * scale_factor)
#                 bottom = int(face.bottom() * scale_factor)
                
#                 bbox = (top, right, bottom, left)
                
#                 # Get landmarks
#                 landmarks = self._get_landmarks(frame, bbox)
                
#                 detection = FaceDetection(
#                     bbox=bbox,
#                     landmarks=landmarks,
#                     confidence=0.8  # Default confidence for dlib detections
#                 )
                
#                 detections.append(detection)
            
#             return detections
            
#         except Exception as e:
#             logger.error(f"Dlib face detection error: {e}")
#             return []
    
#     def extract_face_roi(self, frame: np.ndarray, detection: FaceDetection, 
#                         padding: float = 0.2) -> np.ndarray:
#         """Extract face region of interest with padding."""
#         top, right, bottom, left = detection.bbox
        
#         # Add padding
#         height, width = frame.shape[:2]
#         pad_h = int((bottom - top) * padding)
#         pad_w = int((right - left) * padding)
        
#         # Ensure coordinates are within frame bounds
#         top = max(0, top - pad_h)
#         bottom = min(height, bottom + pad_h)
#         left = max(0, left - pad_w)
#         right = min(width, right + pad_w)
        
#         return frame[top:bottom, left:right]
    
#     def draw_face_overlay(self, frame: np.ndarray, detections: List[FaceDetection], 
#                          show_landmarks: bool = False, show_info: bool = True) -> np.ndarray:
#         """Draw face detection overlay on frame."""
#         overlay = frame.copy()
        
#         for i, detection in enumerate(detections):
#             top, right, bottom, left = detection.bbox
            
#             # Color based on confidence or recognition status
#             if hasattr(detection, 'face_id') and detection.face_id:
#                 color = (0, 255, 0)  # Green for recognized faces
#                 label = f"ID: {detection.face_id}"
#             else:
#                 color = (0, 0, 255)  # Red for unknown faces
#                 label = f"Unknown #{i}"
            
#             # Draw bounding box
#             cv2.rectangle(overlay, (left, top), (right, bottom), color, 2)
            
#             # Draw info text
#             if show_info:
#                 info_text = f"{label} ({detection.confidence:.2f})"
#                 text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
#                 # Background rectangle for text
#                 cv2.rectangle(overlay, (left, top - text_size[1] - 10), 
#                             (left + text_size[0], top), color, -1)
                
#                 # Text
#                 cv2.putText(overlay, info_text, (left, top - 5), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
#             # Draw landmarks if available
#             if show_landmarks and detection.landmarks is not None:
#                 for (x, y) in detection.landmarks:
#                     cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 0), -1)
        
#         return overlay
    
#     def get_detection_statistics(self) -> Dict:
#         """Get face detection performance statistics."""
#         avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
#         avg_encoding_time = np.mean(self.encoding_times) if self.encoding_times else 0
        
#         return {
#             'total_detections': self.total_detections,
#             'average_detection_time_ms': avg_detection_time * 1000,
#             'average_encoding_time_ms': avg_encoding_time * 1000,
#             'detection_fps': 1.0 / avg_detection_time if avg_detection_time > 0 else 0,
#             'model': self.model,
#             'detection_scale': self.detection_scale,
#             'cuda_available': self.use_cuda
#         }
    
#     def optimize_for_realtime(self):
#         """Optimize settings for real-time processing."""
#         # Reduce detection scale for faster processing
#         self.detection_scale = min(0.5, self.detection_scale)
        
#         # Switch to HOG model if using CNN for better speed
#         if self.model == "cnn":
#             self.model = "hog"
#             logger.info("Switched to HOG model for real-time optimization")
        
#         logger.info(f"Optimized for real-time: scale={self.detection_scale}, model={self.model}")
    
#     def cleanup_gpu_memory(self):
#         """Clean up GPU memory if using CUDA."""
#         if self.use_cuda and torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             logger.debug("GPU memory cache cleared")

"""
Face detection module optimized for real-time surveillance.
Uses dlib and face_recognition library for offline processing.
"""
import cv2
import numpy as np
import face_recognition  # Add missing import
import dlib
from typing import List, Tuple, Dict, Optional
import time
import torch

from utils.config import config
from utils.logger import logger

class FaceDetection:
    """Represents a detected face."""
    
    def __init__(self, bbox: Tuple[int, int, int, int], landmarks: np.ndarray = None,
                 encoding: np.ndarray = None, confidence: float = 1.0):
        self.bbox = bbox  # (top, right, bottom, left)
        self.landmarks = landmarks
        self.encoding = encoding
        self.confidence = confidence
        self.timestamp = time.time()
        self.face_id = None  # Will be set by face matcher
        
        # Convert bbox format for easier use
        self.top, self.right, self.bottom, self.left = bbox
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.center = ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

class FaceDetector:
    """High-performance face detector with multiple backends."""
    
    def __init__(self, model: str = None):
        self.model = model or config.face.model
        self.detection_scale = config.face.detection_scale
        
        # Performance tracking
        self.detection_times = []
        self.encoding_times = []
        self.total_detections = 0
        
        # GPU optimization
        self.use_cuda = config.gpu.use_cuda and torch.cuda.is_available()
        
        # Initialize face detector
        self.face_detector = None
        self.landmark_predictor = None
        self._initialize_detector()
        
        logger.info(f"Face detector initialized with model: {self.model}")
    
    def _initialize_detector(self):
        """Initialize the face detection models."""
        try:
            if self.model == "cnn" and self.use_cuda:
                # Use CNN model with GPU acceleration if available
                logger.info("Using CNN face detector with GPU acceleration")
                # Note: CNN model requires more setup for GPU, fallback to HOG if issues
                try:
                    # Test CNN model
                    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                    face_recognition.face_locations(test_image, model="cnn")
                    logger.info("CNN model verified")
                except Exception as e:
                    logger.warning(f"CNN model failed, falling back to HOG: {e}")
                    self.model = "hog"
            
            # Initialize dlib components for additional features
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Try to load the shape predictor
            try:
                # This requires downloading shape_predictor_68_face_landmarks.dat
                # from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
                predictor_path = "models/shape_predictor_68_face_landmarks.dat"
                self.landmark_predictor = dlib.shape_predictor(predictor_path)
                logger.info("Landmark predictor loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load landmark predictor: {e}")
                self.landmark_predictor = None
                
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            self.model = "hog"  # Fallback to most reliable method
    
    def detect_faces(self, frame: np.ndarray, return_encodings: bool = True) -> List[FaceDetection]:
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input frame as numpy array
            return_encodings: Whether to compute face encodings
            
        Returns:
            List of FaceDetection objects
        """
        start_time = time.time()
        
        try:
            # Scale down image for faster detection
            if self.detection_scale < 1.0:
                small_frame = cv2.resize(frame, (0, 0), fx=self.detection_scale, fy=self.detection_scale)
                scale_factor = 1.0 / self.detection_scale
            else:
                small_frame = frame
                scale_factor = 1.0
            
            # Convert BGR to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
            
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            
            detections = []
            
            for face_location in face_locations:
                # Scale back to original coordinates
                top, right, bottom, left = face_location
                if self.detection_scale < 1.0:
                    top = int(top * scale_factor)
                    right = int(right * scale_factor)
                    bottom = int(bottom * scale_factor)
                    left = int(left * scale_factor)
                
                bbox = (top, right, bottom, left)
                
                # Extract face encoding if requested
                encoding = None
                if return_encodings:
                    encoding_start = time.time()
                    
                    # Get face encoding
                    face_encodings = face_recognition.face_encodings(
                        rgb_frame, [face_location]
                    )
                    
                    if face_encodings:
                        encoding = face_encodings[0]
                    
                    encoding_time = time.time() - encoding_start
                    self.encoding_times.append(encoding_time)
                
                # Get landmarks if predictor is available
                landmarks = None
                if self.landmark_predictor:
                    landmarks = self._get_landmarks(frame, bbox)
                
                # Calculate confidence (simplified)
                face_width = right - left
                face_height = bottom - top
                confidence = min(1.0, (face_width * face_height) / (50 * 50))  # Normalize by minimum face size
                
                detection = FaceDetection(
                    bbox=bbox,
                    landmarks=landmarks,
                    encoding=encoding,
                    confidence=confidence
                )
                
                detections.append(detection)
            
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
        """Get facial landmarks using dlib."""
        if self.landmark_predictor is None:
            return None
        
        try:
            top, right, bottom, left = bbox
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Convert to dlib rectangle
            dlib_rect = dlib.rectangle(left, top, right, bottom)
            
            # Get landmarks
            landmarks = self.landmark_predictor(gray, dlib_rect)
            
            # Convert to numpy array
            landmark_points = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            return landmark_points
            
        except Exception as e:
            logger.warning(f"Landmark detection failed: {e}")
            return None
    
    def detect_faces_dlib(self, frame: np.ndarray) -> List[FaceDetection]:
        """Alternative face detection using pure dlib (faster but less accurate)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Scale down for faster detection
            if self.detection_scale < 1.0:
                small_gray = cv2.resize(gray, (0, 0), fx=self.detection_scale, fy=self.detection_scale)
                scale_factor = 1.0 / self.detection_scale
            else:
                small_gray = gray
                scale_factor = 1.0
            
            # Detect faces
            dlib_faces = self.face_detector(small_gray)
            
            detections = []
            for face in dlib_faces:
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
            
            return detections
            
        except Exception as e:
            logger.error(f"Dlib face detection error: {e}")
            return []
    
    def extract_face_roi(self, frame: np.ndarray, detection: FaceDetection, 
                        padding: float = 0.2) -> np.ndarray:
        """Extract face region of interest with padding."""
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
        
        return frame[top:bottom, left:right]
    
    def draw_face_overlay(self, frame: np.ndarray, detections: List[FaceDetection], 
                         show_landmarks: bool = False, show_info: bool = True) -> np.ndarray:
        """Draw face detection overlay on frame."""
        overlay = frame.copy()
        
        for i, detection in enumerate(detections):
            top, right, bottom, left = detection.bbox
            
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
                text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background rectangle for text
                cv2.rectangle(overlay, (left, top - text_size[1] - 10), 
                            (left + text_size[0], top), color, -1)
                
                # Text
                cv2.putText(overlay, info_text, (left, top - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw landmarks if available
            if show_landmarks and detection.landmarks is not None:
                for (x, y) in detection.landmarks:
                    cv2.circle(overlay, (int(x), int(y)), 2, (255, 255, 0), -1)
        
        return overlay
    
    def get_detection_statistics(self) -> Dict:
        """Get face detection performance statistics."""
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        avg_encoding_time = np.mean(self.encoding_times) if self.encoding_times else 0
        
        return {
            'total_detections': self.total_detections,
            'average_detection_time_ms': avg_detection_time * 1000,
            'average_encoding_time_ms': avg_encoding_time * 1000,
            'detection_fps': 1.0 / avg_detection_time if avg_detection_time > 0 else 0,
            'model': self.model,
            'detection_scale': self.detection_scale,
            'cuda_available': self.use_cuda
        }
    
    def optimize_for_realtime(self):
        """Optimize settings for real-time processing."""
        # Reduce detection scale for faster processing
        self.detection_scale = min(0.5, self.detection_scale)
        
        # Switch to HOG model if using CNN for better speed
        if self.model == "cnn":
            self.model = "hog"
            logger.info("Switched to HOG model for real-time optimization")

        logger.info(f"Optimized for real-time: scale={self.detection_scale}, model={self.model}")
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory if using CUDA."""
        if self.use_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")