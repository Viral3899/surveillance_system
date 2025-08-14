"""
Advanced motion detection module for surveillance system.
Optimized for real-time processing with multiple detection methods and comprehensive error handling.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
import time
import threading
import logging

# Set up logging
logger = logging.getLogger(__name__)

class MotionDetectionMethod(Enum):
    """Available motion detection methods."""
    BACKGROUND_SUBTRACTION = "background_subtraction"
    FRAME_DIFFERENCING = "frame_differencing"
    OPTICAL_FLOW = "optical_flow"
    HYBRID = "hybrid"

class MotionEvent:
    """Represents a detected motion event with validation."""
    
    def __init__(self, bbox: Tuple[int, int, int, int], area: float, 
                 confidence: float, timestamp: float):
        # Validate inputs
        if len(bbox) != 4:
            raise ValueError("bbox must have 4 elements: (x, y, width, height)")
        
        if area < 0:
            raise ValueError("area must be non-negative")
        
        if not 0.0 <= confidence <= 1.0:
            confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range
        
        self.bbox = bbox  # (x, y, width, height)
        self.area = max(0, area)
        self.confidence = confidence
        self.timestamp = timestamp
        
        # Calculate center point
        x, y, w, h = bbox
        self.center = (x + w//2, y + h//2)
        
        # Validate dimensions
        if w <= 0 or h <= 0:
            logger.warning(f"Invalid motion event dimensions: {w}x{h}")

class MotionDetector:
    """Advanced motion detection with multiple algorithms and error handling."""
    
    def __init__(self, method: MotionDetectionMethod = MotionDetectionMethod.BACKGROUND_SUBTRACTION):
        self.method = method
        self.background_subtractor = None
        self.previous_frame = None
        self.reference_frame = None
        
        # Motion tracking
        self.motion_history = []
        self.motion_regions = []
        self._motion_lock = threading.Lock()
        
        # Performance metrics
        self.processing_times = []
        self.frame_count = 0
        self._stats_lock = threading.Lock()
        
        # Configuration
        self._load_config()
        
        self._initialize_detector()
        
        logger.info(f"Motion detector initialized with method: {method.value}")
    
    def _load_config(self):
        """Load configuration with defaults."""
        try:
            from utils.config import config
            self.threshold = config.detection.threshold
            self.min_area = config.detection.min_area
            self.gaussian_blur = config.detection.gaussian_blur
            self.morphology_kernel_size = config.detection.morphology_kernel_size
            self.learning_rate = config.detection.learning_rate
            self.background_subtractor_type = config.detection.background_subtractor
        except ImportError:
            logger.warning("Config not available, using defaults")
            self.threshold = 25
            self.min_area = 500
            self.gaussian_blur = (21, 21)
            self.morphology_kernel_size = (5, 5)
            self.learning_rate = 0.01
            self.background_subtractor_type = "MOG2"
    
    def _initialize_detector(self):
        """Initialize the motion detection algorithm with error handling."""
        try:
            if self.method == MotionDetectionMethod.BACKGROUND_SUBTRACTION:
                if self.background_subtractor_type == "MOG2":
                    self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=500,
                        varThreshold=self.threshold,
                        detectShadows=True
                    )
                elif self.background_subtractor_type == "KNN":
                    self.background_subtractor = cv2.createBackgroundSubtractorKNN(
                        history=500,
                        dist2Threshold=400,
                        detectShadows=True
                    )
                else:
                    logger.warning(f"Unknown background subtractor: {self.background_subtractor_type}, using MOG2")
                    self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=500,
                        varThreshold=self.threshold,
                        detectShadows=True
                    )
                
                logger.info(f"Background subtractor initialized: {self.background_subtractor_type}")
            
            # Initialize optical flow if needed
            if self.method == MotionDetectionMethod.OPTICAL_FLOW:
                self.lk_params = dict(
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                self.feature_params = dict(
                    maxCorners=100,
                    qualityLevel=0.3,
                    minDistance=7,
                    blockSize=7
                )
                logger.info("Optical flow parameters initialized")
                
        except Exception as e:
            logger.error(f"Error initializing motion detector: {e}")
            # Fallback to simple frame differencing
            self.method = MotionDetectionMethod.FRAME_DIFFERENCING
            logger.info("Falling back to frame differencing method")
    
    def detect_motion(self, frame: np.ndarray) -> List[MotionEvent]:
        """
        Detect motion in the given frame with comprehensive error handling.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of MotionEvent objects
        """
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided to detect_motion")
            return []
        
        start_time = time.time()
        
        try:
            # Validate frame
            if len(frame.shape) not in [2, 3]:
                logger.warning(f"Invalid frame shape: {frame.shape}")
                return []
            
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                logger.warning(f"Invalid frame dimensions: {width}x{height}")
                return []
            
            # Detect motion based on method
            if self.method == MotionDetectionMethod.BACKGROUND_SUBTRACTION:
                events = self._detect_background_subtraction(frame)
            elif self.method == MotionDetectionMethod.FRAME_DIFFERENCING:
                events = self._detect_frame_differencing(frame)
            elif self.method == MotionDetectionMethod.OPTICAL_FLOW:
                events = self._detect_optical_flow(frame)
            elif self.method == MotionDetectionMethod.HYBRID:
                events = self._detect_hybrid(frame)
            else:
                logger.error(f"Unknown motion detection method: {self.method}")
                events = []
            
            # Validate events
            valid_events = []
            for event in events:
                if self._validate_motion_event(event, (width, height)):
                    valid_events.append(event)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.processing_times.append(processing_time)
                self.frame_count += 1
                
                # Keep only recent processing times for performance monitoring
                if len(self.processing_times) > 100:
                    self.processing_times = self.processing_times[-100:]
            
            # Update motion history
            with self._motion_lock:
                motion_data = {
                    'timestamp': time.time(),
                    'event_count': len(valid_events),
                    'total_area': sum(event.area for event in valid_events),
                    'max_confidence': max([event.confidence for event in valid_events], default=0.0)
                }
                self.motion_history.append(motion_data)
                
                # Keep only recent history
                if len(self.motion_history) > 1000:
                    self.motion_history = self.motion_history[-1000:]
            
            return valid_events
            
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return []
    
    def _validate_motion_event(self, event: MotionEvent, frame_size: Tuple[int, int]) -> bool:
        """Validate a motion event."""
        try:
            width, height = frame_size
            x, y, w, h = event.bbox
            
            # Check bounds
            if not (0 <= x < width and 0 <= y < height):
                return False
            if not (x + w <= width and y + h <= height):
                return False
            
            # Check minimum size
            if w < 10 or h < 10:
                return False
            
            # Check area consistency
            if abs(event.area - (w * h)) > 0.1 * (w * h):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating motion event: {e}")
            return False
    
    def _detect_background_subtraction(self, frame: np.ndarray) -> List[MotionEvent]:
        """Detect motion using background subtraction with error handling."""
        if self.background_subtractor is None:
            logger.warning("Background subtractor not initialized")
            return []
        
        try:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame, learningRate=self.learning_rate)
            
            # Validate mask
            if fg_mask is None or fg_mask.size == 0:
                logger.warning("Invalid foreground mask")
                return []
            
            # Post-process the mask
            processed_mask = self._post_process_mask(fg_mask)
            if processed_mask is None:
                return []
            
            # Find contours
            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            return self._contours_to_events(contours)
            
        except Exception as e:
            logger.error(f"Background subtraction error: {e}")
            return []
    
    def _detect_frame_differencing(self, frame: np.ndarray) -> List[MotionEvent]:
        """Detect motion using frame differencing with error handling."""
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            # Apply Gaussian blur
            try:
                gray = cv2.GaussianBlur(gray, self.gaussian_blur, 0)
            except cv2.error as e:
                logger.warning(f"Gaussian blur error: {e}")
                # Use smaller blur if original fails
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            if self.previous_frame is None:
                self.previous_frame = gray
                return []
            
            # Compute frame difference
            frame_diff = cv2.absdiff(self.previous_frame, gray)
            
            # Threshold
            _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
            
            # Morphological operations
            try:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morphology_kernel_size)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            except Exception as e:
                logger.warning(f"Morphological operations error: {e}")
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            self.previous_frame = gray
            
            return self._contours_to_events(contours)
            
        except Exception as e:
            logger.error(f"Frame differencing error: {e}")
            return []
    
    def _detect_optical_flow(self, frame: np.ndarray) -> List[MotionEvent]:
        """Detect motion using optical flow with error handling."""
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            if self.previous_frame is None:
                self.previous_frame = gray
                return []
            
            # Detect features to track
            try:
                corners = cv2.goodFeaturesToTrack(self.previous_frame, mask=None, **self.feature_params)
            except Exception as e:
                logger.warning(f"Feature detection error: {e}")
                self.previous_frame = gray
                return []
            
            if corners is None or len(corners) == 0:
                self.previous_frame = gray
                return []
            
            # Calculate optical flow
            try:
                next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                    self.previous_frame, gray, corners, None, **self.lk_params
                )
            except Exception as e:
                logger.warning(f"Optical flow calculation error: {e}")
                self.previous_frame = gray
                return []
            
            # Select good points
            if len(status) == 0:
                self.previous_frame = gray
                return []
            
            good_new = next_pts[status == 1]
            good_old = corners[status == 1]
            
            if len(good_new) == 0:
                self.previous_frame = gray
                return []
            
            # Calculate motion vectors
            motion_vectors = good_new - good_old
            motion_magnitudes = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)
            
            # Filter significant motion
            significant_motion = motion_magnitudes > 2.0
            
            if not np.any(significant_motion):
                self.previous_frame = gray
                return []
            
            # Create bounding boxes around motion areas
            motion_points = good_new[significant_motion]
            events = self._points_to_events(motion_points, motion_magnitudes[significant_motion])
            
            self.previous_frame = gray
            
            return events
            
        except Exception as e:
            logger.error(f"Optical flow error: {e}")
            return []
    
    def _detect_hybrid(self, frame: np.ndarray) -> List[MotionEvent]:
        """Combine multiple detection methods with error handling."""
        try:
            bg_events = self._detect_background_subtraction(frame)
            fd_events = self._detect_frame_differencing(frame)
            
            # Merge events from different methods
            all_events = bg_events + fd_events
            
            if not all_events:
                return []
            
            # Remove duplicates and merge overlapping events
            merged_events = self._merge_overlapping_events(all_events)
            
            return merged_events
            
        except Exception as e:
            logger.error(f"Hybrid detection error: {e}")
            return []
    
    def _post_process_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Post-process the foreground mask with error handling."""
        try:
            if mask is None or mask.size == 0:
                return None
            
            # Remove noise
            processed_mask = cv2.medianBlur(mask, 5)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morphology_kernel_size)
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
            
            return processed_mask
            
        except Exception as e:
            logger.warning(f"Mask post-processing error: {e}")
            return mask
    
    def _contours_to_events(self, contours: List) -> List[MotionEvent]:
        """Convert contours to motion events with validation."""
        events = []
        current_time = time.time()
        
        try:
            for contour in contours:
                if contour is None or len(contour) < 3:
                    continue
                
                try:
                    area = cv2.contourArea(contour)
                    
                    if area > self.min_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Validate bounding rectangle
                        if w <= 0 or h <= 0:
                            continue
                        
                        # Calculate confidence based on area and shape
                        confidence = min(1.0, area / (self.min_area * 5))
                        
                        # Additional confidence based on aspect ratio
                        aspect_ratio = w / h if h > 0 else 1.0
                        if 0.2 <= aspect_ratio <= 5.0:  # Reasonable aspect ratios
                            confidence *= 1.0
                        else:
                            confidence *= 0.5  # Reduce confidence for unusual shapes
                        
                        event = MotionEvent(
                            bbox=(x, y, w, h),
                            area=area,
                            confidence=confidence,
                            timestamp=current_time
                        )
                        events.append(event)
                        
                except Exception as e:
                    logger.debug(f"Error processing contour: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"Error converting contours to events: {e}")
        
        return events
    
    def _points_to_events(self, points: np.ndarray, magnitudes: np.ndarray) -> List[MotionEvent]:
        """Convert motion points to events with clustering."""
        events = []
        current_time = time.time()
        
        try:
            if len(points) == 0:
                return events
            
            # Simple clustering by grouping nearby points
            clusters = self._cluster_points(points, magnitudes)
            
            for cluster in clusters:
                cluster_points = cluster['points']
                cluster_magnitudes = cluster['magnitudes']
                
                if len(cluster_points) == 0:
                    continue
                
                # Calculate bounding box for cluster
                min_x = int(np.min(cluster_points[:, 0]))
                max_x = int(np.max(cluster_points[:, 0]))
                min_y = int(np.min(cluster_points[:, 1]))
                max_y = int(np.max(cluster_points[:, 1]))
                
                # Add some padding
                padding = 10
                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                width = max_x - min_x + 2 * padding
                height = max_y - min_y + 2 * padding
                
                if width <= 0 or height <= 0:
                    continue
                
                # Calculate confidence based on motion magnitude
                avg_magnitude = np.mean(cluster_magnitudes)
                confidence = min(1.0, avg_magnitude / 10.0)
                
                area = width * height
                
                event = MotionEvent(
                    bbox=(min_x, min_y, width, height),
                    area=area,
                    confidence=confidence,
                    timestamp=current_time
                )
                events.append(event)
                
        except Exception as e:
            logger.warning(f"Error converting points to events: {e}")
        
        return events
    
    def _cluster_points(self, points: np.ndarray, magnitudes: np.ndarray, 
                       max_distance: float = 50.0) -> List[Dict]:
        """Simple clustering of motion points."""
        if len(points) == 0:
            return []
        
        try:
            clusters = []
            used = np.zeros(len(points), dtype=bool)
            
            for i, point in enumerate(points):
                if used[i]:
                    continue
                
                # Start new cluster
                cluster_points = [point]
                cluster_magnitudes = [magnitudes[i]]
                used[i] = True
                
                # Find nearby points
                for j, other_point in enumerate(points):
                    if used[j]:
                        continue
                    
                    distance = np.linalg.norm(point - other_point)
                    if distance <= max_distance:
                        cluster_points.append(other_point)
                        cluster_magnitudes.append(magnitudes[j])
                        used[j] = True
                
                clusters.append({
                    'points': np.array(cluster_points),
                    'magnitudes': np.array(cluster_magnitudes)
                })
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Error clustering points: {e}")
            # Return individual points as separate clusters
            return [{'points': np.array([point]), 'magnitudes': np.array([mag])} 
                   for point, mag in zip(points, magnitudes)]
    
    def _merge_overlapping_events(self, events: List[MotionEvent]) -> List[MotionEvent]:
        """Merge overlapping motion events with improved algorithm."""
        if len(events) <= 1:
            return events
        
        try:
            # Sort events by area (largest first)
            events.sort(key=lambda e: e.area, reverse=True)
            
            merged = []
            
            for event in events:
                merged_with_existing = False
                
                for i, merged_event in enumerate(merged):
                    if self._events_overlap(event, merged_event):
                        # Merge the events
                        new_bbox = self._merge_bboxes(merged_event.bbox, event.bbox)
                        new_area = max(merged_event.area, event.area)
                        new_confidence = max(merged_event.confidence, event.confidence)
                        new_timestamp = max(merged_event.timestamp, event.timestamp)
                        
                        # Create new merged event
                        merged_event_new = MotionEvent(
                            bbox=new_bbox,
                            area=new_area,
                            confidence=new_confidence,
                            timestamp=new_timestamp
                        )
                        
                        merged[i] = merged_event_new
                        merged_with_existing = True
                        break
                
                if not merged_with_existing:
                    merged.append(event)
            
            return merged
            
        except Exception as e:
            logger.warning(f"Error merging overlapping events: {e}")
            return events
    
    def _events_overlap(self, event1: MotionEvent, event2: MotionEvent, 
                       overlap_threshold: float = 0.3) -> bool:
        """Check if two events overlap significantly."""
        try:
            x1, y1, w1, h1 = event1.bbox
            x2, y2, w2, h2 = event2.bbox
            
            # Calculate overlap area
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            
            # Calculate union area
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - overlap_area
            
            if union_area <= 0:
                return False
            
            # Check if overlap is significant
            overlap_ratio = overlap_area / union_area
            return overlap_ratio >= overlap_threshold
            
        except Exception as e:
            logger.debug(f"Error checking event overlap: {e}")
            return False
    
    def _merge_bboxes(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Merge two bounding boxes."""
        try:
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            min_x = min(x1, x2)
            min_y = min(y1, y2)
            max_x = max(x1 + w1, x2 + w2)
            max_y = max(y1 + h1, y2 + h2)
            
            return (min_x, min_y, max_x - min_x, max_y - min_y)
            
        except Exception as e:
            logger.debug(f"Error merging bboxes: {e}")
            return bbox1  # Return first bbox if merge fails
    
    def draw_motion_overlay(self, frame: np.ndarray, events: List[MotionEvent]) -> np.ndarray:
        """Draw motion detection overlay on frame with error handling."""
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame for motion overlay")
            return frame
        
        try:
            overlay = frame.copy()
            
            for event in events:
                try:
                    x, y, w, h = event.bbox
                    
                    # Validate coordinates
                    height, width = frame.shape[:2]
                    if not (0 <= x < width and 0 <= y < height and 
                           x + w <= width and y + h <= height):
                        logger.debug(f"Invalid bbox for overlay: {event.bbox}")
                        continue
                    
                    # Color based on confidence
                    if event.confidence > 0.8:
                        color = (0, 0, 255)  # Red for high confidence
                    elif event.confidence > 0.5:
                        color = (0, 165, 255)  # Orange for medium confidence
                    else:
                        color = (0, 255, 255)  # Yellow for low confidence
                    
                    # Draw bounding box
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw confidence and area text
                    text = f"C:{event.confidence:.2f} A:{int(event.area)}"
                    
                    # Calculate text position
                    text_x = max(0, x)
                    text_y = max(20, y - 10)
                    
                    # Ensure text doesn't go outside frame
                    if text_y < 20:
                        text_y = y + h + 20
                    
                    try:
                        cv2.putText(overlay, text, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    except Exception as e:
                        logger.debug(f"Error drawing text: {e}")
                    
                    # Draw center point
                    center_x, center_y = event.center
                    if 0 <= center_x < width and 0 <= center_y < height:
                        cv2.circle(overlay, (center_x, center_y), 3, color, -1)
                    
                except Exception as e:
                    logger.debug(f"Error drawing motion event: {e}")
                    continue
            
            return overlay
            
        except Exception as e:
            logger.error(f"Error creating motion overlay: {e}")
            return frame
    
    def get_motion_statistics(self) -> Dict:
        """Get motion detection statistics with error handling."""
        try:
            with self._stats_lock:
                avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
                processing_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            
            with self._motion_lock:
                if not self.motion_history:
                    recent_history = []
                else:
                    recent_history = self.motion_history[-100:]  # Last 100 frames
            
            if recent_history:
                avg_events = np.mean([h['event_count'] for h in recent_history])
                avg_area = np.mean([h['total_area'] for h in recent_history])
                max_confidence = max([h['max_confidence'] for h in recent_history])
            else:
                avg_events = 0
                avg_area = 0
                max_confidence = 0
            
            return {
                'total_frames_processed': self.frame_count,
                'average_processing_time_ms': avg_processing_time * 1000,
                'average_events_per_frame': avg_events,
                'average_motion_area': avg_area,
                'max_confidence_recent': max_confidence,
                'method': self.method.value,
                'fps': processing_fps,
                'background_subtractor_type': getattr(self, 'background_subtractor_type', 'N/A'),
                'threshold': self.threshold,
                'min_area': self.min_area
            }
            
        except Exception as e:
            logger.error(f"Error getting motion statistics: {e}")
            return {
                'total_frames_processed': self.frame_count,
                'method': self.method.value,
                'error': str(e)
            }
    
    def reset_detector(self):
        """Reset the motion detector state with error handling."""
        try:
            with self._motion_lock:
                if self.background_subtractor:
                    # Reinitialize background subtractor
                    self._initialize_detector()
                
                self.previous_frame = None
                self.reference_frame = None
                self.motion_history.clear()
                self.motion_regions.clear()
            
            with self._stats_lock:
                self.processing_times.clear()
                self.frame_count = 0
            
            logger.info("Motion detector reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting motion detector: {e}")
    
    def is_significant_motion(self, events: List[MotionEvent], 
                            min_confidence: float = None) -> bool:
        """Check if there's significant motion in the events."""
        try:
            if not events:
                return False
            
            if min_confidence is None:
                min_confidence = 0.7  # Default threshold
            
            significant_events = [e for e in events if e.confidence >= min_confidence]
            
            if not significant_events:
                return False
            
            total_area = sum(event.area for event in significant_events)
            
            # Consider significant if motion covers substantial area
            # Assume a reasonable frame size for calculation
            frame_area = 640 * 480  # Default assumption
            motion_ratio = total_area / frame_area
            
            return motion_ratio > 0.05  # 5% of frame
            
        except Exception as e:
            logger.warning(f"Error checking significant motion: {e}")
            return False
    
    def get_motion_density_map(self, frame_shape: Tuple[int, int], 
                              events: List[MotionEvent]) -> np.ndarray:
        """Generate a motion density heatmap with error handling."""
        try:
            height, width = frame_shape
            density_map = np.zeros((height, width), dtype=np.float32)
            
            for event in events:
                try:
                    x, y, w, h = event.bbox
                    
                    # Validate coordinates
                    if not (0 <= x < width and 0 <= y < height and 
                           x + w <= width and y + h <= height):
                        continue
                    
                    # Add gaussian blob for motion area
                    center_x, center_y = event.center
                    
                    # Create gaussian kernel
                    sigma = max(w, h) / 4
                    kernel_size = int(sigma * 3)
                    
                    if kernel_size > 0 and kernel_size < min(width, height) // 2:
                        y_coords, x_coords = np.ogrid[:kernel_size*2+1, :kernel_size*2+1]
                        kernel = np.exp(-((x_coords - kernel_size)**2 + 
                                        (y_coords - kernel_size)**2) / (2 * sigma**2))
                        kernel = kernel * event.confidence
                        
                        # Add to density map
                        start_y = max(0, center_y - kernel_size)
                        end_y = min(height, center_y + kernel_size + 1)
                        start_x = max(0, center_x - kernel_size)
                        end_x = min(width, center_x + kernel_size + 1)
                        
                        kernel_start_y = max(0, kernel_size - center_y)
                        kernel_end_y = kernel_start_y + (end_y - start_y)
                        kernel_start_x = max(0, kernel_size - center_x)
                        kernel_end_x = kernel_start_x + (end_x - start_x)
                        
                        if (kernel_end_y > kernel_start_y and kernel_end_x > kernel_start_x and
                            kernel_end_y <= kernel.shape[0] and kernel_end_x <= kernel.shape[1]):
                            density_map[start_y:end_y, start_x:end_x] += \
                                kernel[kernel_start_y:kernel_end_y, kernel_start_x:kernel_end_x]
                    
                except Exception as e:
                    logger.debug(f"Error adding motion to density map: {e}")
                    continue
            
            return density_map
            
        except Exception as e:
            logger.error(f"Error creating motion density map: {e}")
            return np.zeros(frame_shape, dtype=np.float32)
    
    def update_parameters(self, **kwargs):
        """Update motion detection parameters at runtime."""
        try:
            if 'threshold' in kwargs:
                new_threshold = kwargs['threshold']
                if 0 <= new_threshold <= 255:
                    self.threshold = new_threshold
                    logger.info(f"Updated threshold to: {new_threshold}")
                    
                    # Update background subtractor if applicable
                    if self.background_subtractor and hasattr(self.background_subtractor, 'setVarThreshold'):
                        self.background_subtractor.setVarThreshold(new_threshold)
                        
            if 'min_area' in kwargs:
                new_min_area = kwargs['min_area']
                if new_min_area > 0:
                    self.min_area = new_min_area
                    logger.info(f"Updated min_area to: {new_min_area}")
                    
            if 'learning_rate' in kwargs:
                new_learning_rate = kwargs['learning_rate']
                if 0.0 <= new_learning_rate <= 1.0:
                    self.learning_rate = new_learning_rate
                    logger.info(f"Updated learning_rate to: {new_learning_rate}")
                    
        except Exception as e:
            logger.error(f"Error updating motion detection parameters: {e}")
    
    def get_detector_status(self) -> Dict:
        """Get current detector status and health information."""
        try:
            status = {
                'method': self.method.value,
                'is_initialized': True,
                'frame_count': self.frame_count,
                'has_background_subtractor': self.background_subtractor is not None,
                'has_previous_frame': self.previous_frame is not None,
                'motion_history_length': len(self.motion_history) if hasattr(self, 'motion_history') else 0,
                'processing_times_available': len(self.processing_times) if hasattr(self, 'processing_times') else 0
            }
            
            # Add method-specific status
            if self.method == MotionDetectionMethod.BACKGROUND_SUBTRACTION:
                status['background_subtractor_type'] = getattr(self, 'background_subtractor_type', 'Unknown')
                
            return status
            
        except Exception as e:
            logger.error(f"Error getting detector status: {e}")
            return {'error': str(e), 'is_initialized': False}
    
    def __del__(self):
        """Cleanup resources when detector is destroyed."""
        try:
            if hasattr(self, 'background_subtractor') and self.background_subtractor:
                del self.background_subtractor
        except Exception as e:
            logger.debug(f"Error during motion detector cleanup: {e}")  