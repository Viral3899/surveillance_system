"""
Advanced motion detection module for surveillance system.
Optimized for real-time processing with multiple detection methods.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum
import time

from utils.config import config
from utils.logger import logger

class MotionDetectionMethod(Enum):
    """Available motion detection methods."""
    BACKGROUND_SUBTRACTION = "background_subtraction"
    FRAME_DIFFERENCING = "frame_differencing"
    OPTICAL_FLOW = "optical_flow"
    HYBRID = "hybrid"

class MotionEvent:
    """Represents a detected motion event."""
    
    def __init__(self, bbox: Tuple[int, int, int, int], area: float, 
                 confidence: float, timestamp: float):
        self.bbox = bbox  # (x, y, width, height)
        self.area = area
        self.confidence = confidence
        self.timestamp = timestamp
        self.center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)

class MotionDetector:
    """Advanced motion detection with multiple algorithms."""
    
    def __init__(self, method: MotionDetectionMethod = MotionDetectionMethod.BACKGROUND_SUBTRACTION):
        self.method = method
        self.background_subtractor = None
        self.previous_frame = None
        self.reference_frame = None
        
        # Motion tracking
        self.motion_history = []
        self.motion_regions = []
        
        # Performance metrics
        self.processing_times = []
        self.frame_count = 0
        
        self._initialize_detector()
        
        logger.info(f"Motion detector initialized with method: {method.value}")
    
    def _initialize_detector(self):
        """Initialize the motion detection algorithm."""
        if self.method == MotionDetectionMethod.BACKGROUND_SUBTRACTION:
            if config.detection.background_subtractor == "MOG2":
                self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500,
                    varThreshold=config.detection.threshold,
                    detectShadows=True
                )
            elif config.detection.background_subtractor == "KNN":
                self.background_subtractor = cv2.createBackgroundSubtractorKNN(
                    history=500,
                    dist2Threshold=400,
                    detectShadows=True
                )
        
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
    
    def detect_motion(self, frame: np.ndarray) -> List[MotionEvent]:
        """
        Detect motion in the given frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of MotionEvent objects
        """
        start_time = time.time()
        
        try:
            if self.method == MotionDetectionMethod.BACKGROUND_SUBTRACTION:
                events = self._detect_background_subtraction(frame)
            elif self.method == MotionDetectionMethod.FRAME_DIFFERENCING:
                events = self._detect_frame_differencing(frame)
            elif self.method == MotionDetectionMethod.OPTICAL_FLOW:
                events = self._detect_optical_flow(frame)
            elif self.method == MotionDetectionMethod.HYBRID:
                events = self._detect_hybrid(frame)
            else:
                events = []
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.frame_count += 1
            
            # Keep only recent processing times for performance monitoring
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]
            
            # Update motion history
            self.motion_history.append({
                'timestamp': time.time(),
                'event_count': len(events),
                'total_area': sum(event.area for event in events)
            })
            
            # Keep only recent history
            if len(self.motion_history) > 1000:
                self.motion_history = self.motion_history[-1000:]
            
            return events
            
        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return []
    
    def _detect_background_subtraction(self, frame: np.ndarray) -> List[MotionEvent]:
        """Detect motion using background subtraction."""
        if self.background_subtractor is None:
            return []
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame, learningRate=config.detection.learning_rate)
        
        # Post-process the mask
        fg_mask = self._post_process_mask(fg_mask)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return self._contours_to_events(contours)
    
    def _detect_frame_differencing(self, frame: np.ndarray) -> List[MotionEvent]:
        """Detect motion using frame differencing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, config.detection.gaussian_blur, 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return []
        
        # Compute frame difference
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        
        # Threshold
        _, thresh = cv2.threshold(frame_diff, config.detection.threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.detection.morphology_kernel_size)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.previous_frame = gray
        
        return self._contours_to_events(contours)
    
    def _detect_optical_flow(self, frame: np.ndarray) -> List[MotionEvent]:
        """Detect motion using optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return []
        
        # Detect features to track
        corners = cv2.goodFeaturesToTrack(self.previous_frame, mask=None, **self.feature_params)
        
        if corners is None or len(corners) == 0:
            self.previous_frame = gray
            return []
        
        # Calculate optical flow
        next_pts, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_frame, gray, corners, None, **self.lk_params
        )
        
        # Select good points
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
    
    def _detect_hybrid(self, frame: np.ndarray) -> List[MotionEvent]:
        """Combine multiple detection methods."""
        bg_events = self._detect_background_subtraction(frame)
        fd_events = self._detect_frame_differencing(frame)
        
        # Merge events from different methods
        all_events = bg_events + fd_events
        
        # Remove duplicates and merge overlapping events
        merged_events = self._merge_overlapping_events(all_events)
        
        return merged_events
    
    def _post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Post-process the foreground mask."""
        # Remove noise
        mask = cv2.medianBlur(mask, 5)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.detection.morphology_kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _contours_to_events(self, contours: List) -> List[MotionEvent]:
        """Convert contours to motion events."""
        events = []
        current_time = time.time()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > config.detection.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on area and shape
                confidence = min(1.0, area / (config.detection.min_area * 5))
                
                event = MotionEvent(
                    bbox=(x, y, w, h),
                    area=area,
                    confidence=confidence,
                    timestamp=current_time
                )
                events.append(event)
        
        return events
    
    def _points_to_events(self, points: np.ndarray, magnitudes: np.ndarray) -> List[MotionEvent]:
        """Convert motion points to events."""
        events = []
        current_time = time.time()
        
        # Group nearby points
        for i, (point, magnitude) in enumerate(zip(points, magnitudes)):
            x, y = int(point[0]), int(point[1])
            
            # Create bounding box around motion point
            size = int(magnitude * 2)
            bbox = (max(0, x - size), max(0, y - size), size * 2, size * 2)
            
            confidence = min(1.0, magnitude / 10.0)
            
            event = MotionEvent(
                bbox=bbox,
                area=size * size,
                confidence=confidence,
                timestamp=current_time
            )
            events.append(event)
        
        return events
    
    def _merge_overlapping_events(self, events: List[MotionEvent]) -> List[MotionEvent]:
        """Merge overlapping motion events."""
        if len(events) <= 1:
            return events
        
        # Sort events by area (largest first)
        events.sort(key=lambda e: e.area, reverse=True)
        
        merged = []
        
        for event in events:
            overlapped = False
            
            for merged_event in merged:
                if self._events_overlap(event, merged_event):
                    # Merge the events
                    merged_event.bbox = self._merge_bboxes(merged_event.bbox, event.bbox)
                    merged_event.area = max(merged_event.area, event.area)
                    merged_event.confidence = max(merged_event.confidence, event.confidence)
                    overlapped = True
                    break
            
            if not overlapped:
                merged.append(event)
        
        return merged
    
    def _events_overlap(self, event1: MotionEvent, event2: MotionEvent) -> bool:
        """Check if two events overlap."""
        x1, y1, w1, h1 = event1.bbox
        x2, y2, w2, h2 = event2.bbox
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    def _merge_bboxes(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Merge two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        min_x = min(x1, x2)
        min_y = min(y1, y2)
        max_x = max(x1 + w1, x2 + w2)
        max_y = max(y1 + h1, y2 + h2)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def draw_motion_overlay(self, frame: np.ndarray, events: List[MotionEvent]) -> np.ndarray:
        """Draw motion detection overlay on frame."""
        overlay = frame.copy()
        
        for event in events:
            x, y, w, h = event.bbox
            
            # Color based on confidence
            if event.confidence > 0.8:
                color = (0, 0, 255)  # Red for high confidence
            elif event.confidence > 0.5:
                color = (0, 165, 255)  # Orange for medium confidence
            else:
                color = (0, 255, 255)  # Yellow for low confidence
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence and area
            text = f"C:{event.confidence:.2f} A:{int(event.area)}"
            cv2.putText(overlay, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1, cv2.LINE_AA)
        
        return overlay
    
    def get_motion_statistics(self) -> Dict:
        """Get motion detection statistics."""
        if not self.motion_history:
            return {}
        
        recent_history = self.motion_history[-100:]  # Last 100 frames
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_events = np.mean([h['event_count'] for h in recent_history])
        avg_area = np.mean([h['total_area'] for h in recent_history])
        
        return {
            'total_frames_processed': self.frame_count,
            'average_processing_time_ms': avg_processing_time * 1000,
            'average_events_per_frame': avg_events,
            'average_motion_area': avg_area,
            'method': self.method.value,
            'fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        }
    
    def reset_detector(self):
        """Reset the motion detector state."""
        if self.background_subtractor:
            self.background_subtractor = None
            self._initialize_detector()
        
        self.previous_frame = None
        self.reference_frame = None
        self.motion_history.clear()
        self.motion_regions.clear()
        
        logger.info("Motion detector reset")
    
    def is_significant_motion(self, events: List[MotionEvent], 
                            min_confidence: float = None) -> bool:
        """Check if there's significant motion in the events."""
        min_confidence = min_confidence or config.anomaly.min_confidence
        
        significant_events = [e for e in events if e.confidence >= min_confidence]
        
        if not significant_events:
            return False
        
        total_area = sum(event.area for event in significant_events)
        frame_area = config.camera.resolution[0] * config.camera.resolution[1]
        
        # Consider significant if motion covers more than threshold of frame
        motion_ratio = total_area / frame_area
        return motion_ratio > config.anomaly.motion_threshold
    
    def get_motion_density_map(self, frame_shape: Tuple[int, int], 
                              events: List[MotionEvent]) -> np.ndarray:
        """Generate a motion density heatmap."""
        density_map = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        
        for event in events:
            x, y, w, h = event.bbox
            
            # Add gaussian blob for motion area
            center_x, center_y = x + w // 2, y + h // 2
            
            # Create gaussian kernel
            sigma = max(w, h) / 4
            kernel_size = int(sigma * 3)
            
            if kernel_size > 0:
                y_coords, x_coords = np.ogrid[:kernel_size*2+1, :kernel_size*2+1]
                kernel = np.exp(-((x_coords - kernel_size)**2 + (y_coords - kernel_size)**2) / (2 * sigma**2))
                kernel = kernel * event.confidence
                
                # Add to density map
                start_y = max(0, center_y - kernel_size)
                end_y = min(frame_shape[0], center_y + kernel_size + 1)
                start_x = max(0, center_x - kernel_size)
                end_x = min(frame_shape[1], center_x + kernel_size + 1)
                
                kernel_start_y = max(0, kernel_size - center_y)
                kernel_end_y = kernel_start_y + (end_y - start_y)
                kernel_start_x = max(0, kernel_size - center_x)
                kernel_end_x = kernel_start_x + (end_x - start_x)
                
                if kernel_end_y > kernel_start_y and kernel_end_x > kernel_start_x:
                    density_map[start_y:end_y, start_x:end_x] += \
                        kernel[kernel_start_y:kernel_end_y, kernel_start_x:kernel_end_x]
        
        return density_map  