"""
Anomaly detection module for surveillance system.
Detects unusual patterns, behaviors, and events using heuristics and lightweight ML.
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import time
from collections import deque
import pickle
import os

from detection.motion_detector import MotionEvent
from face_recognition.face_detector import FaceDetection
from utils.config import config
from utils.logger import logger

class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    UNUSUAL_MOTION = "unusual_motion"
    LOITERING = "loitering"
    INTRUSION = "intrusion"
    ABANDONED_OBJECT = "abandoned_object"
    CROWD_FORMATION = "crowd_formation"
    UNUSUAL_BEHAVIOR = "unusual_behavior"
    FACE_UNKNOWN = "face_unknown"
    MULTIPLE_FACES = "multiple_faces"
    NO_MOTION_EXPECTED = "no_motion_expected"
    RAPID_MOTION = "rapid_motion"

@dataclass
class AnomalyEvent:
    """Represents a detected anomaly."""
    anomaly_type: AnomalyType
    confidence: float
    timestamp: float
    location: Tuple[int, int]  # (x, y) center point
    bbox: Tuple[int, int, int, int] = None  # (x, y, width, height)
    metadata: Dict[str, Any] = None
    description: str = ""
    severity: float = 1.0  # 0.0 to 1.0, higher is more severe

class BaselineProfile:
    """Stores baseline behavior patterns for comparison."""
    
    def __init__(self):
        self.motion_patterns = {
            'typical_motion_areas': [],
            'typical_motion_times': [],
            'typical_motion_intensity': 0.0,
            'motion_frequency': 0.0
        }
        
        self.face_patterns = {
            'typical_face_count': 0,
            'known_face_frequency': 0.0,
            'face_locations': []
        }
        
        self.temporal_patterns = {
            'active_hours': [],
            'quiet_hours': [],
            'day_of_week_patterns': {}
        }
        
        self.learned_samples = 0
        self.last_update = time.time()

class AnomalyDetector:
    """Advanced anomaly detection system."""
    
    def __init__(self, learning_period: int = 3600):  # 1 hour learning period
        self.learning_period = learning_period
        self.is_learning = True
        self.baseline_profile = BaselineProfile()
        
        # Event history for pattern analysis
        self.motion_history = deque(maxlen=1000)
        self.face_history = deque(maxlen=1000)
        self.anomaly_history = deque(maxlen=500)
        
        # Tracking for complex anomalies
        self.object_tracker = {}
        self.loitering_tracker = {}
        self.motion_zones = {}
        
        # Thresholds and parameters
        self.motion_threshold = config.anomaly.motion_threshold
        self.time_window = config.anomaly.time_window
        self.min_confidence = config.anomaly.min_confidence
        
        # Performance metrics
        self.detection_times = []
        self.total_anomalies = 0
        
        # Load existing baseline if available
        self._load_baseline_profile()
        
        logger.info("Anomaly detector initialized")
    
    def detect_anomalies(self, motion_events: List[MotionEvent], 
                        face_detections: List[FaceDetection],
                        frame: np.ndarray) -> List[AnomalyEvent]:
        """
        Detect anomalies based on motion and face detection data.
        
        Args:
            motion_events: List of detected motion events
            face_detections: List of detected faces
            frame: Current frame for analysis
            
        Returns:
            List of detected anomalies
        """
        start_time = time.time()
        current_timestamp = time.time()
        
        # Update history
        self._update_history(motion_events, face_detections, current_timestamp)
        
        anomalies = []
        
        try:
            # Motion-based anomalies
            motion_anomalies = self._detect_motion_anomalies(motion_events, frame)
            anomalies.extend(motion_anomalies)
            
            # Face-based anomalies
            face_anomalies = self._detect_face_anomalies(face_detections, frame)
            anomalies.extend(face_anomalies)
            
            # Behavioral anomalies
            behavior_anomalies = self._detect_behavioral_anomalies(
                motion_events, face_detections, frame
            )
            anomalies.extend(behavior_anomalies)
            
            # Temporal anomalies
            temporal_anomalies = self._detect_temporal_anomalies(
                motion_events, face_detections
            )
            anomalies.extend(temporal_anomalies)
            
            # Update learning if in learning mode
            if self.is_learning:
                self._update_baseline(motion_events, face_detections)
            
            # Filter and prioritize anomalies
            filtered_anomalies = self._filter_and_prioritize_anomalies(anomalies)
            
            # Update tracking
            self.total_anomalies += len(filtered_anomalies)
            
            processing_time = time.time() - start_time
            self.detection_times.append(processing_time)
            
            # Clean up old timing data
            if len(self.detection_times) > 100:
                self.detection_times = self.detection_times[-100:]
            
            # Log significant anomalies
            for anomaly in filtered_anomalies:
                if anomaly.confidence > 0.7:
                    logger.info(f"Anomaly detected: {anomaly.anomaly_type.value} "
                              f"(confidence: {anomaly.confidence:.2f})")
            
            return filtered_anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return []
    
    def _detect_motion_anomalies(self, motion_events: List[MotionEvent], 
                                frame: np.ndarray) -> List[AnomalyEvent]:
        """Detect motion-based anomalies."""
        anomalies = []
        current_time = time.time()
        
        if not motion_events:
            # Check if motion was expected
            if self._is_motion_expected(current_time):
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.NO_MOTION_EXPECTED,
                    confidence=0.6,
                    timestamp=current_time,
                    location=(frame.shape[1]//2, frame.shape[0]//2),
                    description="No motion detected when activity was expected"
                ))
            return anomalies
        
        # Calculate motion intensity
        total_motion_area = sum(event.area for event in motion_events)
        frame_area = frame.shape[0] * frame.shape[1]
        motion_intensity = total_motion_area / frame_area
        
        # Unusual motion intensity
        if not self.is_learning and self.baseline_profile.motion_patterns['typical_motion_intensity'] > 0:
            baseline_intensity = self.baseline_profile.motion_patterns['typical_motion_intensity']
            intensity_ratio = motion_intensity / baseline_intensity
            
            if intensity_ratio > 3.0:  # 3x more motion than typical
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.RAPID_MOTION,
                    confidence=min(0.9, intensity_ratio / 5.0),
                    timestamp=current_time,
                    location=self._get_motion_center(motion_events),
                    description=f"Motion intensity {intensity_ratio:.1f}x higher than baseline"
                ))
        
        # Check for intrusion in restricted areas
        intrusion_anomalies = self._detect_intrusion(motion_events, frame)
        anomalies.extend(intrusion_anomalies)
        
        # Detect abandoned objects
        abandoned_anomalies = self._detect_abandoned_objects(motion_events, frame)
        anomalies.extend(abandoned_anomalies)
        
        return anomalies
    
    def _detect_face_anomalies(self, face_detections: List[FaceDetection], 
                              frame: np.ndarray) -> List[AnomalyEvent]:
        """Detect face-based anomalies."""
        anomalies = []
        current_time = time.time()
        
        if not face_detections:
            return anomalies
        
        # Multiple faces anomaly
        if len(face_detections) > 3:  # Configurable threshold
            center_x = np.mean([det.center[0] for det in face_detections])
            center_y = np.mean([det.center[1] for det in face_detections])
            
            anomalies.append(AnomalyEvent(
                anomaly_type=AnomalyType.MULTIPLE_FACES,
                confidence=min(0.9, len(face_detections) / 10.0),
                timestamp=current_time,
                location=(int(center_x), int(center_y)),
                description=f"Detected {len(face_detections)} faces simultaneously"
            ))
        
        # Unknown faces
        unknown_faces = [det for det in face_detections if det.face_id is None]
        if unknown_faces:
            for face in unknown_faces:
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.FACE_UNKNOWN,
                    confidence=face.confidence * 0.7,  # Lower confidence for unknown faces
                    timestamp=current_time,
                    location=face.center,
                    bbox=(face.left, face.top, face.width, face.height),
                    description="Unknown face detected"
                ))
        
        return anomalies
    
    def _detect_behavioral_anomalies(self, motion_events: List[MotionEvent], 
                                   face_detections: List[FaceDetection],
                                   frame: np.ndarray) -> List[AnomalyEvent]:
        """Detect behavioral pattern anomalies."""
        anomalies = []
        current_time = time.time()
        
        # Loitering detection
        loitering_anomalies = self._detect_loitering(face_detections, motion_events)
        anomalies.extend(loitering_anomalies)
        
        # Crowd formation detection
        if len(face_detections) >= 2:
            crowd_anomaly = self._detect_crowd_formation(face_detections)
            if crowd_anomaly:
                anomalies.append(crowd_anomaly)
        
        return anomalies
    
    def _detect_temporal_anomalies(self, motion_events: List[MotionEvent], 
                                  face_detections: List[FaceDetection]) -> List[AnomalyEvent]:
        """Detect time-based anomalies."""
        anomalies = []
        current_time = time.time()
        
        # Activity during quiet hours
        if self._is_quiet_hours(current_time) and (motion_events or face_detections):
            activity_score = len(motion_events) + len(face_detections) * 2
            
            anomalies.append(AnomalyEvent(
                anomaly_type=AnomalyType.UNUSUAL_BEHAVIOR,
                confidence=min(0.8, activity_score / 10.0),
                timestamp=current_time,
                location=(320, 240),  # Default center
                description=f"Activity detected during quiet hours (score: {activity_score})"
            ))
        
        return anomalies
    
    def _detect_loitering(self, face_detections: List[FaceDetection], 
                         motion_events: List[MotionEvent]) -> List[AnomalyEvent]:
        """Detect loitering behavior."""
        anomalies = []
        current_time = time.time()
        loitering_threshold = 300  # 5 minutes
        
        # Track face positions over time
        for face in face_detections:
            face_key = face.face_id or f"unknown_{hash(str(face.center))}"
            
            if face_key not in self.loitering_tracker:
                self.loitering_tracker[face_key] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'positions': [face.center],
                    'total_movement': 0.0
                }
            else:
                tracker = self.loitering_tracker[face_key]
                tracker['last_seen'] = current_time
                
                # Calculate movement
                if tracker['positions']:
                    last_pos = tracker['positions'][-1]
                    movement = np.sqrt((face.center[0] - last_pos[0])**2 + 
                                     (face.center[1] - last_pos[1])**2)
                    tracker['total_movement'] += movement
                
                tracker['positions'].append(face.center)
                
                # Keep only recent positions
                if len(tracker['positions']) > 100:
                    tracker['positions'] = tracker['positions'][-100:]
                
                # Check for loitering
                duration = current_time - tracker['first_seen']
                if duration > loitering_threshold:
                    avg_movement = tracker['total_movement'] / len(tracker['positions'])
                    
                    if avg_movement < 20:  # Low movement threshold
                        confidence = min(0.9, duration / (loitering_threshold * 2))
                        
                        anomalies.append(AnomalyEvent(
                            anomaly_type=AnomalyType.LOITERING,
                            confidence=confidence,
                            timestamp=current_time,
                            location=face.center,
                            description=f"Person loitering for {duration/60:.1f} minutes"
                        ))
        
        # Clean up old tracking data
        cutoff_time = current_time - loitering_threshold * 2
        self.loitering_tracker = {
            k: v for k, v in self.loitering_tracker.items() 
            if v['last_seen'] > cutoff_time
        }
        
        return anomalies
    
    def _detect_intrusion(self, motion_events: List[MotionEvent], 
                         frame: np.ndarray) -> List[AnomalyEvent]:
        """Detect intrusion into restricted areas."""
        # This is a simplified implementation
        # In practice, you would define restricted zones
        anomalies = []
        
        # Example: detect motion in corners (potential restricted areas)
        frame_height, frame_width = frame.shape[:2]
        restricted_zones = [
            (0, 0, frame_width//4, frame_height//4),  # Top-left
            (frame_width*3//4, 0, frame_width//4, frame_height//4),  # Top-right
        ]
        
        for event in motion_events:
            event_center = (event.bbox[0] + event.bbox[2]//2, 
                          event.bbox[1] + event.bbox[3]//2)
            
            for zone in restricted_zones:
                zone_x, zone_y, zone_w, zone_h = zone
                if (zone_x <= event_center[0] <= zone_x + zone_w and
                    zone_y <= event_center[1] <= zone_y + zone_h):
                    
                    anomalies.append(AnomalyEvent(
                        anomaly_type=AnomalyType.INTRUSION,
                        confidence=event.confidence * 0.8,
                        timestamp=time.time(),
                        location=event_center,
                        bbox=event.bbox,
                        description="Motion detected in restricted area"
                    ))
        
        return anomalies
    
    def _detect_abandoned_objects(self, motion_events: List[MotionEvent], 
                                 frame: np.ndarray) -> List[AnomalyEvent]:
        """Detect potentially abandoned objects."""
        # Simplified implementation - look for stationary objects
        anomalies = []
        current_time = time.time()
        
        # Track stationary areas
        for event in motion_events:
            if event.confidence > 0.3:  # Low confidence might indicate stationary object
                object_key = f"obj_{event.bbox[0]}_{event.bbox[1]}"
                
                if object_key not in self.object_tracker:
                    self.object_tracker[object_key] = {
                        'first_seen': current_time,
                        'bbox': event.bbox,
                        'confidence_sum': event.confidence
                    }
                else:
                    tracker = self.object_tracker[object_key]
                    duration = current_time - tracker['first_seen']
                    
                    if duration > 600:  # 10 minutes
                        anomalies.append(AnomalyEvent(
                            anomaly_type=AnomalyType.ABANDONED_OBJECT,
                            confidence=min(0.7, duration / 1200),  # Max confidence at 20 minutes
                            timestamp=current_time,
                            location=(event.bbox[0] + event.bbox[2]//2, 
                                    event.bbox[1] + event.bbox[3]//2),
                            bbox=event.bbox,
                            description=f"Stationary object for {duration/60:.1f} minutes"
                        ))
        
        return anomalies
    
    def _detect_crowd_formation(self, face_detections: List[FaceDetection]) -> Optional[AnomalyEvent]:
        """Detect crowd formation patterns."""
        if len(face_detections) < 3:
            return None
        
        # Calculate average distance between faces
        positions = [det.center for det in face_detections]
        total_distance = 0
        pair_count = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                 (positions[i][1] - positions[j][1])**2)
                total_distance += distance
                pair_count += 1
        
        avg_distance = total_distance / pair_count if pair_count > 0 else float('inf')
        
        # If faces are close together, consider it crowd formation
        if avg_distance < 150:  # Adjustable threshold
            center_x = np.mean([pos[0] for pos in positions])
            center_y = np.mean([pos[1] for pos in positions])
            
            confidence = min(0.8, len(face_detections) / 10.0)
            
            return AnomalyEvent(
                anomaly_type=AnomalyType.CROWD_FORMATION,
                confidence=confidence,
                timestamp=time.time(),
                location=(int(center_x), int(center_y)),
                description=f"Crowd of {len(face_detections)} people detected"
            )
        
        return None
    
    def _update_history(self, motion_events: List[MotionEvent], 
                       face_detections: List[FaceDetection], timestamp: float):
        """Update event history for pattern analysis."""
        # Motion history
        motion_data = {
            'timestamp': timestamp,
            'event_count': len(motion_events),
            'total_area': sum(event.area for event in motion_events),
            'avg_confidence': np.mean([event.confidence for event in motion_events]) if motion_events else 0
        }
        self.motion_history.append(motion_data)
        
        # Face history
        face_data = {
            'timestamp': timestamp,
            'face_count': len(face_detections),
            'known_faces': len([det for det in face_detections if det.face_id]),
            'unknown_faces': len([det for det in face_detections if not det.face_id])
        }
        self.face_history.append(face_data)
    
    def _update_baseline(self, motion_events: List[MotionEvent], 
                        face_detections: List[FaceDetection]):
        """Update baseline patterns during learning phase."""
        current_time = time.time()
        
        # Check if learning period is complete
        if current_time - self.baseline_profile.last_update > self.learning_period:
            self.is_learning = False
            self._save_baseline_profile()
            logger.info("Learning phase completed - baseline profile saved")
            return
        
        # Update motion patterns
        if motion_events:
            total_area = sum(event.area for event in motion_events)
            current_intensity = total_area / (640 * 480)  # Normalize by typical frame size
            
            # Running average
            samples = self.baseline_profile.learned_samples
            old_intensity = self.baseline_profile.motion_patterns['typical_motion_intensity']
            new_intensity = (old_intensity * samples + current_intensity) / (samples + 1)
            self.baseline_profile.motion_patterns['typical_motion_intensity'] = new_intensity
        
        # Update face patterns
        if face_detections:
            current_face_count = len(face_detections)
            samples = self.baseline_profile.learned_samples
            old_count = self.baseline_profile.face_patterns['typical_face_count']
            new_count = (old_count * samples + current_face_count) / (samples + 1)
            self.baseline_profile.face_patterns['typical_face_count'] = new_count
        
        self.baseline_profile.learned_samples += 1
    
    def _is_motion_expected(self, timestamp: float) -> bool:
        """Check if motion is expected at this time based on historical patterns."""
        if self.is_learning:
            return False
        
        # Simplified implementation - check if it's during active hours
        hour = time.localtime(timestamp).tm_hour
        return hour in self.baseline_profile.temporal_patterns.get('active_hours', [])
    
    def _is_quiet_hours(self, timestamp: float) -> bool:
        """Check if current time is during quiet hours."""
        hour = time.localtime(timestamp).tm_hour
        # Default quiet hours: 11 PM to 6 AM
        return hour >= 23 or hour <= 6
    
    def _get_motion_center(self, motion_events: List[MotionEvent]) -> Tuple[int, int]:
        """Get center point of motion events."""
        if not motion_events:
            return (320, 240)  # Default center
        
        centers = [(event.bbox[0] + event.bbox[2]//2, event.bbox[1] + event.bbox[3]//2) 
                  for event in motion_events]
        
        avg_x = int(np.mean([c[0] for c in centers]))
        avg_y = int(np.mean([c[1] for c in centers]))
        
        return (avg_x, avg_y)
    
    def _filter_and_prioritize_anomalies(self, anomalies: List[AnomalyEvent]) -> List[AnomalyEvent]:
        """Filter and prioritize anomalies based on confidence and severity."""
        # Filter by minimum confidence
        filtered = [a for a in anomalies if a.confidence >= self.min_confidence]
        
        # Remove duplicate anomalies (same type, close in time and space)
        unique_anomalies = []
        for anomaly in filtered:
            is_duplicate = False
            for existing in unique_anomalies:
                if (anomaly.anomaly_type == existing.anomaly_type and
                    abs(anomaly.timestamp - existing.timestamp) < 30 and  # 30 seconds
                    self._calculate_distance(anomaly.location, existing.location) < 100):
                    is_duplicate = True
                    # Keep the higher confidence anomaly
                    if anomaly.confidence > existing.confidence:
                        unique_anomalies.remove(existing)
                        unique_anomalies.append(anomaly)
                    break
            
            if not is_duplicate:
                unique_anomalies.append(anomaly)
        
        # Sort by confidence and severity
        unique_anomalies.sort(key=lambda x: (x.confidence * x.severity), reverse=True)
        
        # Limit number of anomalies to avoid spam
        return unique_anomalies[:5]
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _load_baseline_profile(self):
        """Load existing baseline profile from file."""
        try:
            profile_file = os.path.join(config.logging.output_dir, "baseline_profile.pkl")
            if os.path.exists(profile_file):
                with open(profile_file, 'rb') as f:
                    self.baseline_profile = pickle.load(f)
                self.is_learning = False
                logger.info("Baseline profile loaded")
            else:
                logger.info("No existing baseline profile found - starting learning phase")
        except Exception as e:
            logger.error(f"Failed to load baseline profile: {e}")
    
    def _save_baseline_profile(self):
        """Save the learned baseline profile to file."""
        try:
            profile_file = os.path.join(config.logging.output_dir, "baseline_profile.pkl")
            with open(profile_file, 'wb') as f:
                pickle.dump(self.baseline_profile, f)
            logger.info("Baseline profile saved")
        except Exception as e:
            logger.error(f"Failed to save baseline profile: {e}")
    
    def draw_anomaly_overlay(self, frame: np.ndarray, anomalies: List[AnomalyEvent]) -> np.ndarray:
        """Draw anomaly detection overlay on frame."""
        overlay = frame.copy()
        
        # Color map for different anomaly types
        color_map = {
            AnomalyType.UNUSUAL_MOTION: (0, 0, 255),      # Red
            AnomalyType.LOITERING: (0, 165, 255),         # Orange
            AnomalyType.INTRUSION: (0, 0, 139),           # Dark red
            AnomalyType.ABANDONED_OBJECT: (0, 255, 255),  # Yellow
            AnomalyType.CROWD_FORMATION: (255, 0, 255),   # Magenta
            AnomalyType.FACE_UNKNOWN: (255, 0, 0),        # Blue
            AnomalyType.MULTIPLE_FACES: (255, 255, 0),    # Cyan
            AnomalyType.RAPID_MOTION: (0, 69, 255),       # Red-orange
        }
        
        for anomaly in anomalies:
            color = color_map.get(anomaly.anomaly_type, (128, 128, 128))
            
            # Draw marker at anomaly location
            center = anomaly.location
            radius = int(20 + anomaly.confidence * 30)
            
            cv2.circle(overlay, center, radius, color, 3)
            cv2.circle(overlay, center, 5, color, -1)
            
            # Draw bounding box if available
            if anomaly.bbox:
                x, y, w, h = anomaly.bbox
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw text label
            text = f"{anomaly.anomaly_type.value}"
            confidence_text = f"({anomaly.confidence:.2f})"
            
            # Calculate text position
            text_x = max(10, center[0] - 50)
            text_y = max(30, center[1] - radius - 10)
            
            # Background for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(overlay, (text_x, text_y - text_size[1] - 5), 
                         (text_x + text_size[0], text_y + 5), color, -1)
            
            # Text
            cv2.putText(overlay, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Confidence text
            conf_y = text_y + 20
            cv2.putText(overlay, confidence_text, (text_x, conf_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        return overlay
    
    def get_anomaly_statistics(self) -> Dict:
        """Get anomaly detection statistics."""
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        
        # Recent anomaly counts by type
        recent_time = time.time() - 3600  # Last hour
        recent_anomalies = [a for a in self.anomaly_history if a.timestamp > recent_time]
        
        anomaly_counts = {}
        for anomaly in recent_anomalies:
            anomaly_type = anomaly.anomaly_type.value
            anomaly_counts[anomaly_type] = anomaly_counts.get(anomaly_type, 0) + 1
        
        return {
            'total_anomalies_detected': self.total_anomalies,
            'recent_anomalies_count': len(recent_anomalies),
            'anomaly_types_detected': anomaly_counts,
            'is_learning': self.is_learning,
            'learning_samples': self.baseline_profile.learned_samples,
            'average_detection_time_ms': avg_detection_time * 1000,
            'detection_fps': 1.0 / avg_detection_time if avg_detection_time > 0 else 0
        }
    
    def reset_learning(self):
        """Reset the learning process."""
        self.is_learning = True
        self.baseline_profile = BaselineProfile()
        logger.info("Anomaly detector learning reset")
    
    def set_learning_mode(self, enable: bool):
        """Enable or disable learning mode."""
        self.is_learning = enable
        if enable:
            logger.info("Anomaly detector learning enabled")
        else:
            logger.info("Anomaly detector learning disabled")
    
    def update_thresholds(self, motion_threshold: float = None, 
                         min_confidence: float = None, time_window: int = None):
        """Update anomaly detection thresholds."""
        if motion_threshold is not None:
            self.motion_threshold = max(0.0, min(1.0, motion_threshold))
        if min_confidence is not None:
            self.min_confidence = max(0.0, min(1.0, min_confidence))
        if time_window is not None:
            self.time_window = max(1, time_window)
        
        logger.info(f"Updated thresholds: motion={self.motion_threshold}, "
                   f"confidence={self.min_confidence}, window={self.time_window}")
    
    def get_baseline_summary(self) -> Dict:
        """Get a summary of the learned baseline patterns."""
        if self.baseline_profile.learned_samples == 0:
            return {'status': 'No baseline learned yet'}
        
        return {
            'learned_samples': self.baseline_profile.learned_samples,
            'typical_motion_intensity': self.baseline_profile.motion_patterns['typical_motion_intensity'],
            'typical_face_count': self.baseline_profile.face_patterns['typical_face_count'],
            'learning_complete': not self.is_learning,
            'last_update': time.ctime(self.baseline_profile.last_update)
        }