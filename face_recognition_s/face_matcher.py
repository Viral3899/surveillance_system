"""
Face recognition and matching module for surveillance system.
Manages known face database and performs real-time matching.
"""
import os
import pickle
import numpy as np
import face_recognition as fr
import cv2
import shutil
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass

from .face_detector import FaceDetection
from utils.config import config
from utils.logger import logger

@dataclass
class KnownFace:
    """Represents a known face in the database."""
    name: str
    encoding: np.ndarray
    image_path: str
    added_timestamp: float
    last_seen: float = 0
    recognition_count: int = 0

class FaceMatcher:
    """Face recognition and matching system."""
    
    def __init__(self):
        self.known_faces: Dict[str, KnownFace] = {}
        self.face_encodings: List[np.ndarray] = []
        self.face_names: List[str] = []
        self.tolerance = config.face.tolerance
        
        # Performance tracking
        self.matching_times = []
        self.recognition_history = []
        
        # Load existing face database
        self._load_face_database()
        
        logger.info(f"Face matcher initialized with {len(self.known_faces)} known faces")
    
    def _load_face_database(self):
        """Load known faces from the database file."""
        encodings_file = os.path.join(
            config.face.face_gallery_path,
            config.face.encodings_file
        )
        
        if os.path.exists(encodings_file):
            try:
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('known_faces', {})
                    
                # Rebuild encoding lists for faster matching
                self._rebuild_encoding_lists()
                
                logger.info(f"Loaded {len(self.known_faces)} faces from database")
                
            except Exception as e:
                logger.error(f"Failed to load face database: {e}")
                self.known_faces = {}
        else:
            # Try to build database from gallery images
            self._build_face_database_from_gallery()
    
    def _build_face_database_from_gallery(self):
        """Build face database from images in the gallery folder."""
        gallery_path = config.face.face_gallery_path
        
        if not os.path.exists(gallery_path):
            logger.warning(f"Gallery path does not exist: {gallery_path}")
            return
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        face_count = 0
        
        for filename in os.listdir(gallery_path):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(gallery_path, filename)
                
                # Use filename (without extension) as person name
                person_name = os.path.splitext(filename)[0]
                
                if self.add_face_from_image(image_path, person_name):
                    face_count += 1
        
        if face_count > 0:
            self.save_face_database()
            logger.info(f"Built face database from gallery: {face_count} faces")
    
    def add_face_from_image(self, image_path: str, person_name: str) -> bool:
        """Add a new face to the database from an image file."""
        try:
            # Load and process image
            image = fr.load_image_file(image_path)
            
            # Find face encodings
            face_encodings = fr.face_encodings(image)
            
            if not face_encodings:
                logger.warning(f"No face found in {image_path}")
                return False
            
            if len(face_encodings) > 1:
                logger.warning(f"Multiple faces found in {image_path}, using first one")
            
            # Use the first face encoding
            face_encoding = face_encodings[0]
            
            # Create known face entry
            known_face = KnownFace(
                name=person_name,
                encoding=face_encoding,
                image_path=image_path,
                added_timestamp=time.time()
            )
            
            # Add to database
            self.known_faces[person_name] = known_face
            self._rebuild_encoding_lists()
            
            logger.info(f"Added face for {person_name} from {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add face from {image_path}: {e}")
            return False
    
    def add_face_from_detection(self, frame: np.ndarray, detection: FaceDetection, 
                              person_name: str) -> bool:
        """Add a new face to the database from a detection."""
        try:
            if detection.encoding is None:
                logger.warning("Detection has no encoding")
                return False
            
            # Save face image
            face_roi = self._extract_face_roi(frame, detection)
            image_filename = f"{person_name}_{int(time.time())}.jpg"
            image_path = os.path.join(config.face.face_gallery_path, image_filename)
            cv2.imwrite(image_path, face_roi)
            
            # Create known face entry
            known_face = KnownFace(
                name=person_name,
                encoding=detection.encoding,
                image_path=image_path,
                added_timestamp=time.time()
            )
            
            # Add to database
            self.known_faces[person_name] = known_face
            self._rebuild_encoding_lists()
            
            logger.info(f"Added face for {person_name} from detection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add face from detection: {e}")
            return False
    
    def _extract_face_roi(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """Extract face region from frame."""
        top, right, bottom, left = detection.bbox
        return frame[top:bottom, left:right]
    
    def match_faces(self, detections: List[FaceDetection]) -> List[FaceDetection]:
        """Match detected faces against known faces database."""
        if not self.known_faces or not detections:
            return detections
        
        start_time = time.time()
        
        try:
            # Extract encodings from detections
            detection_encodings = []
            valid_detections = []
            
            for detection in detections:
                if detection.encoding is not None:
                    detection_encodings.append(detection.encoding)
                    valid_detections.append(detection)
            
            if not detection_encodings:
                return detections
            
            # Compare against known faces
            for i, detection_encoding in enumerate(detection_encodings):
                detection = valid_detections[i]
                
                if len(self.face_encodings) == 0:
                    continue
                
                # Compare against all known faces
                matches = fr.compare_faces(
                    self.face_encodings, 
                    detection_encoding,
                    tolerance=self.tolerance
                )
                
                # Calculate face distances for confidence scoring
                face_distances = fr.face_distance(
                    self.face_encodings, 
                    detection_encoding
                )
                
                # Find best match
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
                
                if (best_match_index is not None and 
                    matches[best_match_index] and 
                    face_distances[best_match_index] < self.tolerance):
                    
                    # Match found
                    matched_name = self.face_names[best_match_index]
                    confidence = 1.0 - face_distances[best_match_index]
                    
                    # Update detection with match info
                    detection.face_id = matched_name
                    detection.confidence = min(detection.confidence, confidence)
                    
                    # Update known face statistics
                    if matched_name in self.known_faces:
                        known_face = self.known_faces[matched_name]
                        known_face.last_seen = time.time()
                        known_face.recognition_count += 1
                    
                    # Log recognition event
                    self.recognition_history.append({
                        'timestamp': time.time(),
                        'name': matched_name,
                        'confidence': confidence,
                        'distance': face_distances[best_match_index]
                    })
                    
                    logger.debug(f"Face matched: {matched_name} (confidence: {confidence:.3f})")
                
                else:
                    # No match found
                    detection.face_id = None
            
            matching_time = time.time() - start_time
            self.matching_times.append(matching_time)
            
            # Clean up old timing data
            if len(self.matching_times) > 100:
                self.matching_times = self.matching_times[-100:]
            
            # Clean up old recognition history
            if len(self.recognition_history) > 1000:
                self.recognition_history = self.recognition_history[-1000:]
            
            return detections
            
        except Exception as e:
            logger.error(f"Face matching error: {e}")
            return detections
    
    def _rebuild_encoding_lists(self):
        """Rebuild encoding and name lists for faster matching."""
        self.face_encodings = []
        self.face_names = []
        
        for name, known_face in self.known_faces.items():
            self.face_encodings.append(known_face.encoding)
            self.face_names.append(name)
    
    def save_face_database(self) -> bool:
        """Save the face database to file."""
        encodings_file = os.path.join(
            config.face.face_gallery_path,
            config.face.encodings_file
        )
        
        try:
            data = {
                'known_faces': self.known_faces,
                'version': '1.0',
                'saved_timestamp': time.time()
            }
            
            with open(encodings_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved face database with {len(self.known_faces)} faces")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save face database: {e}")
            return False
    
    def remove_face(self, person_name: str) -> bool:
        """Remove a face from the database."""
        if person_name in self.known_faces:
            # Remove image file if it exists
            known_face = self.known_faces[person_name]
            if os.path.exists(known_face.image_path):
                try:
                    os.remove(known_face.image_path)
                except Exception as e:
                    logger.warning(f"Could not remove image file: {e}")
            
            # Remove from database
            del self.known_faces[person_name]
            self._rebuild_encoding_lists()
            
            logger.info(f"Removed face: {person_name}")
            return True
        
        return False
    
    def update_face_tolerance(self, new_tolerance: float):
        """Update face matching tolerance."""
        self.tolerance = max(0.0, min(1.0, new_tolerance))
        logger.info(f"Updated face tolerance to {self.tolerance}")
    
    def get_face_database_info(self) -> Dict:
        """Get information about the face database."""
        if not self.known_faces:
            return {'total_faces': 0, 'faces': []}
        
        current_time = time.time()
        faces_info = []
        
        for name, known_face in self.known_faces.items():
            face_info = {
                'name': name,
                'image_path': known_face.image_path,
                'added_date': time.ctime(known_face.added_timestamp),
                'last_seen': time.ctime(known_face.last_seen) if known_face.last_seen > 0 else 'Never',
                'recognition_count': known_face.recognition_count,
                'days_since_added': int((current_time - known_face.added_timestamp) / 86400)
            }
            faces_info.append(face_info)
        
        # Sort by recognition count
        faces_info.sort(key=lambda x: x['recognition_count'], reverse=True)
        
        return {
            'total_faces': len(self.known_faces),
            'faces': faces_info,
            'database_file': os.path.join(config.face.face_gallery_path, config.face.encodings_file)
        }
    
    def get_recognition_statistics(self) -> Dict:
        """Get face recognition performance statistics."""
        avg_matching_time = np.mean(self.matching_times) if self.matching_times else 0
        
        # Recent recognition stats
        recent_recognitions = [
            r for r in self.recognition_history 
            if time.time() - r['timestamp'] < 3600  # Last hour
        ]
        
        recognition_counts = {}
        for recognition in recent_recognitions:
            name = recognition['name']
            recognition_counts[name] = recognition_counts.get(name, 0) + 1
        
        return {
            'total_known_faces': len(self.known_faces),
            'average_matching_time_ms': avg_matching_time * 1000,
            'matching_fps': 1.0 / avg_matching_time if avg_matching_time > 0 else 0,
            'tolerance': self.tolerance,
            'recent_recognitions_count': len(recent_recognitions),
            'most_recognized_faces': dict(sorted(recognition_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:5])
        }
    
    def find_similar_faces(self, detection: FaceDetection, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the most similar faces to a detection."""
        if detection.encoding is None or not self.face_encodings:
            return []
        
        try:
            # Calculate distances to all known faces
            face_distances = fr.face_distance(
                self.face_encodings, 
                detection.encoding
            )
            
            # Get top-k similar faces
            similar_indices = np.argsort(face_distances)[:top_k]
            
            similar_faces = []
            for idx in similar_indices:
                if idx < len(self.face_names):
                    name = self.face_names[idx]
                    distance = face_distances[idx]
                    similarity = 1.0 - distance
                    similar_faces.append((name, similarity))
            
            return similar_faces
            
        except Exception as e:
            logger.error(f"Error finding similar faces: {e}")
            return []
    
    def export_face_database(self, export_path: str) -> bool:
        """Export face database to a different location."""
        try:
            import shutil
            
            # Create export directory
            os.makedirs(export_path, exist_ok=True)
            
            # Export encodings file
            encodings_file = os.path.join(config.face.face_gallery_path, config.face.encodings_file)
            if os.path.exists(encodings_file):
                shutil.copy2(encodings_file, export_path)
            
            # Export face images
            for name, known_face in self.known_faces.items():
                if os.path.exists(known_face.image_path):
                    filename = os.path.basename(known_face.image_path)
                    shutil.copy2(known_face.image_path, os.path.join(export_path, filename))
            
            logger.info(f"Exported face database to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export face database: {e}")
            return False
    
    def import_face_database(self, import_path: str) -> bool:
        """Import face database from a different location."""
        try:
            encodings_file = os.path.join(import_path, config.face.encodings_file)
            
            if not os.path.exists(encodings_file):
                logger.error(f"No encodings file found at {import_path}")
                return False
            
            # Backup current database
            current_encodings = os.path.join(config.face.face_gallery_path, config.face.encodings_file)
            if os.path.exists(current_encodings):
                backup_name = f"{config.face.encodings_file}.backup.{int(time.time())}"
                backup_path = os.path.join(config.face.face_gallery_path, backup_name)
                shutil.copy2(current_encodings, backup_path)
            
            # Import new database
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
                imported_faces = data.get('known_faces', {})
            
            # Merge with existing faces
            for name, known_face in imported_faces.items():
                if name not in self.known_faces:
                    # Copy image file if it exists
                    if os.path.exists(os.path.join(import_path, os.path.basename(known_face.image_path))):
                        import shutil
                        dest_path = os.path.join(config.face.face_gallery_path, 
                                               os.path.basename(known_face.image_path))
                        shutil.copy2(os.path.join(import_path, os.path.basename(known_face.image_path)), 
                                   dest_path)
                        known_face.image_path = dest_path
                    
                    self.known_faces[name] = known_face
            
            # Rebuild encoding lists and save
            self._rebuild_encoding_lists()
            self.save_face_database()
            
            logger.info(f"Imported {len(imported_faces)} faces from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import face database: {e}")
            return False
    
    def cleanup_old_faces(self, days_threshold: int = 30, min_recognition_count: int = 1):
        """Remove old faces that haven't been seen recently."""
        current_time = time.time()
        threshold_seconds = days_threshold * 86400
        
        faces_to_remove = []
        
        for name, known_face in self.known_faces.items():
            days_since_added = (current_time - known_face.added_timestamp) / 86400
            days_since_seen = (current_time - known_face.last_seen) / 86400 if known_face.last_seen > 0 else float('inf')
            
            if (days_since_added > days_threshold and 
                known_face.recognition_count < min_recognition_count and
                days_since_seen > days_threshold):
                faces_to_remove.append(name)
        
        # Remove old faces
        removed_count = 0
        for name in faces_to_remove:
            if self.remove_face(name):
                removed_count += 1
        
        if removed_count > 0:
            self.save_face_database()
            logger.info(f"Cleaned up {removed_count} old faces")
        
        return removed_count