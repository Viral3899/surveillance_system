#!/usr/bin/env python3
"""
Complete Employee Attendance Module with Segmentation Fault Prevention
====================================================================

This version includes all missing methods and comprehensive error handling
to prevent segmentation faults during face processing.

Key improvements:
- All missing methods implemented
- Sequential image processing instead of parallel
- Better memory management and cleanup
- Image validation before processing
- Graceful error recovery
- Reduced memory footprint
"""

import os
import cv2
import numpy as np
import pandas as pd
import face_recognition
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pickle
import threading
import time
import shutil
import hashlib
import gc  # Garbage collection for memory management

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmployeeAttendanceModule:
    """
    Memory-safe Employee Attendance Module that prevents segmentation faults.
    
    This version prioritizes stability over performance by:
    - Processing images sequentially instead of in parallel
    - Adding extensive validation and error handling
    - Managing memory more carefully
    - Providing fallback mechanisms
    """
    
    def __init__(self, 
                 face_dir: str = "faces",
                 attendance_file: str = "attendance.xlsx",
                 cooldown_seconds: int = 600,
                 tolerance: float = 0.5,
                 encodings_cache: str = "face_encodings.pkl",
                 backup_enabled: bool = True,
                 max_image_size: int = 1024):
        """
        Initialize the Safe Employee Attendance Module.
        
        Args:
            face_dir (str): Directory containing employee face images
            attendance_file (str): Excel file path for attendance logging
            cooldown_seconds (int): Cooldown period to prevent duplicate entries
            tolerance (float): Face recognition tolerance (lower = stricter)
            encodings_cache (str): Cache file for face encodings
            backup_enabled (bool): Enable automatic backups
            max_image_size (int): Maximum image dimension to prevent memory issues
        """
        # Input validation
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError("tolerance must be between 0.0 and 1.0")
        
        self.face_dir = Path(face_dir)
        self.attendance_file = Path(attendance_file)
        self.cooldown_seconds = cooldown_seconds
        self.tolerance = tolerance
        self.encodings_cache = Path(encodings_cache)
        self.backup_enabled = backup_enabled
        self.max_image_size = max_image_size
        
        # Face recognition data with thread safety
        self.known_face_encodings: List[np.ndarray] = []
        self.known_employee_ids: List[str] = []
        self.employee_metadata: Dict[str, Dict] = {}
        self._face_data_lock = threading.RLock()
        
        # Attendance tracking with thread safety
        self.last_seen_time: Dict[str, datetime] = {}
        self.visit_counts: Dict[str, int] = {}
        self.daily_stats: Dict[str, Dict] = {}
        self._attendance_lock = threading.RLock()
        
        # Performance tracking
        self.detection_count = 0
        self.attendance_logs = 0
        self.processing_times = []
        self._stats_lock = threading.Lock()
        
        # Error tracking
        self.error_count = 0
        self.last_error_time = None
        self.consecutive_errors = 0
        
        # Initialize the module safely
        self._initialize_module_safely()
        
        logger.info(f"Safe Employee Attendance Module initialized with {len(self.known_employee_ids)} employees")
    
    def _initialize_module_safely(self):
        """Initialize the module with comprehensive safety checks."""
        try:
            # Create directories
            self._create_directories()
            
            # Initialize attendance file
            self._initialize_attendance_file()
            
            # Load known faces with safety measures
            if not self.load_known_faces_safely():
                logger.warning("No faces loaded, system will run in detection-only mode")
            
            # Force garbage collection after initialization
            gc.collect()
            
            # Create initial backup if enabled
            if self.backup_enabled:
                self._create_backup()
            
            logger.info("Safe module initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Safe module initialization failed: {e}")
            self._handle_error(e)
            raise
    
    def _create_directories(self):
        """Create necessary directories with proper error handling."""
        directories = [
            self.face_dir,
            self.attendance_file.parent,
            self.encodings_cache.parent,
            Path("backup"),
            Path("backup/daily"),
            Path("logs")
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created/verified directory: {directory}")
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")
        
        # Create sample face directory structure if empty
        if not any(self.face_dir.iterdir()) if self.face_dir.exists() else True:
            logger.warning(f"Face directory '{self.face_dir}' is empty.")
            logger.info("Expected format: faces/EMP001.jpg, faces/EMP002.png, etc.")
    
    def _initialize_attendance_file(self):
        """Initialize the Excel attendance file with proper headers."""
        try:
            if not self.attendance_file.exists():
                # Create new attendance file with headers
                df = pd.DataFrame(columns=[
                    'Employee_ID', 'Employee_Name', 'Date', 'Time', 
                    'Timestamp', 'Visit_Type', 'Visit_Count', 'Confidence'
                ])
                
                df.to_excel(self.attendance_file, index=False, engine='openpyxl')
                logger.info(f"Created new attendance file: {self.attendance_file}")
            else:
                # Load existing visit counts
                try:
                    df = pd.read_excel(self.attendance_file, engine='openpyxl')
                    if not df.empty and 'Employee_ID' in df.columns:
                        with self._attendance_lock:
                            visit_counts = df.groupby('Employee_ID').size().to_dict()
                            self.visit_counts.update(visit_counts)
                        
                        logger.info(f"Loaded existing attendance data for {len(visit_counts)} employees")
                    else:
                        logger.warning("Existing attendance file has invalid format")
                        
                except Exception as e:
                    logger.error(f"Error reading existing attendance file: {e}")
                    self._create_backup_and_new_file()
                    
        except Exception as e:
            logger.error(f"Error initializing attendance file: {e}")
            self._handle_error(e)
    
    def _create_backup_and_new_file(self):
        """Create backup of corrupted file and start fresh."""
        try:
            backup_path = self.attendance_file.with_suffix('.corrupted.xlsx')
            shutil.copy2(self.attendance_file, backup_path)
            logger.info(f"Backed up corrupted file to: {backup_path}")
        except Exception as backup_error:
            logger.error(f"Could not backup corrupted file: {backup_error}")
        
        # Create new file
        df = pd.DataFrame(columns=[
            'Employee_ID', 'Employee_Name', 'Date', 'Time', 
            'Timestamp', 'Visit_Type', 'Visit_Count', 'Confidence'
        ])
        df.to_excel(self.attendance_file, index=False, engine='openpyxl')
        logger.info(f"Created new attendance file: {self.attendance_file}")
    
    def load_known_faces_safely(self) -> bool:
        """
        Load and encode all faces with safety measures to prevent segfaults.
        
        Returns:
            bool: True if faces were loaded successfully, False otherwise
        """
        try:
            # Try to load from cache first
            if self._load_encodings_cache():
                logger.info("Loaded face encodings from cache")
                return True
            
            # Load from face images with safety measures
            success = self._load_faces_from_images_safely()
            
            if success:
                # Save to cache for faster future loading
                self._save_encodings_cache()
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading known faces safely: {e}")
            self._handle_error(e)
            return False
    
    def _load_encodings_cache(self) -> bool:
        """Load face encodings from cache file with validation."""
        if not self.encodings_cache.exists():
            return False
        
        try:
            with open(self.encodings_cache, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache data structure
            required_keys = ['encodings', 'employee_ids', 'metadata']
            if not all(key in cache_data for key in required_keys):
                logger.warning("Invalid cache format, rebuilding from images")
                return False
            
            # Validate data consistency
            encodings = cache_data['encodings']
            employee_ids = cache_data['employee_ids']
            metadata = cache_data.get('metadata', {})
            
            if len(encodings) != len(employee_ids):
                logger.warning("Cache data inconsistency, rebuilding")
                return False
            
            # Load data with thread safety
            with self._face_data_lock:
                self.known_face_encodings = encodings
                self.known_employee_ids = employee_ids
                self.employee_metadata = metadata
            
            logger.info(f"Loaded {len(employee_ids)} face encodings from cache")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load encodings cache: {e}")
            try:
                self.encodings_cache.unlink()
                logger.info("Removed corrupted cache file")
            except Exception:
                pass
            return False
    
    def _save_encodings_cache(self) -> bool:
        """Save face encodings to cache file."""
        try:
            with self._face_data_lock:
                cache_data = {
                    'encodings': self.known_face_encodings.copy(),
                    'employee_ids': self.known_employee_ids.copy(),
                    'metadata': self.employee_metadata.copy(),
                    'save_timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            
            # Save to temporary file first
            temp_cache = self.encodings_cache.with_suffix('.tmp')
            with open(temp_cache, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_cache.replace(self.encodings_cache)
            
            logger.info(f"Saved {len(self.known_employee_ids)} face encodings to cache")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save encodings cache: {e}")
            # Clean up temporary file
            temp_cache = self.encodings_cache.with_suffix('.tmp')
            if temp_cache.exists():
                try:
                    temp_cache.unlink()
                except Exception:
                    pass
            return False
    
    def _create_backup(self) -> bool:
        """Create backup of attendance file."""
        if not self.attendance_file.exists():
            return False
        
        try:
            # Create backup directory structure
            backup_dir = Path("backup")
            daily_backup_dir = backup_dir / "daily"
            backup_dir.mkdir(exist_ok=True)
            daily_backup_dir.mkdir(exist_ok=True)
            
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"attendance_backup_{timestamp}.xlsx"
            backup_path = backup_dir / backup_filename
            
            # Create daily backup
            daily_backup_filename = f"attendance_{datetime.now().strftime('%Y%m%d')}.xlsx"
            daily_backup_path = daily_backup_dir / daily_backup_filename
            
            # Copy files
            shutil.copy2(self.attendance_file, backup_path)
            shutil.copy2(self.attendance_file, daily_backup_path)
            
            logger.debug(f"Created backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
            return False
    
    def _load_faces_from_images_safely(self) -> bool:
        """Load and encode faces from image files with extensive safety measures."""
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        if not self.face_dir.exists():
            logger.warning(f"Face directory does not exist: {self.face_dir}")
            return False
        
        image_files = [
            f for f in self.face_dir.iterdir() 
            if f.suffix.lower() in supported_formats
        ]
        
        if not image_files:
            logger.warning(f"No image files found in {self.face_dir}")
            return False
        
        logger.info(f"Loading {len(image_files)} face images from: {self.face_dir} (sequential processing for safety)")
        
        # Process images sequentially (not in parallel) to prevent memory issues
        loaded_count = 0
        new_encodings = []
        new_employee_ids = []
        new_metadata = {}
        
        for i, image_path in enumerate(image_files):
            try:
                logger.info(f"Processing image {i+1}/{len(image_files)}: {image_path.name}")
                
                result = self._process_face_image_safely(image_path)
                if result:
                    employee_id, encoding, metadata = result
                    new_encodings.append(encoding)
                    new_employee_ids.append(employee_id)
                    new_metadata[employee_id] = metadata
                    loaded_count += 1
                    logger.info(f"Successfully loaded face for employee: {employee_id}")
                else:
                    logger.warning(f"Failed to process: {image_path.name}")
                
                # Force garbage collection after each image to prevent memory buildup
                gc.collect()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Critical error processing {image_path}: {e}")
                # Continue with other images even if one fails
                continue
        
        if loaded_count > 0:
            # Update data with thread safety
            with self._face_data_lock:
                self.known_face_encodings = new_encodings
                self.known_employee_ids = new_employee_ids
                self.employee_metadata = new_metadata
            
            logger.info(f"Successfully loaded {loaded_count} employee faces")
            return True
        else:
            logger.error("No faces were loaded. Please check your face gallery.")
            return False
    
    def _process_face_image_safely(self, image_path: Path) -> Optional[Tuple[str, np.ndarray, Dict]]:
        """Process a single face image with extensive safety measures."""
        try:
            employee_id = image_path.stem
            
            # Validate employee ID
            if not employee_id or len(employee_id) < 2:
                logger.warning(f"Invalid employee ID from filename: {image_path}")
                return None
            
            # Validate file existence and basic properties
            if not image_path.exists():
                logger.warning(f"Image file does not exist: {image_path}")
                return None
            
            file_size = image_path.stat().st_size
            if file_size == 0:
                logger.warning(f"Empty image file: {image_path}")
                return None
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                logger.warning(f"Image file too large: {image_path}")
                return None
            
            # Load and validate image using OpenCV first (more robust)
            try:
                cv_image = cv2.imread(str(image_path))
                if cv_image is None:
                    logger.warning(f"Could not load image with OpenCV: {image_path}")
                    return None
                
                # Check image dimensions and resize if necessary
                height, width = cv_image.shape[:2]
                if max(height, width) > self.max_image_size:
                    # Resize to prevent memory issues
                    scale = self.max_image_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    cv_image = cv2.resize(cv_image, (new_width, new_height))
                    logger.info(f"Resized image {image_path.name} to {new_width}x{new_height}")
                
                # Convert to RGB for face_recognition
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                
            except Exception as e:
                logger.warning(f"OpenCV processing failed for {image_path}: {e}")
                return None
            
            # Now try face_recognition processing with safety measures
            try:
                # Find face encodings with error handling
                encodings = face_recognition.face_encodings(rgb_image)
                
                if len(encodings) == 0:
                    logger.warning(f"No face found in {image_path}")
                    return None
                elif len(encodings) > 1:
                    logger.warning(f"Multiple faces found in {image_path}, using the first one")
                
                # Use the first face encoding
                encoding = encodings[0]
                
                # Validate encoding
                if encoding is None or len(encoding) == 0:
                    logger.warning(f"Invalid face encoding: {image_path}")
                    return None
                
                # Create metadata
                metadata = {
                    'image_path': str(image_path),
                    'file_size': file_size,
                    'image_shape': rgb_image.shape,
                    'encoding_length': len(encoding),
                    'added_timestamp': datetime.now().isoformat(),
                    'file_hash': self._calculate_file_hash(image_path),
                    'processed_safely': True,
                    'name': employee_id  # Add name field
                }
                
                logger.debug(f"Successfully processed face: {employee_id}")
                return employee_id, encoding, metadata
                
            except Exception as e:
                logger.warning(f"Face recognition processing failed for {image_path}: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Critical error processing face image {image_path}: {e}")
            return None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for integrity checking."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.debug(f"Error calculating file hash: {e}")
            return ""
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a video frame for face detection and attendance logging with safety measures.
        
        Args:
            frame (np.ndarray): Input video frame
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: Annotated frame and detection results
        """
        start_time = time.time()
        
        # Input validation
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided to process_frame")
            return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8), []
        
        # Check if we have known faces
        with self._face_data_lock:
            if len(self.known_face_encodings) == 0:
                logger.debug("No known faces loaded, returning original frame")
                return frame, []
        
        try:
            self.detection_count += 1
            current_time = datetime.now()
            
            # Validate frame dimensions
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                logger.warning(f"Invalid frame shape: {frame.shape}")
                return frame, []
            
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                logger.warning(f"Invalid frame dimensions: {width}x{height}")
                return frame, []
            
            # Resize frame for faster processing and memory safety
            processing_scale = 0.25  # More aggressive scaling for safety
            small_frame = cv2.resize(frame, (0, 0), fx=processing_scale, fy=processing_scale)
            
            # Convert BGR to RGB with error handling
            try:
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                logger.warning(f"Color conversion error: {e}")
                return frame, []
            
            # Detect faces with safety measures
            try:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                
                # Limit number of faces processed to prevent memory issues
                max_faces = 5
                if len(face_locations) > max_faces:
                    logger.info(f"Too many faces detected ({len(face_locations)}), processing only first {max_faces}")
                    face_locations = face_locations[:max_faces]
                
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
            except Exception as e:
                logger.warning(f"Face detection error: {e}")
                return frame, []
            
            detection_results = []
            annotated_frame = frame.copy()
            
            # Process each detected face with safety measures
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                try:
                    # Scale back up face locations
                    scale_factor = 1.0 / processing_scale
                    top = int(top * scale_factor)
                    right = int(right * scale_factor)
                    bottom = int(bottom * scale_factor)
                    left = int(left * scale_factor)
                    
                    # Ensure coordinates are within frame bounds
                    top = max(0, min(height, top))
                    bottom = max(0, min(height, bottom))
                    left = max(0, min(width, left))
                    right = max(0, min(width, right))
                    
                    # Validate face dimensions
                    if bottom <= top or right <= left:
                        logger.debug("Invalid face dimensions after scaling")
                        continue
                    
                    # Match against known faces
                    employee_id, employee_name, confidence = self._match_face_safely(face_encoding)
                    
                    # Handle attendance logging
                    visit_type = None
                    visit_count = 0
                    
                    if employee_id != "Unknown":
                        if self._should_log_attendance(employee_id, current_time):
                            visit_type, visit_count = self._log_attendance(
                                employee_id, employee_name, current_time, confidence
                            )
                        else:
                            visit_type, visit_count = self._get_current_visit_info(employee_id)
                    
                    # Prepare detection result
                    detection_result = {
                        'employee_id': employee_id,
                        'employee_name': employee_name,
                        'confidence': confidence,
                        'bbox': (left, top, right, bottom),
                        'visit_type': visit_type,
                        'visit_count': visit_count,
                        'time': current_time.isoformat(),
                        'processing_time': time.time() - start_time
                    }
                    detection_results.append(detection_result)
                    
                    # Draw detection on frame
                    annotated_frame = self._draw_detection(annotated_frame, detection_result)
                    
                except Exception as e:
                    logger.warning(f"Error processing face detection: {e}")
                    continue
            
            # Update performance metrics
            processing_time = time.time() - start_time
            with self._stats_lock:
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times = self.processing_times[-100:]
            
            # Reset consecutive error count on successful processing
            self.consecutive_errors = 0
            
            # Force garbage collection periodically
            if self.detection_count % 50 == 0:
                gc.collect()
            
            return annotated_frame, detection_results
            
        except Exception as e:
            logger.error(f"Critical error in process_frame: {e}")
            self._handle_error(e)
            return frame, []
    
    def _match_face_safely(self, face_encoding: np.ndarray) -> Tuple[str, str, float]:
        """Match a face encoding against known faces with safety measures."""
        try:
            with self._face_data_lock:
                if len(self.known_face_encodings) == 0:
                    return "Unknown", "Unknown", 0.0
                
                # Compare against known faces with error handling
                try:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=self.tolerance
                    )
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                except Exception as e:
                    logger.warning(f"Face comparison error: {e}")
                    return "Unknown", "Unknown", 0.0
            
            employee_id = "Unknown"
            employee_name = "Unknown"
            confidence = 0.0
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    employee_id = self.known_employee_ids[best_match_index]
                    employee_name = self.employee_metadata.get(employee_id, {}).get('name', employee_id)
                    confidence = max(0.0, min(1.0, 1.0 - face_distances[best_match_index]))
            
            return employee_id, employee_name, confidence
            
        except Exception as e:
            logger.warning(f"Error matching face safely: {e}")
            return "Unknown", "Unknown", 0.0
    
    def _should_log_attendance(self, employee_id: str, current_time: datetime) -> bool:
        """Check if attendance should be logged based on cooldown logic."""
        try:
            with self._attendance_lock:
                if employee_id not in self.last_seen_time:
                    return True
                
                time_since_last_seen = current_time - self.last_seen_time[employee_id]
                return time_since_last_seen.total_seconds() >= self.cooldown_seconds
                
        except Exception as e:
            logger.warning(f"Error checking attendance cooldown: {e}")
            return False
    
    def _log_attendance(self, employee_id: str, employee_name: str, 
                       timestamp: datetime, confidence: float) -> Tuple[str, int]:
        """Save attendance record with safety measures."""
        try:
            visit_type = self._determine_visit_type(employee_id)
            
            with self._attendance_lock:
                self.last_seen_time[employee_id] = timestamp
                self.visit_counts[employee_id] = self.visit_counts.get(employee_id, 0) + 1
                visit_count = self.visit_counts[employee_id]
            
            new_record = {
                'Employee_ID': employee_id,
                'Employee_Name': employee_name,
                'Date': timestamp.strftime('%Y-%m-%d'),
                'Time': timestamp.strftime('%H:%M:%S'),
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Visit_Type': visit_type,
                'Visit_Count': visit_count,
                'Confidence': round(confidence, 3)
            }
            
            success = self._save_attendance_record_safely(new_record)
            
            if success:
                self.attendance_logs += 1
                logger.info(f"Logged attendance: {employee_name} ({employee_id}) - {visit_type} (Visit #{visit_count})")
            
            return visit_type, visit_count
            
        except Exception as e:
            logger.error(f"Error logging attendance for {employee_id}: {e}")
            return "ERROR", 0
    
    def _determine_visit_type(self, employee_id: str) -> str:
        """Determine the type of visit based on time and previous visits."""
        try:
            current_time = datetime.now()
            
            with self._attendance_lock:
                if employee_id not in self.last_seen_time:
                    return "FIRST_VISIT"
                
                last_seen = self.last_seen_time[employee_id]
                time_diff = current_time - last_seen
                
                # Determine visit type based on time difference
                if time_diff.total_seconds() < 3600:  # Less than 1 hour
                    return "QUICK_RETURN"
                elif time_diff.days >= 1:  # More than 1 day
                    return "DAILY_CHECKIN"
                else:
                    return "RETURN_VISIT"
                    
        except Exception as e:
            logger.warning(f"Error determining visit type: {e}")
            return "UNKNOWN"
    
    def _get_current_visit_info(self, employee_id: str) -> Tuple[str, int]:
        """Get current visit information for an employee."""
        try:
            with self._attendance_lock:
                visit_count = self.visit_counts.get(employee_id, 0)
                
                if employee_id in self.last_seen_time:
                    last_seen = self.last_seen_time[employee_id]
                    time_diff = datetime.now() - last_seen
                    
                    if time_diff.total_seconds() < self.cooldown_seconds:
                        return "COOLDOWN", visit_count
                
                return "READY", visit_count
                
        except Exception as e:
            logger.warning(f"Error getting visit info: {e}")
            return "ERROR", 0
    
    def _draw_detection(self, frame: np.ndarray, detection_result: Dict) -> np.ndarray:
        """Draw detection information on the frame."""
        try:
            left, top, right, bottom = detection_result['bbox']
            employee_id = detection_result['employee_id']
            employee_name = detection_result['employee_name']
            confidence = detection_result['confidence']
            visit_type = detection_result.get('visit_type', '')
            visit_count = detection_result.get('visit_count', 0)
            
            # Choose colors based on recognition status
            if employee_id == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                text_color = (255, 255, 255)  # White text
            else:
                color = (0, 255, 0)  # Green for known
                text_color = (0, 0, 0)  # Black text
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Prepare text information
            if employee_id != "Unknown":
                main_text = f"{employee_name} ({employee_id})"
                confidence_text = f"Confidence: {confidence:.2f}"
                
                if visit_type and visit_count > 0:
                    visit_text = f"{visit_type} - Visit #{visit_count}"
                else:
                    visit_text = "Ready to log"
            else:
                main_text = "Unknown Person"
                confidence_text = f"Best match: {confidence:.2f}"
                visit_text = "Not recognized"
            
            # Calculate text positioning
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            
            # Get text sizes
            (main_w, main_h), _ = cv2.getTextSize(main_text, font, font_scale, thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(confidence_text, font, font_scale * 0.8, thickness)
            (visit_w, visit_h), _ = cv2.getTextSize(visit_text, font, font_scale * 0.8, thickness)
            
            # Calculate background rectangle
            text_width = max(main_w, conf_w, visit_w)
            text_height = main_h + conf_h + visit_h + 20
            
            # Draw background rectangle for text
            bg_top = max(0, top - text_height - 10)
            bg_bottom = max(text_height + 10, top)
            bg_left = left
            bg_right = min(frame.shape[1], left + text_width + 10)
            
            # Create semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_left, bg_top), (bg_right, bg_bottom), color, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw text lines
            y_offset = bg_top + main_h + 5
            cv2.putText(frame, main_text, (bg_left + 5, y_offset), font, font_scale, text_color, thickness)
            
            y_offset += conf_h + 5
            cv2.putText(frame, confidence_text, (bg_left + 5, y_offset), font, font_scale * 0.8, text_color, thickness)
            
            y_offset += visit_h + 5
            cv2.putText(frame, visit_text, (bg_left + 5, y_offset), font, font_scale * 0.8, text_color, thickness)
            
            return frame
            
        except Exception as e:
            logger.warning(f"Error drawing detection: {e}")
            return frame
    
    def _save_attendance_record_safely(self, record: Dict) -> bool:
        """Save attendance record with atomic operations and extensive error handling."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if self.backup_enabled and attempt == 0:
                    self._create_backup()
                
                # Load existing data with error handling
                try:
                    if self.attendance_file.exists():
                        df = pd.read_excel(self.attendance_file, engine='openpyxl')
                    else:
                        df = pd.DataFrame(columns=list(record.keys()))
                except Exception as e:
                    logger.warning(f"Error reading existing file: {e}")
                    df = pd.DataFrame(columns=list(record.keys()))
                
                # Add new record
                new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                
                # Save to temporary file first
                temp_file = self.attendance_file.with_suffix('.tmp')
                new_df.to_excel(temp_file, index=False, engine='openpyxl')
                
                # Atomic rename
                temp_file.replace(self.attendance_file)
                
                logger.debug(f"Successfully saved attendance record (attempt {attempt + 1})")
                return True
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to save attendance: {e}")
                
                # Clean up temporary file
                temp_file = self.attendance_file.with_suffix('.tmp')
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass
                
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed to save attendance")
                    return False
                
                time.sleep(0.1 * (attempt + 1))
        
        return False
    
    def _handle_error(self, error: Exception):
        """Handle errors with tracking and recovery."""
        self.error_count += 1
        self.last_error_time = datetime.now()
        self.consecutive_errors += 1
        
        logger.error(f"Error #{self.error_count}: {error}")
        
        # Force garbage collection on errors
        gc.collect()
        
        if self.consecutive_errors >= 3:
            logger.warning("Multiple consecutive errors, attempting recovery")
            self._attempt_recovery()
    
    def _attempt_recovery(self):
        """Attempt to recover from errors."""
        try:
            logger.info("Attempting safe system recovery")
            
            # Clear face data and reload safely
            with self._face_data_lock:
                self.known_face_encodings.clear()
                self.known_employee_ids.clear()
                self.employee_metadata.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Reload faces with safety measures
            if self.load_known_faces_safely():
                logger.info("Face data reloaded successfully")
                self.consecutive_errors = 0
            else:
                logger.warning("Face data reload failed")
                
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
    
    def get_statistics(self) -> Dict:
        """Get module statistics."""
        try:
            with self._face_data_lock:
                total_employees = len(self.known_employee_ids)
            
            with self._stats_lock:
                avg_processing_time = (
                    np.mean(self.processing_times) if self.processing_times else 0.0
                )
            
            with self._attendance_lock:
                total_visits = sum(self.visit_counts.values())
                active_employees = len(self.visit_counts)
            
            return {
                'total_employees': total_employees,
                'active_employees': active_employees,
                'total_visits': total_visits,
                'total_detections': self.detection_count,
                'total_attendance_logs': self.attendance_logs,
                'average_processing_time_ms': avg_processing_time * 1000,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
                'safe_mode': True,
                'max_image_size': self.max_image_size,
                'tolerance': self.tolerance,
                'cooldown_seconds': self.cooldown_seconds
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e), 'safe_mode': True}
    
    def get_employee_list(self) -> List[Dict]:
        """Get list of all known employees with metadata."""
        try:
            with self._face_data_lock:
                employee_list = []
                for emp_id in self.known_employee_ids:
                    metadata = self.employee_metadata.get(emp_id, {})
                    
                    with self._attendance_lock:
                        visit_count = self.visit_counts.get(emp_id, 0)
                        last_seen = self.last_seen_time.get(emp_id)
                    
                    employee_info = {
                        'employee_id': emp_id,
                        'employee_name': metadata.get('name', emp_id),
                        'visit_count': visit_count,
                        'last_seen': last_seen.isoformat() if last_seen else None,
                        'image_path': metadata.get('image_path', ''),
                        'added_timestamp': metadata.get('added_timestamp', ''),
                        'file_size': metadata.get('file_size', 0),
                        'encoding_length': metadata.get('encoding_length', 0)
                    }
                    employee_list.append(employee_info)
                
                return sorted(employee_list, key=lambda x: x['employee_id'])
                
        except Exception as e:
            logger.error(f"Error getting employee list: {e}")
            return []
    
    def get_attendance_summary(self, date: Optional[str] = None) -> Dict:
        """Get attendance summary for a specific date or today."""
        try:
            target_date = date if date else datetime.now().strftime('%Y-%m-%d')
            
            if not self.attendance_file.exists():
                return {'date': target_date, 'total_visits': 0, 'unique_employees': 0, 'visits': []}
            
            df = pd.read_excel(self.attendance_file, engine='openpyxl')
            
            if df.empty:
                return {'date': target_date, 'total_visits': 0, 'unique_employees': 0, 'visits': []}
            
            # Filter by date
            date_mask = df['Date'] == target_date
            day_data = df[date_mask]
            
            if day_data.empty:
                return {'date': target_date, 'total_visits': 0, 'unique_employees': 0, 'visits': []}
            
            # Calculate summary statistics
            total_visits = len(day_data)
            unique_employees = day_data['Employee_ID'].nunique()
            
            # Get visit details
            visits = day_data.to_dict('records')
            
            return {
                'date': target_date,
                'total_visits': total_visits,
                'unique_employees': unique_employees,
                'visits': visits,
                'first_visit': day_data['Time'].min() if not day_data.empty else None,
                'last_visit': day_data['Time'].max() if not day_data.empty else None
            }
            
        except Exception as e:
            logger.error(f"Error getting attendance summary: {e}")
            return {'error': str(e), 'date': target_date}
    
    def add_employee_from_image(self, image_path: str, employee_id: str = None) -> bool:
        """Add a new employee from an image file."""
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                logger.error(f"Image file does not exist: {image_path}")
                return False
            
            # Use filename as employee_id if not provided
            if not employee_id:
                employee_id = image_path.stem
            
            # Validate employee_id
            if not employee_id or len(employee_id) < 2:
                logger.error(f"Invalid employee_id: {employee_id}")
                return False
            
            # Check if employee already exists
            with self._face_data_lock:
                if employee_id in self.known_employee_ids:
                    logger.warning(f"Employee {employee_id} already exists")
                    return False
            
            # Process the image
            result = self._process_face_image_safely(image_path)
            if not result:
                logger.error(f"Failed to process face image: {image_path}")
                return False
            
            processed_id, encoding, metadata = result
            
            # Add to known faces
            with self._face_data_lock:
                self.known_face_encodings.append(encoding)
                self.known_employee_ids.append(employee_id)
                self.employee_metadata[employee_id] = metadata
            
            # Update cache
            self._save_encodings_cache()
            
            logger.info(f"Successfully added employee: {employee_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding employee from image: {e}")
            return False
    
    def remove_employee(self, employee_id: str) -> bool:
        """Remove an employee from the system."""
        try:
            with self._face_data_lock:
                if employee_id not in self.known_employee_ids:
                    logger.warning(f"Employee {employee_id} not found")
                    return False
                
                # Find index
                index = self.known_employee_ids.index(employee_id)
                
                # Remove from all lists/dicts
                self.known_face_encodings.pop(index)
                self.known_employee_ids.pop(index)
                self.employee_metadata.pop(employee_id, None)
            
            # Clean up attendance data
            with self._attendance_lock:
                self.last_seen_time.pop(employee_id, None)
                self.visit_counts.pop(employee_id, None)
            
            # Update cache
            self._save_encodings_cache()
            
            logger.info(f"Successfully removed employee: {employee_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing employee: {e}")
            return False
    
    def clear_attendance_data(self, employee_id: str = None) -> bool:
        """Clear attendance data for a specific employee or all employees."""
        try:
            if employee_id:
                # Clear data for specific employee
                with self._attendance_lock:
                    self.last_seen_time.pop(employee_id, None)
                    self.visit_counts.pop(employee_id, None)
                logger.info(f"Cleared attendance data for employee: {employee_id}")
            else:
                # Clear all attendance data
                with self._attendance_lock:
                    self.last_seen_time.clear()
                    self.visit_counts.clear()
                logger.info("Cleared all attendance data")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing attendance data: {e}")
            return False
    
    def export_attendance_data(self, output_path: str, start_date: str = None, end_date: str = None) -> bool:
        """Export attendance data to a new Excel file."""
        try:
            if not self.attendance_file.exists():
                logger.error("No attendance data to export")
                return False
            
            df = pd.read_excel(self.attendance_file, engine='openpyxl')
            
            if df.empty:
                logger.error("No attendance data to export")
                return False
            
            # Filter by date range if provided
            if start_date or end_date:
                df['Date'] = pd.to_datetime(df['Date'])
                
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    df = df[df['Date'] >= start_dt]
                
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    df = df[df['Date'] <= end_dt]
            
            # Export to new file
            output_path = Path(output_path)
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            logger.info(f"Exported {len(df)} attendance records to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting attendance data: {e}")
            return False
    
    def validate_system_integrity(self) -> Dict:
        """Validate the integrity of the attendance system and return status."""
        validation_results = {
            'status': 'healthy',
            'errors': [],
            'warnings': [],
            'checks_passed': 0,
            'total_checks': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check 1: Face directory exists and has images
            validation_results['total_checks'] += 1
            if not self.face_dir.exists():
                validation_results['errors'].append(f"Face directory does not exist: {self.face_dir}")
            else:
                supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
                image_files = [f for f in self.face_dir.iterdir() if f.suffix.lower() in supported_formats]
                if not image_files:
                    validation_results['warnings'].append(f"No face images found in {self.face_dir}")
                else:
                    validation_results['checks_passed'] += 1
            
            # Check 2: Known faces loaded
            validation_results['total_checks'] += 1
            with self._face_data_lock:
                if len(self.known_face_encodings) == 0:
                    validation_results['warnings'].append("No face encodings loaded")
                else:
                    validation_results['checks_passed'] += 1
                    
                # Check consistency between encodings and employee IDs
                if len(self.known_face_encodings) != len(self.known_employee_ids):
                    validation_results['errors'].append("Inconsistency between face encodings and employee IDs")
            
            # Check 3: Attendance file integrity
            validation_results['total_checks'] += 1
            if not self.attendance_file.exists():
                validation_results['warnings'].append(f"Attendance file does not exist: {self.attendance_file}")
            else:
                try:
                    df = pd.read_excel(self.attendance_file, engine='openpyxl')
                    required_columns = ['Employee_ID', 'Employee_Name', 'Date', 'Time', 'Timestamp']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        validation_results['errors'].append(f"Missing columns in attendance file: {missing_columns}")
                    else:
                        validation_results['checks_passed'] += 1
                except Exception as e:
                    validation_results['errors'].append(f"Cannot read attendance file: {e}")
            
            # Check 4: Cache file integrity
            validation_results['total_checks'] += 1
            if self.encodings_cache.exists():
                try:
                    with open(self.encodings_cache, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    required_keys = ['encodings', 'employee_ids', 'metadata']
                    if all(key in cache_data for key in required_keys):
                        validation_results['checks_passed'] += 1
                    else:
                        validation_results['warnings'].append("Invalid cache file format")
                except Exception as e:
                    validation_results['warnings'].append(f"Cannot read cache file: {e}")
            else:
                validation_results['checks_passed'] += 1  # No cache is fine
            
            # Check 5: Directory permissions
            validation_results['total_checks'] += 1
            try:
                # Test write permissions in key directories
                test_dirs = [self.face_dir.parent, self.attendance_file.parent, Path("backup")]
                for test_dir in test_dirs:
                    if test_dir.exists():
                        test_file = test_dir / f"test_write_{int(time.time())}.tmp"
                        try:
                            test_file.touch()
                            test_file.unlink()
                        except Exception:
                            validation_results['warnings'].append(f"No write permission in {test_dir}")
                            break
                else:
                    validation_results['checks_passed'] += 1
            except Exception as e:
                validation_results['warnings'].append(f"Cannot check directory permissions: {e}")
            
            # Check 6: Memory and performance
            validation_results['total_checks'] += 1
            try:
                with self._stats_lock:
                    if len(self.processing_times) > 0:
                        avg_time = np.mean(self.processing_times)
                        if avg_time > 5.0:  # More than 5 seconds average
                            validation_results['warnings'].append(f"High average processing time: {avg_time:.2f}s")
                        else:
                            validation_results['checks_passed'] += 1
                    else:
                        validation_results['checks_passed'] += 1  # No processing yet is fine
            except Exception as e:
                validation_results['warnings'].append(f"Cannot check performance metrics: {e}")
            
            # Check 7: Error tracking
            validation_results['total_checks'] += 1
            if self.consecutive_errors >= 3:
                validation_results['errors'].append(f"High consecutive error count: {self.consecutive_errors}")
            elif self.error_count > 50:
                validation_results['warnings'].append(f"High total error count: {self.error_count}")
            else:
                validation_results['checks_passed'] += 1
            
            # Check 8: Configuration validation
            validation_results['total_checks'] += 1
            config_issues = []
            if not 0.0 <= self.tolerance <= 1.0:
                config_issues.append(f"Invalid tolerance value: {self.tolerance}")
            if self.cooldown_seconds < 0:
                config_issues.append(f"Invalid cooldown: {self.cooldown_seconds}")
            if self.max_image_size < 100:
                config_issues.append(f"Max image size too small: {self.max_image_size}")
            
            if config_issues:
                validation_results['errors'].extend(config_issues)
            else:
                validation_results['checks_passed'] += 1
            
            # Determine overall status
            if validation_results['errors']:
                validation_results['status'] = 'error'
            elif validation_results['warnings']:
                validation_results['status'] = 'warning'
            else:
                validation_results['status'] = 'healthy'
            
            # Add summary statistics
            validation_results.update({
                'total_employees': len(self.known_employee_ids),
                'total_detections': self.detection_count,
                'total_attendance_logs': self.attendance_logs,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'success_rate': f"{(validation_results['checks_passed'] / validation_results['total_checks'] * 100):.1f}%"
            })
            
            logger.info(f"System validation completed: {validation_results['status']} "
                       f"({validation_results['checks_passed']}/{validation_results['total_checks']} checks passed)")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during system validation: {e}")
            validation_results.update({
                'status': 'error',
                'errors': [f"Validation failed: {e}"],
                'checks_passed': 0
            })
            return validation_results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            # Final backup if enabled
            if self.backup_enabled:
                self._create_backup()
            
            # Force final garbage collection
            gc.collect()
            
            logger.info("Safe Employee Attendance Module shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Convenience function for creating the safe module
def create_safe_attendance_module(**kwargs) -> EmployeeAttendanceModule:
    """Create and return a EmployeeAttendanceModule instance."""
    try:
        return EmployeeAttendanceModule(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create safe attendance module: {e}")
        raise


if __name__ == "__main__":
    print("Safe Employee Attendance Module - Testing Mode")
    
    try:
        # Initialize with safety measures
        with EmployeeAttendanceModule(
            face_dir="faces",
            attendance_file="attendance.xlsx",
            cooldown_seconds=5,
            tolerance=0.5,
            backup_enabled=True,
            max_image_size=800  # Smaller images for safety
        ) as attendance:
            
            stats = attendance.get_statistics()
            print("\nSafe Module Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Test employee list
            employees = attendance.get_employee_list()
            if employees:
                print(f"\nLoaded Employees ({len(employees)}):")
                for emp in employees:
                    print(f"  - {emp['employee_id']}: {emp['employee_name']}")
            else:
                print("\nNo employees loaded")
            
            print("\nSafe module test completed successfully")
            
    except Exception as e:
        print(f"Safe module test failed: {e}")
        import traceback
        traceback.print_exc()