#!/usr/bin/env python3
"""
Complete Employee Attendance Module with Segmentation Fault Prevention
====================================================================

This is a comprehensive facial recognition-based attendance system with:
1. Complete thread safety and synchronization
2. Graceful shutdown procedures
3. Memory leak prevention
4. All method implementations completed
5. Comprehensive error handling and recovery
6. Backup and restore functionality
7. Detailed reporting capabilities

Key Features:
- Safe face recognition processing
- Real-time attendance logging
- Comprehensive statistics and reporting
- Thread-safe operations
- Memory management
- Backup/restore functionality
- Excel-based attendance logging
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import face_recognition
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import pickle
import threading
import time
import shutil
import hashlib
import gc
import signal
import atexit
import weakref
from contextlib import contextmanager
import traceback
import json

# Configure logging with thread safety
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('attendance_system.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class SafeShutdownHandler:
    """Handles safe shutdown procedures to prevent segfaults."""
    
    def __init__(self):
        self.shutdown_requested = threading.Event()
        self.active_threads = set()
        self.cleanup_callbacks = []
        self._lock = threading.RLock()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register atexit handler
        atexit.register(self.cleanup)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.request_shutdown()
    
    def request_shutdown(self):
        """Request graceful shutdown of all components."""
        self.shutdown_requested.set()
        logger.info("Shutdown requested, cleaning up...")
    
    def register_thread(self, thread):
        """Register a thread for cleanup tracking."""
        with self._lock:
            self.active_threads.add(thread)
    
    def unregister_thread(self, thread):
        """Unregister a thread from cleanup tracking."""
        with self._lock:
            self.active_threads.discard(thread)
    
    def register_cleanup(self, callback):
        """Register a cleanup callback."""
        with self._lock:
            self.cleanup_callbacks.append(callback)
    
    def cleanup(self):
        """Perform complete system cleanup."""
        logger.info("Starting safe shutdown cleanup...")
        
        try:
            # Set shutdown flag
            self.shutdown_requested.set()
            
            # Execute cleanup callbacks
            with self._lock:
                for callback in reversed(self.cleanup_callbacks):
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Error in cleanup callback: {e}")
            
            # Wait for threads to finish
            timeout = 5.0
            start_time = time.time()
            
            with self._lock:
                active_threads = list(self.active_threads)
            
            for thread in active_threads:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time > 0 and thread.is_alive():
                    thread.join(timeout=remaining_time)
                    if thread.is_alive():
                        logger.warning(f"Thread {thread.name} did not shutdown gracefully")
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Safe shutdown cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown cleanup: {e}")

# Global shutdown handler
_shutdown_handler = SafeShutdownHandler()

class ThreadSafeCounter:
    """Thread-safe counter for statistics."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
    
    def get(self) -> int:
        with self._lock:
            return self._value
    
    def reset(self):
        with self._lock:
            self._value = 0

class MemoryManager:
    """Manages memory usage and prevents leaks."""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cleanup_threshold = 0.8  # Clean up at 80% of max memory
        
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss
            
            if memory_usage > self.max_memory_bytes * self.cleanup_threshold:
                logger.warning(f"High memory usage: {memory_usage / 1024 / 1024:.1f} MB")
                gc.collect()
                return False
            
            return True
        except ImportError:
            # psutil not available, skip memory checking
            return True
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")
            return True
    
    def force_cleanup(self):
        """Force memory cleanup."""
        gc.collect()

class EmployeeAttendanceModule:
    """
    Complete Employee Attendance Module with comprehensive safety measures.
    """
    
    def __init__(self, 
                 face_dir: str = "faces",
                 attendance_file: str = "attendance.xlsx",
                 cooldown_seconds: int = 600,
                 tolerance: float = 0.5,
                 encodings_cache: str = "face_encodings.pkl",
                 backup_enabled: bool = True,
                 max_image_size: int = 1024,
                 max_memory_mb: int = 512):
        """Initialize the Safe Employee Attendance Module."""
        
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
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(max_memory_mb)
        
        # Thread safety and shutdown management
        self.shutdown_requested = threading.Event()
        self._shutdown_lock = threading.RLock()
        
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
        
        # Performance tracking with thread-safe counters
        self.detection_count = ThreadSafeCounter()
        self.attendance_logs = ThreadSafeCounter()
        self.processing_times = []
        self._stats_lock = threading.Lock()
        
        # Error tracking
        self.error_count = ThreadSafeCounter()
        self.last_error_time = None
        self.consecutive_errors = ThreadSafeCounter()
        
        # Active resources tracking
        self._active_resources = weakref.WeakSet()
        
        # Register for cleanup
        _shutdown_handler.register_cleanup(self._cleanup_resources)
        
        # Initialize the module safely
        self._initialize_module_safely()
        
        logger.info(f"Safe Employee Attendance Module initialized with {len(self.known_employee_ids)} employees")
    
    def _initialize_module_safely(self):
        """Initialize the module with comprehensive safety checks."""
        try:
            # Check for shutdown request
            if _shutdown_handler.shutdown_requested.is_set():
                logger.warning("Shutdown requested during initialization")
                return
            
            # Create directories
            self._create_directories()
            
            # Initialize attendance file
            self._initialize_attendance_file()
            
            # Load known faces with safety measures
            if not self.load_known_faces_safely():
                logger.warning("No faces loaded, system will run in detection-only mode")
            
            # Force garbage collection after initialization
            self.memory_manager.force_cleanup()
            
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
            Path("logs"),
            Path("reports")
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
        """Load and encode all faces with safety measures to prevent segfaults."""
        try:
            if _shutdown_handler.shutdown_requested.is_set():
                logger.info("Shutdown requested, skipping face loading")
                return False
            
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
        
        logger.info(f"Loading {len(image_files)} face images from: {self.face_dir}")
        
        # Process images sequentially for safety
        loaded_count = 0
        new_encodings = []
        new_employee_ids = []
        new_metadata = {}
        
        for i, image_path in enumerate(image_files):
            # Check for shutdown request
            if _shutdown_handler.shutdown_requested.is_set():
                logger.info("Shutdown requested during face loading")
                break
            
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
                
                # Memory management
                self.memory_manager.check_memory_usage()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Critical error processing {image_path}: {e}")
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
            
            # Load and validate image using OpenCV first
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
                    'name': employee_id
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
        """Process a video frame for face detection and attendance logging."""
        # Check for shutdown request
        if _shutdown_handler.shutdown_requested.is_set():
            logger.debug("Shutdown requested, skipping frame processing")
            return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8), []
        
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
            self.detection_count.increment()
            current_time = datetime.now()
            
            # Validate frame dimensions
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                logger.warning(f"Invalid frame shape: {frame.shape}")
                return frame, []
            
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                logger.warning(f"Invalid frame dimensions: {width}x{height}")
                return frame, []
            
            # Memory check before processing
            if not self.memory_manager.check_memory_usage():
                logger.warning("High memory usage, skipping frame processing")
                return frame, []
            
            # Resize frame for faster processing
            processing_scale = 0.25
            small_frame = cv2.resize(frame, (0, 0), fx=processing_scale, fy=processing_scale)
            
            # Convert BGR to RGB
            try:
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                logger.warning(f"Color conversion error: {e}")
                return frame, []
            
            # Detect faces
            try:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                
                # Limit number of faces processed
                max_faces = 3
                if len(face_locations) > max_faces:
                    logger.info(f"Too many faces detected ({len(face_locations)}), processing only first {max_faces}")
                    face_locations = face_locations[:max_faces]
                
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
            except Exception as e:
                logger.warning(f"Face detection error: {e}")
                return frame, []
            
            detection_results = []
            annotated_frame = frame.copy()
            
            # Process each detected face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Check for shutdown during processing
                if _shutdown_handler.shutdown_requested.is_set():
                    break
                
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
            self.consecutive_errors.reset()
            
            # Periodic memory cleanup
            if self.detection_count.get() % 30 == 0:
                self.memory_manager.force_cleanup()
            
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
                self.attendance_logs.increment()
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
        self.error_count.increment()
        self.last_error_time = datetime.now()
        self.consecutive_errors.increment()
        
        logger.error(f"Error #{self.error_count.get()}: {error}")
        
        # Force garbage collection on errors
        self.memory_manager.force_cleanup()
        
        if self.consecutive_errors.get() >= 3:
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
            self.memory_manager.force_cleanup()
            
            # Reload faces with safety measures
            if self.load_known_faces_safely():
                logger.info("Face data reloaded successfully")
                self.consecutive_errors.reset()
            else:
                logger.warning("Face data reload failed")
                
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
    
    def _cleanup_resources(self):
        """Clean up all resources during shutdown."""
        try:
            logger.info("Cleaning up attendance module resources...")
            
            # Set shutdown flag
            self.shutdown_requested.set()
            
            # Wait a moment for any ongoing operations to complete
            time.sleep(0.1)
            
            # Clear all data structures
            with self._face_data_lock:
                self.known_face_encodings.clear()
                self.known_employee_ids.clear()
                self.employee_metadata.clear()
            
            with self._attendance_lock:
                self.last_seen_time.clear()
                self.visit_counts.clear()
                self.daily_stats.clear()
            
            # Clear processing times
            with self._stats_lock:
                self.processing_times.clear()
            
            # Force final memory cleanup
            self.memory_manager.force_cleanup()
            
            logger.info("Attendance module cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
    
    # ========== PUBLIC API METHODS ==========
    
    def get_statistics(self) -> Dict:
        """Get comprehensive module statistics."""
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
                'total_detections': self.detection_count.get(),
                'total_attendance_logs': self.attendance_logs.get(),
                'average_processing_time_ms': avg_processing_time * 1000,
                'error_count': self.error_count.get(),
                'consecutive_errors': self.consecutive_errors.get(),
                'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
                'safe_mode': True,
                'max_image_size': self.max_image_size,
                'tolerance': self.tolerance,
                'cooldown_seconds': self.cooldown_seconds,
                'memory_manager_active': True,
                'shutdown_handler_active': not _shutdown_handler.shutdown_requested.is_set()
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
                logger.error("No attendance data found")
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
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            logger.info(f"Successfully exported {len(df)} records to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting attendance data: {e}")
            return False
    
    def backup_system(self) -> bool:
        """Create a complete system backup."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"backup/system_backup_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup attendance file
            if self.attendance_file.exists():
                shutil.copy2(self.attendance_file, backup_dir / "attendance.xlsx")
            
            # Backup encodings cache
            if self.encodings_cache.exists():
                shutil.copy2(self.encodings_cache, backup_dir / "face_encodings.pkl")
            
            # Backup face images
            if self.face_dir.exists():
                faces_backup = backup_dir / "faces"
                shutil.copytree(self.face_dir, faces_backup, dirs_exist_ok=True)
            
            # Create backup metadata
            metadata = {
                'backup_timestamp': datetime.now().isoformat(),
                'system_stats': self.get_statistics(),
                'employee_list': self.get_employee_list(),
                'backup_version': '1.0'
            }
            
            with open(backup_dir / "backup_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"System backup created: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating system backup: {e}")
            return False
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore system from a backup."""
        try:
            backup_dir = Path(backup_path)
            
            if not backup_dir.exists():
                logger.error(f"Backup directory does not exist: {backup_dir}")
                return False
            
            # Check backup metadata
            metadata_file = backup_dir / "backup_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Restoring backup from: {metadata.get('backup_timestamp', 'Unknown')}")
            
            # Restore attendance file
            backup_attendance = backup_dir / "attendance.xlsx"
            if backup_attendance.exists():
                shutil.copy2(backup_attendance, self.attendance_file)
                logger.info("Restored attendance file")
            
            # Restore encodings cache
            backup_encodings = backup_dir / "face_encodings.pkl"
            if backup_encodings.exists():
                shutil.copy2(backup_encodings, self.encodings_cache)
                logger.info("Restored encodings cache")
            
            # Restore face images
            backup_faces = backup_dir / "faces"
            if backup_faces.exists():
                if self.face_dir.exists():
                    shutil.rmtree(self.face_dir)
                shutil.copytree(backup_faces, self.face_dir)
                logger.info("Restored face images")
            
            # Reload system after restore
            self.load_known_faces_safely()
            
            logger.info(f"System restored from backup: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
    
    def export_attendance_report(self, output_dir: str = "reports", 
                                start_date: str = None, end_date: str = None) -> bool:
        """Export comprehensive attendance report with analytics."""
        try:
            if not self.attendance_file.exists():
                logger.error("No attendance data available for report generation")
                return False
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load attendance data
            df = pd.read_excel(self.attendance_file, engine='openpyxl')
            
            if df.empty:
                logger.error("No attendance data found")
                return False
            
            # Filter by date range if provided
            filtered_df = df.copy()
            
            if start_date or end_date:
                filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
                
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    filtered_df = filtered_df[filtered_df['Date'] >= start_dt]
                
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    filtered_df = filtered_df[filtered_df['Date'] <= end_dt]
            
            # Generate timestamp for report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create comprehensive report
            report_data = {
                'generation_info': {
                    'generated_at': datetime.now().isoformat(),
                    'total_records': len(filtered_df),
                    'date_range': f"{start_date or 'All'} to {end_date or 'All'}",
                    'unique_employees': filtered_df['Employee_ID'].nunique() if not filtered_df.empty else 0
                },
                'summary_statistics': {},
                'employee_details': {},
                'daily_summary': {},
                'visit_type_analysis': {}
            }
            
            if not filtered_df.empty:
                # Summary statistics
                report_data['summary_statistics'] = {
                    'total_visits': len(filtered_df),
                    'unique_employees': filtered_df['Employee_ID'].nunique(),
                    'average_visits_per_employee': len(filtered_df) / filtered_df['Employee_ID'].nunique(),
                    'date_range_days': (pd.to_datetime(filtered_df['Date'].max()) - 
                                      pd.to_datetime(filtered_df['Date'].min())).days + 1 if len(filtered_df) > 1 else 1,
                    'first_visit': filtered_df['Timestamp'].min(),
                    'last_visit': filtered_df['Timestamp'].max()
                }
                
                # Employee-wise analysis
                employee_stats = filtered_df.groupby('Employee_ID').agg({
                    'Employee_Name': 'first',
                    'Visit_Count': 'max',
                    'Date': ['count', 'min', 'max'],
                    'Confidence': 'mean'
                }).round(3)
                
                employee_stats.columns = ['Name', 'Total_Visits', 'Days_Active', 'First_Date', 'Last_Date', 'Avg_Confidence']
                report_data['employee_details'] = employee_stats.to_dict('index')
                
                # Daily summary
                daily_stats = filtered_df.groupby('Date').agg({
                    'Employee_ID': 'nunique',
                    'Visit_Count': 'sum'
                })
                daily_stats.columns = ['Unique_Employees', 'Total_Visits']
                report_data['daily_summary'] = daily_stats.to_dict('index')
                
                # Visit type analysis
                if 'Visit_Type' in filtered_df.columns:
                    visit_type_stats = filtered_df.groupby('Visit_Type').size().to_dict()
                    report_data['visit_type_analysis'] = visit_type_stats
            
            # Export detailed Excel report
            excel_filename = f"attendance_report_{timestamp}.xlsx"
            excel_path = output_path / excel_filename
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Raw data
                filtered_df.to_excel(writer, sheet_name='Raw_Data', index=False)
                
                # Summary statistics
                if report_data['employee_details']:
                    pd.DataFrame.from_dict(report_data['employee_details'], orient='index').to_excel(
                        writer, sheet_name='Employee_Summary'
                    )
                
                # Daily summary
                if report_data['daily_summary']:
                    pd.DataFrame.from_dict(report_data['daily_summary'], orient='index').to_excel(
                        writer, sheet_name='Daily_Summary'
                    )
                
                # System statistics
                system_stats = self.get_statistics()
                pd.DataFrame([system_stats]).to_excel(writer, sheet_name='System_Stats', index=False)
            
            # Export JSON report for API consumption
            json_filename = f"attendance_report_{timestamp}.json"
            json_path = output_path / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            # Create summary text report
            text_filename = f"attendance_summary_{timestamp}.txt"
            text_path = output_path / text_filename
            
            with open(text_path, 'w') as f:
                f.write("EMPLOYEE ATTENDANCE REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {report_data['generation_info']['generated_at']}\n")
                f.write(f"Date Range: {report_data['generation_info']['date_range']}\n")
                f.write(f"Total Records: {report_data['generation_info']['total_records']}\n")
                f.write(f"Unique Employees: {report_data['generation_info']['unique_employees']}\n\n")
                
                if report_data['summary_statistics']:
                    f.write("SUMMARY STATISTICS\n")
                    f.write("-" * 20 + "\n")
                    for key, value in report_data['summary_statistics'].items():
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                    f.write("\n")
                
                if report_data['visit_type_analysis']:
                    f.write("VISIT TYPE BREAKDOWN\n")
                    f.write("-" * 20 + "\n")
                    for visit_type, count in report_data['visit_type_analysis'].items():
                        f.write(f"{visit_type}: {count}\n")
                    f.write("\n")
                
                f.write("FILES GENERATED\n")
                f.write("-" * 15 + "\n")
                f.write(f"Excel Report: {excel_filename}\n")
                f.write(f"JSON Data: {json_filename}\n")
                f.write(f"Text Summary: {text_filename}\n")
            
            logger.info(f"Successfully generated attendance report in: {output_path}")
            logger.info(f"Report covers {report_data['generation_info']['total_records']} records")
            logger.info(f"Files generated: {excel_filename}, {json_filename}, {text_filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating attendance report: {e}")
            return False
    
    def get_live_camera_feed(self, camera_index: int = 0) -> bool:
        """Start live camera feed for real-time attendance monitoring."""
        try:
            logger.info(f"Starting live camera feed (camera {camera_index})")
            
            # Initialize camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                logger.error(f"Could not open camera {camera_index}")
                return False
            
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera initialized successfully. Press 'q' to quit, 's' to take screenshot")
            
            frame_count = 0
            screenshot_count = 0
            
            while True:
                # Check for shutdown request
                if _shutdown_handler.shutdown_requested.is_set():
                    logger.info("Shutdown requested, stopping camera feed")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                frame_count += 1
                
                # Process every nth frame to reduce load
                if frame_count % 3 == 0:  # Process every 3rd frame
                    processed_frame, detections = self.process_frame(frame)
                    
                    # Display frame
                    cv2.imshow('Employee Attendance System', processed_frame)
                    
                    # Print detection info
                    if detections:
                        for detection in detections:
                            emp_id = detection['employee_id']
                            confidence = detection['confidence']
                            visit_type = detection.get('visit_type', 'N/A')
                            logger.info(f"Detected: {emp_id} (confidence: {confidence:.2f}, type: {visit_type})")
                else:
                    # Just display the raw frame
                    cv2.imshow('Employee Attendance System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User requested quit")
                    break
                elif key == ord('s'):
                    # Take screenshot
                    screenshot_path = f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    Path("screenshots").mkdir(exist_ok=True)
                    cv2.imwrite(screenshot_path, frame)
                    screenshot_count += 1
                    logger.info(f"Screenshot saved: {screenshot_path}")
                elif key == ord('r'):
                    # Reload face data
                    logger.info("Reloading face data...")
                    if self.load_known_faces_safely():
                        logger.info("Face data reloaded successfully")
                    else:
                        logger.warning("Face data reload failed")
                elif key == ord('b'):
                    # Create backup
                    logger.info("Creating system backup...")
                    if self.backup_system():
                        logger.info("Backup created successfully")
                    else:
                        logger.warning("Backup creation failed")
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            logger.info(f"Camera feed stopped. Processed {frame_count} frames, {screenshot_count} screenshots taken")
            return True
            
        except Exception as e:
            logger.error(f"Error in live camera feed: {e}")
            try:
                cap.release()
                cv2.destroyAllWindows()
            except:
                pass
            return False
    
    def process_video_file(self, video_path: str, output_path: str = None) -> bool:
        """Process a video file for attendance detection."""
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                logger.error(f"Video file does not exist: {video_path}")
                return False
            
            logger.info(f"Processing video file: {video_path}")
            
            # Initialize video capture
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Video properties: {frame_count} frames, {fps} FPS, {duration:.1f} seconds")
            
            # Setup output video if requested
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            else:
                out = None
            
            # Process video
            processed_frames = 0
            detections_log = []
            
            while True:
                # Check for shutdown request
                if _shutdown_handler.shutdown_requested.is_set():
                    logger.info("Shutdown requested, stopping video processing")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                processed_frames += 1
                
                # Log detections with timestamp
                if detections:
                    frame_time = processed_frames / fps
                    for detection in detections:
                        detection['video_timestamp'] = frame_time
                        detections_log.append(detection.copy())
                
                # Write to output video
                if out:
                    out.write(processed_frame)
                
                # Progress reporting
                if processed_frames % (fps * 10) == 0:  # Every 10 seconds
                    progress = (processed_frames / frame_count) * 100
                    logger.info(f"Processing progress: {progress:.1f}% ({processed_frames}/{frame_count} frames)")
            
            # Cleanup
            cap.release()
            if out:
                out.release()
            
            logger.info(f"Video processing completed: {processed_frames} frames processed")
            logger.info(f"Total detections: {len(detections_log)}")
            
            # Save detections log
            if detections_log:
                log_path = video_path.parent / f"{video_path.stem}_detections.json"
                with open(log_path, 'w') as f:
                    json.dump(detections_log, f, indent=2, default=str)
                logger.info(f"Detections log saved: {log_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing video file: {e}")
            try:
                cap.release()
                if 'out' in locals() and out:
                    out.release()
            except:
                pass
            return False
    
    def get_attendance_trends(self, days: int = 30) -> Dict:
        """Analyze attendance trends over specified number of days."""
        try:
            if not self.attendance_file.exists():
                return {'error': 'No attendance data available'}
            
            df = pd.read_excel(self.attendance_file, engine='openpyxl')
            if df.empty:
                return {'error': 'No attendance data found'}
            
            # Filter data for specified days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df_filtered = df[df['Date'] >= start_date.date()]
            
            if df_filtered.empty:
                return {'error': f'No data found for last {days} days'}
            
            # Calculate trends
            trends = {
                'period': f'Last {days} days',
                'start_date': start_date.date().isoformat(),
                'end_date': end_date.date().isoformat(),
                'total_visits': len(df_filtered),
                'unique_employees': df_filtered['Employee_ID'].nunique(),
                'daily_averages': {},
                'employee_activity': {},
                'peak_hours': {},
                'visit_patterns': {}
            }
            
            # Daily statistics
            daily_stats = df_filtered.groupby('Date').agg({
                'Employee_ID': 'nunique',
                'Visit_Count': 'sum'
            })
            daily_stats.columns = ['unique_employees', 'total_visits']
            
            trends['daily_averages'] = {
                'avg_employees_per_day': daily_stats['unique_employees'].mean(),
                'avg_visits_per_day': daily_stats['total_visits'].mean(),
                'most_active_day': daily_stats['total_visits'].idxmax().isoformat(),
                'least_active_day': daily_stats['total_visits'].idxmin().isoformat()
            }
            
            # Employee activity ranking
            employee_activity = df_filtered.groupby('Employee_ID').agg({
                'Employee_Name': 'first',
                'Date': 'nunique',
                'Visit_Count': 'sum'
            }).sort_values('Visit_Count', ascending=False)
            
            trends['employee_activity'] = employee_activity.head(10).to_dict('index')
            
            # Peak hours analysis
            if 'Time' in df_filtered.columns:
                df_filtered['Hour'] = pd.to_datetime(df_filtered['Time']).dt.hour
                hourly_activity = df_filtered.groupby('Hour').size()
                
                trends['peak_hours'] = {
                    'busiest_hour': int(hourly_activity.idxmax()),
                    'quietest_hour': int(hourly_activity.idxmin()),
                    'hourly_distribution': hourly_activity.to_dict()
                }
            
            # Visit patterns
            if 'Visit_Type' in df_filtered.columns:
                visit_patterns = df_filtered['Visit_Type'].value_counts()
                trends['visit_patterns'] = visit_patterns.to_dict()
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing attendance trends: {e}")
            return {'error': str(e)}
    
    def maintenance_mode(self, enable: bool = True) -> bool:
        """Enable or disable maintenance mode."""
        try:
            if enable:
                logger.info("Entering maintenance mode...")
                
                # Create maintenance backup
                if self.backup_system():
                    logger.info("Maintenance backup created")
                
                # Clear memory caches
                self.memory_manager.force_cleanup()
                
                # Validate data integrity
                integrity_check = self._check_data_integrity()
                logger.info(f"Data integrity check: {'PASSED' if integrity_check else 'FAILED'}")
                
                logger.info("Maintenance mode enabled")
                return True
            else:
                logger.info("Exiting maintenance mode...")
                
                # Reload face data
                if self.load_known_faces_safely():
                    logger.info("Face data reloaded successfully")
                
                logger.info("Maintenance mode disabled")
                return True
                
        except Exception as e:
            logger.error(f"Error in maintenance mode: {e}")
            return False
    
    def _check_data_integrity(self) -> bool:
        """Check data integrity of all system components."""
        try:
            integrity_issues = []
            
            # Check face encodings consistency
            with self._face_data_lock:
                if len(self.known_face_encodings) != len(self.known_employee_ids):
                    integrity_issues.append("Face encodings and employee IDs count mismatch")
                
                for emp_id in self.known_employee_ids:
                    if emp_id not in self.employee_metadata:
                        integrity_issues.append(f"Missing metadata for employee: {emp_id}")
            
            # Check attendance file
            if self.attendance_file.exists():
                try:
                    df = pd.read_excel(self.attendance_file, engine='openpyxl')
                    required_columns = ['Employee_ID', 'Employee_Name', 'Date', 'Time', 'Timestamp']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        integrity_issues.append(f"Missing attendance columns: {missing_columns}")
                except Exception as e:
                    integrity_issues.append(f"Attendance file corruption: {e}")
            
            # Check encodings cache
            if self.encodings_cache.exists():
                try:
                    with open(self.encodings_cache, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    if 'encodings' not in cache_data or 'employee_ids' not in cache_data:
                        integrity_issues.append("Invalid encodings cache structure")
                except Exception as e:
                    integrity_issues.append(f"Encodings cache corruption: {e}")
            
            if integrity_issues:
                logger.warning(f"Data integrity issues found: {integrity_issues}")
                return False
            else:
                logger.info("All data integrity checks passed")
                return True
                
        except Exception as e:
            logger.error(f"Error checking data integrity: {e}")
            return False
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health information."""
        try:
            health_info = {
                'timestamp': datetime.now().isoformat(),
                'status': 'HEALTHY',
                'components': {},
                'performance': {},
                'recommendations': []
            }
            
            # Check face recognition component
            with self._face_data_lock:
                face_component_health = {
                    'status': 'OK' if len(self.known_face_encodings) > 0 else 'WARNING',
                    'total_faces': len(self.known_face_encodings),
                    'cache_exists': self.encodings_cache.exists(),
                    'face_dir_exists': self.face_dir.exists()
                }
                health_info['components']['face_recognition'] = face_component_health
            
            # Check attendance logging component
            attendance_component_health = {
                'status': 'OK' if self.attendance_file.exists() else 'ERROR',
                'file_exists': self.attendance_file.exists(),
                'total_logs': self.attendance_logs.get(),
                'backup_enabled': self.backup_enabled
            }
            health_info['components']['attendance_logging'] = attendance_component_health
            
            # Performance metrics
            with self._stats_lock:
                avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
                
            performance_info = {
                'avg_processing_time_ms': avg_processing_time * 1000,
                'total_detections': self.detection_count.get(),
                'error_rate': self.error_count.get() / max(1, self.detection_count.get()),
                'consecutive_errors': self.consecutive_errors.get(),
                'memory_usage_ok': self.memory_manager.check_memory_usage()
            }
            health_info['performance'] = performance_info
            
            # Generate recommendations
            recommendations = []
            
            if len(self.known_face_encodings) == 0:
                recommendations.append("No face encodings loaded. Add employee faces to the system.")
            
            if not self.attendance_file.exists():
                recommendations.append("Attendance file not found. System will create one on first detection.")
            
            if self.consecutive_errors.get() > 0:
                recommendations.append("Recent errors detected. Consider running maintenance mode.")
            
            if avg_processing_time > 0.5:  # More than 500ms
                recommendations.append("High processing times detected. Consider reducing image sizes or tolerance.")
            
            if not self.memory_manager.check_memory_usage():
                recommendations.append("High memory usage detected. Consider reducing max_memory_mb setting.")
            
            health_info['recommendations'] = recommendations
            
            # Overall status
            if any(comp['status'] == 'ERROR' for comp in health_info['components'].values()):
                health_info['status'] = 'ERROR'
            elif any(comp['status'] == 'WARNING' for comp in health_info['components'].values()):
                health_info['status'] = 'WARNING'
            
            return health_info
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'ERROR',
                'error': str(e)
            }

# ========== MAIN EXECUTION AND EXAMPLES ==========

def main():
    """Main function demonstrating usage of the Employee Attendance Module."""
    try:
        logger.info("Starting Employee Attendance System Demo")
        
        # Initialize the module
        attendance_module = EmployeeAttendanceModule(
            face_dir="faces",
            attendance_file="attendance.xlsx",
            cooldown_seconds=300,  # 5 minutes
            tolerance=0.5,
            backup_enabled=True,
            max_image_size=800,
            max_memory_mb=256
        )
        
        # Display system health
        health = attendance_module.get_system_health()
        logger.info(f"System Health: {health['status']}")
        
        # Display statistics
        stats = attendance_module.get_statistics()
        logger.info(f"Loaded {stats['total_employees']} employees")
        
        # Demo: Live camera feed (uncomment to use)
        # attendance_module.get_live_camera_feed(camera_index=0)
        
        # Demo: Process a video file (uncomment to use)
        # attendance_module.process_video_file("input_video.mp4", "output_video.mp4")
        
        # Demo: Generate attendance report
        if attendance_module.export_attendance_report():
            logger.info("Attendance report generated successfully")
        
        # Demo: Get attendance trends
        trends = attendance_module.get_attendance_trends(days=7)
        if 'error' not in trends:
            logger.info(f"Attendance trends analyzed for {trends['total_visits']} visits")
        
        logger.info("Demo completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    finally:
        logger.info("Shutting down safely...")
        _shutdown_handler.request_shutdown()

if __name__ == "__main__":
    main()