"""
Enhanced MySQL Attendance System for AI Surveillance
===================================================

This module provides MySQL-based attendance tracking with proper database structure,
replacing the Excel-based system for better scalability and data integrity.

Features:
- MySQL database integration with proper schema
- Employee management (add/remove/update)
- Real-time attendance logging with visit tracking
- Daily attendance summaries
- Comprehensive reporting
- Data backup and restoration
- Integration with existing surveillance system

Author: AI Surveillance System
Version: 2.0
"""

import mysql.connector
from mysql.connector import Error
import cv2
import numpy as np
import pandas as pd
import face_recognition
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import os
from pathlib import Path
import pickle
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MySQLAttendanceSystem:
    """
    MySQL-based Employee Attendance System with comprehensive tracking.
    """
    
    def __init__(self,
                 host: str = "localhost",
                 user: str = "root", 
                 password: str = "root",
                 database: str = "attendance_db",
                 face_dir: str = "faces",
                 cooldown_seconds: int = 600,  # 10 minutes default
                 tolerance: float = 0.5):
        """
        Initialize MySQL Attendance System.
        
        Args:
            host (str): MySQL host
            user (str): MySQL username
            password (str): MySQL password
            database (str): Database name
            face_dir (str): Directory containing employee face images
            cooldown_seconds (int): Cooldown between attendance logs (10 minutes = 600 seconds)
            tolerance (float): Face recognition tolerance
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.face_dir = Path(face_dir)
        self.cooldown_seconds = cooldown_seconds
        self.tolerance = tolerance
        
        # Face recognition data
        self.known_face_encodings: List[np.ndarray] = []
        self.known_employee_ids: List[str] = []
        self.employee_names: Dict[str, str] = {}
        
        # Database connection
        self.connection = None
        
        # Performance tracking
        self.detection_count = 0
        self.attendance_logs = 0
        
        # Initialize system
        self._create_directories()
        self._initialize_database()
        self.load_known_faces()
        
        logger.info(f"MySQL Attendance System initialized with {len(self.known_employee_ids)} employees")
    
    def _create_directories(self):
        """Create necessary directories."""
        self.face_dir.mkdir(exist_ok=True)
        if not any(self.face_dir.iterdir()):
            logger.warning(f"Face directory '{self.face_dir}' is empty. Please add employee face images.")
    
    def _get_connection(self):
        """Get MySQL database connection."""
        try:
            if self.connection is None or not self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    autocommit=True,
                    charset='utf8mb4',
                    collation='utf8mb4_unicode_ci'
                )
            return self.connection
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return None
    
    def _initialize_database(self):
        """Initialize MySQL database with proper schema."""
        try:
            # First, create database if it doesn't exist
            temp_connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            cursor = temp_connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            cursor.close()
            temp_connection.close()
            
            # Now connect to the database and create tables
            connection = self._get_connection()
            if connection is None:
                raise Exception("Could not establish database connection")
            
            cursor = connection.cursor()
            
            # Create employees table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id VARCHAR(50) PRIMARY KEY,
                    employee_name VARCHAR(255) NOT NULL,
                    face_encoding LONGTEXT,
                    image_path VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    department VARCHAR(100),
                    position VARCHAR(100),
                    email VARCHAR(255),
                    phone VARCHAR(20)
                )
            """)
            
            # Create attendance_logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    employee_id VARCHAR(50) NOT NULL,
                    employee_name VARCHAR(255) NOT NULL,
                    log_date DATE NOT NULL,
                    log_time TIME NOT NULL,
                    log_datetime DATETIME NOT NULL,
                    visit_type ENUM('IN', 'OUT') NOT NULL,
                    confidence FLOAT NOT NULL,
                    visit_count INT DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
                    INDEX idx_employee_date (employee_id, log_date),
                    INDEX idx_log_datetime (log_datetime),
                    INDEX idx_visit_type (visit_type)
                )
            """)
            
            # Create daily_attendance_summary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_attendance_summary (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    employee_id VARCHAR(50) NOT NULL,
                    employee_name VARCHAR(255) NOT NULL,
                    attendance_date DATE NOT NULL,
                    first_in_time TIME,
                    last_out_time TIME,
                    total_hours DECIMAL(5,2) DEFAULT 0.00,
                    total_visits INT DEFAULT 0,
                    total_in_visits INT DEFAULT 0,
                    total_out_visits INT DEFAULT 0,
                    status ENUM('PRESENT', 'ABSENT', 'PARTIAL') DEFAULT 'PRESENT',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
                    UNIQUE KEY unique_employee_date (employee_id, attendance_date),
                    INDEX idx_attendance_date (attendance_date),
                    INDEX idx_status (status)
                )
            """)
            
            # Create system_logs table for tracking system events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    log_type ENUM('INFO', 'WARNING', 'ERROR', 'ATTENDANCE', 'SYSTEM') NOT NULL,
                    message TEXT NOT NULL,
                    employee_id VARCHAR(50),
                    additional_data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_log_type (log_type),
                    INDEX idx_created_at (created_at)
                )
            """)
            
            cursor.close()
            logger.info("Database schema initialized successfully")
            
        except Error as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def load_known_faces(self) -> bool:
        """Load face encodings from database and face images."""
        try:
            connection = self._get_connection()
            if connection is None:
                return False
            
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT employee_id, employee_name, face_encoding, image_path FROM employees WHERE is_active = TRUE")
            employees = cursor.fetchall()
            cursor.close()
            
            # Load from database first
            loaded_from_db = 0
            for employee in employees:
                if employee['face_encoding']:
                    try:
                        # Decode face encoding from JSON
                        encoding = np.array(json.loads(employee['face_encoding']))
                        self.known_face_encodings.append(encoding)
                        self.known_employee_ids.append(employee['employee_id'])
                        self.employee_names[employee['employee_id']] = employee['employee_name']
                        loaded_from_db += 1
                    except Exception as e:
                        logger.warning(f"Error loading encoding for {employee['employee_id']}: {e}")
            
            logger.info(f"Loaded {loaded_from_db} face encodings from database")
            
            # Load new faces from images that aren't in database
            loaded_from_images = self._load_faces_from_images()
            
            return (loaded_from_db + loaded_from_images) > 0
            
        except Error as e:
            logger.error(f"Error loading known faces: {e}")
            return False
    
    def _load_faces_from_images(self) -> int:
        """Load and encode faces from image files not in database."""
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        loaded_count = 0
        
        for image_path in self.face_dir.iterdir():
            if image_path.suffix.lower() not in supported_formats:
                continue
            
            employee_id = image_path.stem
            
            # Check if already in database
            if employee_id in self.known_employee_ids:
                continue
            
            try:
                # Load and encode the face
                image = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) == 0:
                    logger.warning(f"No face found in {image_path}")
                    continue
                elif len(encodings) > 1:
                    logger.warning(f"Multiple faces found in {image_path}, using the first one")
                
                # Store in memory
                self.known_face_encodings.append(encodings[0])
                self.known_employee_ids.append(employee_id)
                self.employee_names[employee_id] = employee_id  # Use ID as name if not provided
                
                # Store in database
                self._save_employee_to_database(employee_id, employee_id, encodings[0], str(image_path))
                
                loaded_count += 1
                logger.info(f"Loaded and saved new employee: {employee_id}")
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
        
        return loaded_count
    
    def _save_employee_to_database(self, employee_id: str, employee_name: str, 
                                  face_encoding: np.ndarray, image_path: str) -> bool:
        """Save employee data to database."""
        try:
            connection = self._get_connection()
            if connection is None:
                return False
            
            cursor = connection.cursor()
            
            # Convert encoding to JSON string
            encoding_json = json.dumps(face_encoding.tolist())
            
            query = """
                INSERT INTO employees (employee_id, employee_name, face_encoding, image_path)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                employee_name = VALUES(employee_name),
                face_encoding = VALUES(face_encoding),
                image_path = VALUES(image_path),
                updated_at = CURRENT_TIMESTAMP
            """
            
            cursor.execute(query, (employee_id, employee_name, encoding_json, image_path))
            cursor.close()
            
            return True
            
        except Error as e:
            logger.error(f"Error saving employee to database: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, existing_detections: List = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process frame for face detection and attendance logging.
        Can work with existing face detections or perform its own detection.
        """
        if len(self.known_face_encodings) == 0:
            return frame, []
        
        self.detection_count += 1
        current_time = datetime.now()
        
        detection_results = []
        annotated_frame = frame.copy()
        
        # Use existing detections if provided, otherwise detect faces
        if existing_detections:
            face_data = self._process_existing_detections(existing_detections, frame)
        else:
            face_data = self._detect_faces_in_frame(frame)
        
        for face_info in face_data:
            employee_id = "Unknown"
            employee_name = "Unknown"
            confidence = 0.0
            bbox = face_info['bbox']
            face_encoding = face_info.get('encoding')
            
            if face_encoding is not None:
                # Match against known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=self.tolerance
                )
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        employee_id = self.known_employee_ids[best_match_index]
                        employee_name = self.employee_names.get(employee_id, employee_id)
                        confidence = 1.0 - face_distances[best_match_index]
                        
                        # Log attendance if conditions are met
                        if self._should_log_attendance(employee_id, current_time):
                            visit_type, visit_count = self._log_attendance(employee_id, employee_name, current_time, confidence)
                        else:
                            visit_type, visit_count = self._get_current_visit_info(employee_id)
            else:
                # No encoding available
                visit_type = None
                visit_count = 0
            
            # Prepare detection result
            detection_result = {
                'employee_id': employee_id,
                'employee_name': employee_name,
                'confidence': confidence,
                'bbox': bbox,
                'visit_type': visit_type,
                'visit_count': visit_count,
                'time': current_time.isoformat()
            }
            detection_results.append(detection_result)
            
            # Draw detection on frame
            annotated_frame = self._draw_detection(annotated_frame, detection_result)
        
        return annotated_frame, detection_results
    
    def _process_existing_detections(self, detections: List, frame: np.ndarray) -> List[Dict]:
        """Process existing face detection objects."""
        face_data = []
        
        for detection in detections:
            try:
                # Extract bbox - handle different formats
                if hasattr(detection, 'bbox'):
                    bbox = detection.bbox
                elif hasattr(detection, 'box'):
                    bbox = detection.box
                elif isinstance(detection, dict):
                    bbox = detection.get('bbox', (0, 0, 100, 100))
                else:
                    bbox = (0, 0, 100, 100)
                
                # Extract or compute encoding
                encoding = None
                if hasattr(detection, 'encoding') and detection.encoding is not None:
                    encoding = detection.encoding
                elif hasattr(detection, 'face_encoding'):
                    encoding = detection.face_encoding
                else:
                    # Extract face region and compute encoding
                    if len(bbox) == 4:
                        left, top, right, bottom = bbox
                        face_region = frame[top:bottom, left:right]
                        if face_region.size > 0:
                            face_encodings = face_recognition.face_encodings(face_region)
                            if face_encodings:
                                encoding = face_encodings[0]
                
                face_data.append({
                    'bbox': bbox,
                    'encoding': encoding
                })
                
            except Exception as e:
                logger.warning(f"Error processing detection: {e}")
                continue
        
        return face_data
    
    def _detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in frame and extract encodings."""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_data = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Scale back up coordinates
            bbox = (left * 2, top * 2, right * 2, bottom * 2)
            face_data.append({
                'bbox': bbox,
                'encoding': encoding
            })
        
        return face_data
    
    def _should_log_attendance(self, employee_id: str, current_time: datetime) -> bool:
        """Check if attendance should be logged based on cooldown."""
        try:
            connection = self._get_connection()
            if connection is None:
                return False
            
            cursor = connection.cursor()
            
            # Get last attendance log for this employee
            query = """
                SELECT log_datetime FROM attendance_logs 
                WHERE employee_id = %s 
                ORDER BY log_datetime DESC 
                LIMIT 1
            """
            cursor.execute(query, (employee_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result is None:
                return True  # First time detection
            
            last_log_time = result[0]
            time_diff = (current_time - last_log_time).total_seconds()
            
            return time_diff >= self.cooldown_seconds
            
        except Error as e:
            logger.error(f"Error checking attendance cooldown: {e}")
            return False
    
    def _log_attendance(self, employee_id: str, employee_name: str, 
                       timestamp: datetime, confidence: float) -> Tuple[str, int]:
        """Log attendance to database and return visit type and count."""
        try:
            connection = self._get_connection()
            if connection is None:
                return None, 0
            
            cursor = connection.cursor()
            
            # Determine visit type (IN/OUT based on last entry)
            visit_type = self._determine_visit_type(employee_id, timestamp.date())
            
            # Get current visit count for today
            today = timestamp.date()
            query = """
                SELECT COUNT(*) FROM attendance_logs 
                WHERE employee_id = %s AND log_date = %s
            """
            cursor.execute(query, (employee_id, today))
            visit_count = cursor.fetchone()[0] + 1
            
            # Insert attendance log
            insert_query = """
                INSERT INTO attendance_logs 
                (employee_id, employee_name, log_date, log_time, log_datetime, 
                 visit_type, confidence, visit_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                employee_id, employee_name, timestamp.date(), timestamp.time(),
                timestamp, visit_type, confidence, visit_count
            ))
            
            # Update daily summary
            self._update_daily_summary(employee_id, employee_name, timestamp, visit_type)
            
            cursor.close()
            
            self.attendance_logs += 1
            logger.info(f"Logged attendance: {employee_name} ({employee_id}) - {visit_type} at {timestamp}")
            
            return visit_type, visit_count
            
        except Error as e:
            logger.error(f"Error logging attendance: {e}")
            return None, 0
    
    def _determine_visit_type(self, employee_id: str, date: datetime.date) -> str:
        """Determine if this should be an IN or OUT visit."""
        try:
            connection = self._get_connection()
            if connection is None:
                return "IN"
            
            cursor = connection.cursor()
            
            # Get last visit type for today
            query = """
                SELECT visit_type FROM attendance_logs 
                WHERE employee_id = %s AND log_date = %s 
                ORDER BY log_datetime DESC 
                LIMIT 1
            """
            cursor.execute(query, (employee_id, date))
            result = cursor.fetchone()
            cursor.close()
            
            if result is None:
                return "IN"  # First visit of the day
            
            last_visit_type = result[0]
            return "OUT" if last_visit_type == "IN" else "IN"
            
        except Error as e:
            logger.error(f"Error determining visit type: {e}")
            return "IN"
    
    def _get_current_visit_info(self, employee_id: str) -> Tuple[str, int]:
        """Get current visit information for employee."""
        try:
            connection = self._get_connection()
            if connection is None:
                return None, 0
            
            cursor = connection.cursor()
            
            today = datetime.now().date()
            query = """
                SELECT visit_type, visit_count FROM attendance_logs 
                WHERE employee_id = %s AND log_date = %s 
                ORDER BY log_datetime DESC 
                LIMIT 1
            """
            cursor.execute(query, (employee_id, today))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return result[0], result[1]
            return None, 0
            
        except Error as e:
            logger.error(f"Error getting visit info: {e}")
            return None, 0
    
    def _update_daily_summary(self, employee_id: str, employee_name: str, 
                             timestamp: datetime, visit_type: str):
        """Update daily attendance summary."""
        try:
            connection = self._get_connection()
            if connection is None:
                return
            
            cursor = connection.cursor()
            today = timestamp.date()
            
            # Get existing summary
            query = """
                SELECT first_in_time, last_out_time, total_visits, 
                       total_in_visits, total_out_visits 
                FROM daily_attendance_summary 
                WHERE employee_id = %s AND attendance_date = %s
            """
            cursor.execute(query, (employee_id, today))
            result = cursor.fetchone()
            
            if result:
                # Update existing summary
                first_in, last_out, total_visits, total_in, total_out = result
                
                if visit_type == "IN":
                    if first_in is None:
                        first_in = timestamp.time()
                    total_in += 1
                else:  # OUT
                    last_out = timestamp.time()
                    total_out += 1
                
                total_visits += 1
                
                # Calculate total hours if we have both in and out
                total_hours = 0.0
                if first_in and last_out:
                    # Simple calculation: last_out - first_in
                    first_datetime = datetime.combine(today, first_in)
                    last_datetime = datetime.combine(today, last_out)
                    if last_datetime > first_datetime:
                        total_hours = (last_datetime - first_datetime).total_seconds() / 3600
                
                update_query = """
                    UPDATE daily_attendance_summary 
                    SET first_in_time = %s, last_out_time = %s, total_hours = %s,
                        total_visits = %s, total_in_visits = %s, total_out_visits = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE employee_id = %s AND attendance_date = %s
                """
                cursor.execute(update_query, (
                    first_in, last_out, total_hours, total_visits, 
                    total_in, total_out, employee_id, today
                ))
            else:
                # Create new summary
                first_in = timestamp.time() if visit_type == "IN" else None
                last_out = timestamp.time() if visit_type == "OUT" else None
                total_in = 1 if visit_type == "IN" else 0
                total_out = 1 if visit_type == "OUT" else 0
                
                insert_query = """
                    INSERT INTO daily_attendance_summary 
                    (employee_id, employee_name, attendance_date, first_in_time, 
                     last_out_time, total_hours, total_visits, total_in_visits, 
                     total_out_visits, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (
                    employee_id, employee_name, today, first_in, last_out,
                    0.0, 1, total_in, total_out, "PRESENT"
                ))
            
            cursor.close()
            
        except Error as e:
            logger.error(f"Error updating daily summary: {e}")
    
    def _draw_detection(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """Draw detection information on frame."""
        bbox = detection['bbox']
        employee_id = detection['employee_id']
        employee_name = detection['employee_name']
        confidence = detection['confidence']
        visit_type = detection.get('visit_type')
        visit_count = detection.get('visit_count', 0)
        
        if len(bbox) == 4:
            left, top, right, bottom = bbox
        else:
            return frame
        
        # Choose color based on recognition status
        if employee_id == "Unknown":
            color = (0, 0, 255)  # Red for unknown
            label = "Unknown"
        else:
            color = (0, 255, 0)  # Green for recognized
            visit_info = f" - {visit_type}" if visit_type else ""
            label = f"{employee_name}{visit_info}"
        
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (left, top - label_size[1] - 10), 
                     (left + label_size[0], top), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (left, top - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw additional info
        if employee_id != "Unknown":
            info_lines = [
                f"ID: {employee_id}",
                f"Conf: {confidence:.2f}",
                f"Visits: {visit_count}"
            ]
            
            for i, info_line in enumerate(info_lines):
                y_pos = bottom + 20 + (i * 15)
                cv2.putText(frame, info_line, (left, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        return frame
    
    def get_daily_attendance(self, date: datetime.date = None) -> List[Dict]:
        """Get daily attendance summary."""
        if date is None:
            date = datetime.now().date()
        
        try:
            connection = self._get_connection()
            if connection is None:
                return []
            
            cursor = connection.cursor(dictionary=True)
            query = """
                SELECT employee_id, employee_name, first_in_time, last_out_time,
                       total_hours, total_visits, total_in_visits, total_out_visits, status
                FROM daily_attendance_summary 
                WHERE attendance_date = %s
                ORDER BY first_in_time
            """
            cursor.execute(query, (date,))
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Error as e:
            logger.error(f"Error getting daily attendance: {e}")
            return []
    
    def get_employee_history(self, employee_id: str, days: int = 30) -> List[Dict]:
        """Get attendance history for specific employee."""
        try:
            connection = self._get_connection()
            if connection is None:
                return []
            
            cursor = connection.cursor(dictionary=True)
            start_date = datetime.now().date() - timedelta(days=days)
            
            query = """
                SELECT attendance_date, first_in_time, last_out_time, 
                       total_hours, total_visits, status
                FROM daily_attendance_summary 
                WHERE employee_id = %s AND attendance_date >= %s
                ORDER BY attendance_date DESC
            """
            cursor.execute(query, (employee_id, start_date))
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Error as e:
            logger.error(f"Error getting employee history: {e}")
            return []
    
    def get_attendance_summary(self, days: int = 7) -> pd.DataFrame:
        """Get attendance summary as DataFrame."""
        try:
            connection = self._get_connection()
            if connection is None:
                return pd.DataFrame()
            
            start_date = datetime.now().date() - timedelta(days=days)
            
            query = """
                SELECT employee_id, employee_name, attendance_date,
                       first_in_time, last_out_time, total_hours, total_visits, status
                FROM daily_attendance_summary 
                WHERE attendance_date >= %s
                ORDER BY attendance_date DESC, employee_id
            """
            
            df = pd.read_sql(query, connection, params=[start_date])
            return df
            
        except Error as e:
            logger.error(f"Error getting attendance summary: {e}")
            return pd.DataFrame()
    
    def export_attendance_report(self, filename: str = None, days: int = 30) -> bool:
        """Export attendance report to Excel."""
        try:
            if filename is None:
                filename = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            # Get data
            summary_df = self.get_attendance_summary(days)
            if summary_df.empty:
                logger.warning("No attendance data to export")
                return False
            
            connection = self._get_connection()
            if connection is None:
                return False
            
            # Get detailed logs
            start_date = datetime.now().date() - timedelta(days=days)
            detailed_query = """
                SELECT employee_id, employee_name, log_date, log_time, 
                       visit_type, confidence, visit_count
                FROM attendance_logs 
                WHERE log_date >= %s
                ORDER BY log_datetime DESC
            """
            detailed_df = pd.read_sql(detailed_query, connection, params=[start_date])
            
            # Get employee statistics
            stats_query = """
                SELECT 
                    e.employee_id,
                    e.employee_name,
                    e.department,
                    e.position,
                    COUNT(DISTINCT das.attendance_date) as days_present,
                    AVG(das.total_hours) as avg_hours_per_day,
                    SUM(das.total_visits) as total_visits,
                    MAX(das.attendance_date) as last_attendance
                FROM employees e
                LEFT JOIN daily_attendance_summary das ON e.employee_id = das.employee_id
                WHERE e.is_active = TRUE AND (das.attendance_date >= %s OR das.attendance_date IS NULL)
                GROUP BY e.employee_id, e.employee_name, e.department, e.position
                ORDER BY days_present DESC
            """
            stats_df = pd.read_sql(stats_query, connection, params=[start_date])
            
            # Export to Excel with multiple sheets
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Daily_Summary', index=False)
                detailed_df.to_excel(writer, sheet_name='Detailed_Logs', index=False)
                stats_df.to_excel(writer, sheet_name='Employee_Statistics', index=False)
                
                # Add a dashboard sheet with key metrics
                dashboard_data = {
                    'Metric': [
                        'Total Employees',
                        'Active Employees (Last 7 days)',
                        'Average Daily Attendance',
                        'Total Attendance Logs',
                        'Report Period (Days)',
                        'Report Generated'
                    ],
                    'Value': [
                        len(stats_df),
                        len(summary_df[summary_df['attendance_date'] >= (datetime.now().date() - timedelta(days=7))]['employee_id'].unique()),
                        f"{len(summary_df) / days:.1f}" if days > 0 else "0",
                        len(detailed_df),
                        days,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                dashboard_df = pd.DataFrame(dashboard_data)
                dashboard_df.to_excel(writer, sheet_name='Dashboard', index=False)
            
            logger.info(f"Attendance report exported: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting attendance report: {e}")
            return False
    
    def add_employee(self, employee_id: str, employee_name: str, image_path: str = None,
                    department: str = None, position: str = None, email: str = None, 
                    phone: str = None) -> bool:
        """Add a new employee to the system."""
        try:
            face_encoding = None
            
            # Process face image if provided
            if image_path and os.path.exists(image_path):
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) == 0:
                    logger.error(f"No face found in image: {image_path}")
                    return False
                elif len(encodings) > 1:
                    logger.warning(f"Multiple faces found in image, using the first one")
                
                face_encoding = encodings[0]
                
                # Copy image to face directory
                import shutil
                dest_path = self.face_dir / f"{employee_id}{Path(image_path).suffix}"
                shutil.copy2(image_path, dest_path)
                image_path = str(dest_path)
            
            # Save to database
            connection = self._get_connection()
            if connection is None:
                return False
            
            cursor = connection.cursor()
            
            encoding_json = json.dumps(face_encoding.tolist()) if face_encoding is not None else None
            
            query = """
                INSERT INTO employees 
                (employee_id, employee_name, face_encoding, image_path, department, position, email, phone)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                employee_name = VALUES(employee_name),
                face_encoding = VALUES(face_encoding),
                image_path = VALUES(image_path),
                department = VALUES(department),
                position = VALUES(position),
                email = VALUES(email),
                phone = VALUES(phone),
                updated_at = CURRENT_TIMESTAMP
            """
            
            cursor.execute(query, (employee_id, employee_name, encoding_json, image_path, 
                                 department, position, email, phone))
            cursor.close()
            
            # Update in-memory data
            if face_encoding is not None:
                if employee_id not in self.known_employee_ids:
                    self.known_face_encodings.append(face_encoding)
                    self.known_employee_ids.append(employee_id)
                self.employee_names[employee_id] = employee_name
            
            logger.info(f"Added employee: {employee_name} ({employee_id})")
            return True
            
        except Error as e:
            logger.error(f"Error adding employee: {e}")
            return False
    
    def remove_employee(self, employee_id: str) -> bool:
        """Remove an employee from the system."""
        try:
            connection = self._get_connection()
            if connection is None:
                return False
            
            cursor = connection.cursor()
            
            # Soft delete - mark as inactive
            query = "UPDATE employees SET is_active = FALSE WHERE employee_id = %s"
            cursor.execute(query, (employee_id,))
            
            cursor.close()
            
            # Remove from in-memory data
            if employee_id in self.known_employee_ids:
                index = self.known_employee_ids.index(employee_id)
                self.known_employee_ids.pop(index)
                self.known_face_encodings.pop(index)
                if employee_id in self.employee_names:
                    del self.employee_names[employee_id]
            
            logger.info(f"Removed employee: {employee_id}")
            return True
            
        except Error as e:
            logger.error(f"Error removing employee: {e}")
            return False
    
    def get_all_employees(self) -> List[Dict]:
        """Get list of all employees."""
        try:
            connection = self._get_connection()
            if connection is None:
                return []
            
            cursor = connection.cursor(dictionary=True)
            query = """
                SELECT employee_id, employee_name, department, position, 
                       email, phone, created_at, is_active
                FROM employees 
                ORDER BY employee_name
            """
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Error as e:
            logger.error(f"Error getting employees: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            connection = self._get_connection()
            if connection is None:
                return {}
            
            cursor = connection.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM employees WHERE is_active = TRUE")
            total_employees = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM attendance_logs")
            total_logs = cursor.fetchone()[0]
            
            # Today's stats
            today = datetime.now().date()
            cursor.execute("SELECT COUNT(DISTINCT employee_id) FROM attendance_logs WHERE log_date = %s", (today,))
            employees_today = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM attendance_logs WHERE log_date = %s", (today,))
            logs_today = cursor.fetchone()[0]
            
            # Recent activity (last 7 days)
            week_ago = today - timedelta(days=7)
            cursor.execute("SELECT COUNT(DISTINCT employee_id) FROM attendance_logs WHERE log_date >= %s", (week_ago,))
            active_employees_week = cursor.fetchone()[0]
            
            # Average daily attendance
            cursor.execute("""
                SELECT AVG(daily_count) FROM (
                    SELECT COUNT(DISTINCT employee_id) as daily_count 
                    FROM attendance_logs 
                    WHERE log_date >= %s 
                    GROUP BY log_date
                ) as daily_stats
            """, (week_ago,))
            result = cursor.fetchone()
            avg_daily_attendance = float(result[0]) if result[0] else 0.0
            
            cursor.close()
            
            stats = {
                'total_employees': total_employees,
                'total_attendance_logs': total_logs,
                'employees_present_today': employees_today,
                'attendance_logs_today': logs_today,
                'active_employees_last_week': active_employees_week,
                'average_daily_attendance': round(avg_daily_attendance, 1),
                'detection_count': self.detection_count,
                'cooldown_seconds': self.cooldown_seconds,
                'tolerance': self.tolerance,
                'face_gallery_path': str(self.face_dir),
                'database_host': self.host,
                'database_name': self.database
            }
            
            return stats
            
        except Error as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                'total_employees': len(self.known_employee_ids),
                'detection_count': self.detection_count,
                'attendance_logs': self.attendance_logs
            }
    
    def backup_database(self, backup_file: str = None) -> bool:
        """Create database backup."""
        try:
            if backup_file is None:
                backup_file = f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
            
            import subprocess
            
            # Use mysqldump to create backup
            cmd = [
                'mysqldump',
                f'--host={self.host}',
                f'--user={self.user}',
                f'--password={self.password}',
                '--routines',
                '--triggers',
                self.database
            ]
            
            with open(backup_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
            
            if result.returncode == 0:
                logger.info(f"Database backup created: {backup_file}")
                return True
            else:
                logger.error(f"Backup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def log_system_event(self, event_type: str, message: str, employee_id: str = None, 
                        additional_data: Dict = None):
        """Log system events."""
        try:
            connection = self._get_connection()
            if connection is None:
                return
            
            cursor = connection.cursor()
            
            data_json = json.dumps(additional_data) if additional_data else None
            
            query = """
                INSERT INTO system_logs (log_type, message, employee_id, additional_data)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (event_type, message, employee_id, data_json))
            cursor.close()
            
        except Error as e:
            logger.error(f"Error logging system event: {e}")
    
    def cleanup_old_logs(self, days_to_keep: int = 90) -> int:
        """Clean up old attendance logs."""
        try:
            connection = self._get_connection()
            if connection is None:
                return 0
            
            cursor = connection.cursor()
            cutoff_date = datetime.now().date() - timedelta(days=days_to_keep)
            
            # Delete old attendance logs
            cursor.execute("DELETE FROM attendance_logs WHERE log_date < %s", (cutoff_date,))
            logs_deleted = cursor.rowcount
            
            # Delete old daily summaries
            cursor.execute("DELETE FROM daily_attendance_summary WHERE attendance_date < %s", (cutoff_date,))
            summaries_deleted = cursor.rowcount
            
            # Delete old system logs
            cursor.execute("DELETE FROM system_logs WHERE created_at < %s", (cutoff_date,))
            system_logs_deleted = cursor.rowcount
            
            cursor.close()
            
            total_deleted = logs_deleted + summaries_deleted + system_logs_deleted
            logger.info(f"Cleaned up {total_deleted} old records (logs: {logs_deleted}, summaries: {summaries_deleted}, system: {system_logs_deleted})")
            
            return total_deleted
            
        except Error as e:
            logger.error(f"Error cleaning up old logs: {e}")
            return 0
    
    def close_connection(self):
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")


# Integration class for existing surveillance system
class AttendanceModule:
    """
    Wrapper class for backwards compatibility with existing surveillance system.
    """
    
    def __init__(self, db_path: str = "attendance.db"):
        """Initialize with MySQL instead of SQLite."""
        # Extract database name from path for backwards compatibility
        db_name = Path(db_path).stem if db_path != "attendance.db" else "attendance_db"
        
        self.attendance_system = MySQLAttendanceSystem(
            host="localhost",
            user="root", 
            password="root",
            database=db_name,
            face_dir="faces",
            cooldown_seconds=600,  # 10 minutes
            tolerance=0.5
        )
        
        # For compatibility
        self.last_detections = {}
    
    def process_frame(self, frame, face_detections: List = None) -> Tuple[np.ndarray, List[Dict]]:
        """Process frame for attendance detection - compatibility wrapper."""
        return self.attendance_system.process_frame(frame, face_detections)
    
    def log_attendance(self, employee_id: str, employee_name: str, confidence: float) -> Dict:
        """Log attendance - compatibility wrapper."""
        try:
            current_time = datetime.now()
            
            # Check cooldown
            if not self.attendance_system._should_log_attendance(employee_id, current_time):
                return {
                    "success": False,
                    "message": f"Cooldown active. Please wait {self.attendance_system.cooldown_seconds} seconds between logs.",
                    "employee_id": employee_id
                }
            
            # Log attendance
            visit_type, visit_count = self.attendance_system._log_attendance(
                employee_id, employee_name, current_time, confidence
            )
            
            if visit_type:
                return {
                    "success": True,
                    "message": f"Successfully logged {visit_type}",
                    "employee_id": employee_id,
                    "employee_name": employee_name,
                    "visit_type": visit_type,
                    "time": current_time.isoformat(),
                    "confidence": confidence,
                    "visit_count": visit_count
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to log attendance",
                    "employee_id": employee_id
                }
                
        except Exception as e:
            logger.error(f"Error in log_attendance wrapper: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "employee_id": employee_id
            }
    
    def get_daily_attendance(self, date: str = None) -> List[Dict]:
        """Get daily attendance - compatibility wrapper."""
        if date:
            date_obj = datetime.strptime(date, '%Y-%m-%d').date()
        else:
            date_obj = None
        
        return self.attendance_system.get_daily_attendance(date_obj)
    
    def get_employee_history(self, employee_id: str, days: int = 30) -> List[Dict]:
        """Get employee history - compatibility wrapper."""
        return self.attendance_system.get_employee_history(employee_id, days)


# Database setup script
def setup_mysql_attendance_database():
    """Setup script to create MySQL attendance database."""
    print("Setting up MySQL Attendance Database...")
    print("Please ensure MySQL server is running with user 'root' and password 'root'")
    
    try:
        # Test connection
        attendance_system = MySQLAttendanceSystem()
        stats = attendance_system.get_statistics()
        
        print(f"✅ Database setup successful!")
        print(f"📊 Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Add sample employee if none exist
        if stats['total_employees'] == 0:
            print("\n🔄 Adding sample employee...")
            success = attendance_system.add_employee(
                employee_id="EMP001",
                employee_name="John Doe",
                department="Engineering",
                position="Software Developer"
            )
            if success:
                print("✅ Sample employee added successfully")
        
        attendance_system.close_connection()
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure MySQL server is running")
        print("2. Verify username 'root' and password 'root' are correct")
        print("3. Grant necessary permissions to the user")
        return False
    
    return True


# Main execution for testing
if __name__ == "__main__":
    print("MySQL Attendance System - Setup and Test")
    
    # Setup database
    if setup_mysql_attendance_database():
        print("\n🎯 Running attendance system test...")
        
        # Initialize system
        attendance = MySQLAttendanceSystem()
        
        # Test with webcam if available
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("📹 Starting webcam test. Press 'q' to quit, 'a' to add employee, 's' for stats.")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    annotated_frame, detections = attendance.process_frame(frame)
                    
                    # Display results
                    cv2.imshow('MySQL Attendance System Test', annotated_frame)
                    
                    # Print detections
                    if detections:
                        for detection in detections:
                            print(f"👤 {detection['employee_name']} ({detection['employee_id']}) - "
                                  f"{detection.get('visit_type', 'N/A')} - Conf: {detection['confidence']:.2f}")
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        stats = attendance.get_statistics()
                        print(f"\n📊 Current Statistics:")
                        for k, v in stats.items():
                            print(f"   {k}: {v}")
                    elif key == ord('a'):
                        print("Add employee feature - please implement via web interface")
                
                cap.release()
                cv2.destroyAllWindows()
            else:
                print("📷 Webcam not available, skipping video test")
            
        except Exception as e:
            print(f"❌ Test error: {e}")
        
        finally:
            attendance.close_connection()
    
    print("✅ Test completed")