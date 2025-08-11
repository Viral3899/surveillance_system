"""
Employee Attendance & Time Log Module for AI Surveillance System
================================================================

This module provides face-based attendance tracking with Excel logging,
cooldown logic, and real-time integration with the surveillance system.

Features:
- Face gallery loading and encoding
- Real-time face detection and matching
- Cooldown logic to prevent duplicate entries
- Excel attendance logging with visit counting
- Offline-ready operation
- Integration with existing surveillance system

Author: AI Surveillance System
Version: 1.0
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmployeeAttendanceModule:
    """
    Employee Attendance Module for face-based attendance tracking.
    
    This class handles face gallery loading, real-time detection,
    cooldown logic, and Excel logging for employee attendance.
    """
    
    def __init__(self, 
                 face_dir: str = "faces",
                 attendance_file: str = "attendance.xlsx",
                 cooldown_seconds: int = 5,
                 tolerance: float = 0.5,
                 encodings_cache: str = "face_encodings.pkl"):
        """
        Initialize the Employee Attendance Module.
        
        Args:
            face_dir (str): Directory containing employee face images
            attendance_file (str): Excel file path for attendance logging
            cooldown_seconds (int): Cooldown period to prevent duplicate entries
            tolerance (float): Face recognition tolerance (lower = stricter)
            encodings_cache (str): Cache file for face encodings
        """
        self.face_dir = Path(face_dir)
        self.attendance_file = Path(attendance_file)
        self.cooldown_seconds = cooldown_seconds
        self.tolerance = tolerance
        self.encodings_cache = Path(encodings_cache)
        
        # Face recognition data
        self.known_face_encodings: List[np.ndarray] = []
        self.known_employee_ids: List[str] = []
        
        # Attendance tracking
        self.last_seen_time: Dict[str, datetime] = {}
        self.visit_counts: Dict[str, int] = {}
        
        # Performance tracking
        self.detection_count = 0
        self.attendance_logs = 0
        
        # Initialize the module
        self._create_directories()
        self._initialize_attendance_file()
        self.load_known_faces()
        
        logger.info(f"Employee Attendance Module initialized with {len(self.known_employee_ids)} employees")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        self.face_dir.mkdir(exist_ok=True)
        self.attendance_file.parent.mkdir(exist_ok=True)
        
        # Create sample face directory structure if empty
        if not any(self.face_dir.iterdir()):
            logger.warning(f"Face directory '{self.face_dir}' is empty. Please add employee face images.")
            logger.info("Expected format: faces/EMP001.jpg, faces/EMP002.png, etc.")
    
    def _initialize_attendance_file(self):
        """Initialize the Excel attendance file with proper headers."""
        if not self.attendance_file.exists():
            # Create new attendance file with headers
            df = pd.DataFrame(columns=['Employee_ID', 'Timestamp', 'Total_Visits'])
            df.to_excel(self.attendance_file, index=False, engine='openpyxl')
            logger.info(f"Created new attendance file: {self.attendance_file}")
        else:
            # Load existing visit counts
            try:
                df = pd.read_excel(self.attendance_file, engine='openpyxl')
                if not df.empty:
                    # Calculate current visit counts for each employee
                    visit_counts = df.groupby('Employee_ID').size().to_dict()
                    self.visit_counts.update(visit_counts)
                    logger.info(f"Loaded existing attendance data for {len(visit_counts)} employees")
            except Exception as e:
                logger.error(f"Error reading existing attendance file: {e}")
    
    def load_known_faces(self) -> bool:
        """
        Load and encode all faces from the face gallery.
        
        Returns:
            bool: True if faces were loaded successfully, False otherwise
        """
        # Try to load from cache first
        if self._load_encodings_cache():
            return True
        
        # Load from face images
        return self._load_faces_from_images()
    
    def _load_encodings_cache(self) -> bool:
        """Load face encodings from cache file."""
        if not self.encodings_cache.exists():
            return False
        
        try:
            with open(self.encodings_cache, 'rb') as f:
                cache_data = pickle.load(f)
                self.known_face_encodings = cache_data['encodings']
                self.known_employee_ids = cache_data['employee_ids']
                logger.info(f"Loaded {len(self.known_employee_ids)} face encodings from cache")
                return True
        except Exception as e:
            logger.warning(f"Failed to load encodings cache: {e}")
            return False
    
    def _save_encodings_cache(self):
        """Save face encodings to cache file for faster loading."""
        try:
            cache_data = {
                'encodings': self.known_face_encodings,
                'employee_ids': self.known_employee_ids,
                'created_time': datetime.now().isoformat()
            }
            with open(self.encodings_cache, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved face encodings cache: {self.encodings_cache}")
        except Exception as e:
            logger.error(f"Failed to save encodings cache: {e}")
    
    def _load_faces_from_images(self) -> bool:
        """Load and encode faces from image files."""
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        loaded_count = 0
        
        logger.info(f"Loading face images from: {self.face_dir}")
        
        for image_path in self.face_dir.iterdir():
            if image_path.suffix.lower() not in supported_formats:
                continue
            
            # Extract employee ID from filename
            employee_id = image_path.stem
            
            try:
                # Load and encode the face
                image = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) == 0:
                    logger.warning(f"No face found in {image_path}")
                    continue
                elif len(encodings) > 1:
                    logger.warning(f"Multiple faces found in {image_path}, using the first one")
                
                # Store the first face encoding
                self.known_face_encodings.append(encodings[0])
                self.known_employee_ids.append(employee_id)
                loaded_count += 1
                
                logger.debug(f"Loaded face for employee: {employee_id}")
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
        
        if loaded_count > 0:
            self._save_encodings_cache()
            logger.info(f"Successfully loaded {loaded_count} employee faces")
            return True
        else:
            logger.error("No faces were loaded. Please check your face gallery.")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a video frame for face detection and attendance logging.
        
        Args:
            frame (np.ndarray): Input video frame
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: Annotated frame and detection results
        """
        if len(self.known_face_encodings) == 0:
            return frame, []
        
        self.detection_count += 1
        current_time = datetime.now()
        
        # Resize frame for faster processing (optional optimization)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        detection_results = []
        annotated_frame = frame.copy()
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations since frame was scaled down
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Match against known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=self.tolerance
            )
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            
            employee_id = "Unknown"
            confidence = 0.0
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    employee_id = self.known_employee_ids[best_match_index]
                    confidence = 1.0 - face_distances[best_match_index]
                    
                    # Check cooldown and log attendance
                    if self._should_log_attendance(employee_id, current_time):
                        self.save_attendance(employee_id, current_time)
            
            # Prepare detection result
            detection_result = {
                'employee_id': employee_id,
                'confidence': confidence,
                'bbox': (left, top, right, bottom),
                'visit_count': self.visit_counts.get(employee_id, 0)
            }
            detection_results.append(detection_result)
            
            # Draw bounding box and labels
            annotated_frame = self._draw_detection(annotated_frame, detection_result)
        
        return annotated_frame, detection_results
    
    def _should_log_attendance(self, employee_id: str, current_time: datetime) -> bool:
        """
        Check if attendance should be logged based on cooldown logic.
        
        Args:
            employee_id (str): Employee identifier
            current_time (datetime): Current timestamp
            
        Returns:
            bool: True if attendance should be logged, False otherwise
        """
        if employee_id not in self.last_seen_time:
            return True
        
        time_since_last_seen = current_time - self.last_seen_time[employee_id]
        return time_since_last_seen.total_seconds() >= self.cooldown_seconds
    
    def save_attendance(self, employee_id: str, timestamp: datetime) -> bool:
        """
        Save attendance record to Excel file.
        
        Args:
            employee_id (str): Employee identifier
            timestamp (datetime): Attendance timestamp
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Update tracking data
            self.last_seen_time[employee_id] = timestamp
            self.visit_counts[employee_id] = self.visit_counts.get(employee_id, 0) + 1
            self.attendance_logs += 1
            
            # Prepare new record
            new_record = {
                'Employee_ID': employee_id,
                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Total_Visits': self.visit_counts[employee_id]
            }
            
            # Load existing data
            try:
                df = pd.read_excel(self.attendance_file, engine='openpyxl')
            except FileNotFoundError:
                df = pd.DataFrame(columns=['Employee_ID', 'Timestamp', 'Total_Visits'])
            
            # Add new record
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            
            # Save to Excel
            df.to_excel(self.attendance_file, index=False, engine='openpyxl')
            
            logger.info(f"Logged attendance: {employee_id} (Visit #{self.visit_counts[employee_id]})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving attendance for {employee_id}: {e}")
            return False
    
    def _draw_detection(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """
        Draw detection bounding box and information on frame.
        
        Args:
            frame (np.ndarray): Input frame
            detection (Dict): Detection information
            
        Returns:
            np.ndarray: Annotated frame
        """
        left, top, right, bottom = detection['bbox']
        employee_id = detection['employee_id']
        confidence = detection['confidence']
        visit_count = detection['visit_count']
        
        # Choose color based on recognition status
        if employee_id == "Unknown":
            color = (0, 0, 255)  # Red for unknown
            label = "Unknown"
        else:
            color = (0, 255, 0)  # Green for recognized
            label = f"{employee_id} (#{visit_count})"
        
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (left, top - label_size[1] - 10), 
                     (left + label_size[0], top), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (left, top - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw confidence if recognized
        if employee_id != "Unknown":
            conf_text = f"Conf: {confidence:.2f}"
            cv2.putText(frame, conf_text, (left, bottom + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        return frame
    
    def get_attendance_summary(self, days: int = 7) -> pd.DataFrame:
        """
        Get attendance summary for the last N days.
        
        Args:
            days (int): Number of days to include in summary
            
        Returns:
            pd.DataFrame: Attendance summary
        """
        try:
            df = pd.read_excel(self.attendance_file, engine='openpyxl')
            if df.empty:
                return pd.DataFrame()
            
            # Convert timestamp and filter by date range
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_df = df[df['Timestamp'] >= cutoff_date]
            
            # Create summary
            summary = recent_df.groupby('Employee_ID').agg({
                'Timestamp': ['count', 'min', 'max'],
                'Total_Visits': 'max'
            }).reset_index()
            
            # Flatten column names
            summary.columns = ['Employee_ID', 'Daily_Visits', 'First_Seen', 'Last_Seen', 'Total_Visits']
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating attendance summary: {e}")
            return pd.DataFrame()
    
    def export_attendance_report(self, output_file: str = None, days: int = 30) -> bool:
        """
        Export detailed attendance report.
        
        Args:
            output_file (str): Output file path (optional)
            days (int): Number of days to include
            
        Returns:
            bool: True if exported successfully
        """
        try:
            if output_file is None:
                output_file = f"attendance_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
            
            # Get full attendance data
            df = pd.read_excel(self.attendance_file, engine='openpyxl')
            if df.empty:
                logger.warning("No attendance data to export")
                return False
            
            # Filter by date range
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_df = df[df['Timestamp'] >= cutoff_date]
            
            # Create summary sheet
            summary = self.get_attendance_summary(days)
            
            # Export to Excel with multiple sheets
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                recent_df.to_excel(writer, sheet_name='Detailed_Log', index=False)
                summary.to_excel(writer, sheet_name='Summary', index=False)
            
            logger.info(f"Attendance report exported: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting attendance report: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        Get module statistics and performance metrics.
        
        Returns:
            Dict: Statistics dictionary
        """
        return {
            'total_employees': len(self.known_employee_ids),
            'total_detections': self.detection_count,
            'total_attendance_logs': self.attendance_logs,
            'active_employees_today': len([
                emp_id for emp_id, last_seen in self.last_seen_time.items()
                if (datetime.now() - last_seen).days == 0
            ]),
            'face_gallery_path': str(self.face_dir),
            'attendance_file_path': str(self.attendance_file),
            'cooldown_seconds': self.cooldown_seconds,
            'tolerance': self.tolerance
        }
    
    def reset_daily_counts(self):
        """Reset daily tracking data (useful for testing or daily resets)."""
        self.last_seen_time.clear()
        self.detection_count = 0
        logger.info("Daily tracking data reset")
    
    def add_new_employee(self, employee_id: str, image_path: str) -> bool:
        """
        Add a new employee to the face gallery.
        
        Args:
            employee_id (str): Employee identifier
            image_path (str): Path to employee's face image
            
        Returns:
            bool: True if added successfully
        """
        try:
            # Load and validate the image
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                logger.error(f"No face found in image: {image_path}")
                return False
            elif len(encodings) > 1:
                logger.warning(f"Multiple faces found in image, using the first one")
            
            # Copy image to face gallery
            import shutil
            dest_path = self.face_dir / f"{employee_id}{Path(image_path).suffix}"
            shutil.copy2(image_path, dest_path)
            
            # Add to known faces
            self.known_face_encodings.append(encodings[0])
            self.known_employee_ids.append(employee_id)
            
            # Update cache
            self._save_encodings_cache()
            
            logger.info(f"Added new employee: {employee_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding employee {employee_id}: {e}")
            return False


# Convenience functions for easy integration
def create_attendance_module(**kwargs) -> EmployeeAttendanceModule:
    """Create and return an EmployeeAttendanceModule instance."""
    return EmployeeAttendanceModule(**kwargs)


def process_frame_with_attendance(module: EmployeeAttendanceModule, 
                                frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """Process frame with attendance tracking."""
    return module.process_frame(frame)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("Employee Attendance Module - Test Mode")
    
    # Initialize module
    attendance = EmployeeAttendanceModule(
        face_dir="faces",
        attendance_file="attendance.xlsx",
        cooldown_seconds=5,
        tolerance=0.5
    )
    
    # Print statistics
    stats = attendance.get_statistics()
    print("\nModule Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test with webcam (if available)
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("\nStarting webcam test. Press 'q' to quit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, detections = attendance.process_frame(frame)
                
                # Display results
                cv2.imshow('Employee Attendance Test', annotated_frame)
                
                # Print detections
                if detections:
                    for detection in detections:
                        print(f"Detected: {detection['employee_id']} "
                              f"(Visits: {detection['visit_count']})")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("Webcam not available for testing")
            
    except Exception as e:
        print(f"Error during webcam test: {e}")
    
    print("\nTest completed.")  