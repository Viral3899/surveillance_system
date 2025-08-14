#!/usr/bin/env python3
"""
Simplified and Fixed Employee Attendance System
==============================================

This is a streamlined version that focuses on reliability over advanced features.
It's designed to work reliably even on systems with limited resources.

Key improvements:
- Sequential image processing (no parallel processing)
- Better error handling and recovery
- Simplified database operations
- Memory-safe operations
- Clear logging and debugging
"""

import cv2
import numpy as np
import pandas as pd
import face_recognition
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pickle
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAttendanceSystem:
    """
    Simplified attendance system focused on reliability.
    
    This version trades some advanced features for stability and ease of use.
    Think of it as the "training wheels" version that helps you get started.
    """
    
    def __init__(self, 
                 face_dir: str = "faces",
                 attendance_file: str = "attendance.xlsx",
                 cooldown_minutes: int = 10):
        """
        Initialize the simple attendance system.
        
        Args:
            face_dir: Directory containing employee photos
            attendance_file: Excel file to store attendance records
            cooldown_minutes: Minutes between attendance logs for same person
        """
        self.face_dir = Path(face_dir)
        self.attendance_file = Path(attendance_file)
        self.cooldown_seconds = cooldown_minutes * 60
        
        # Face recognition data
        self.known_faces = {}  # {employee_id: face_encoding}
        self.employee_names = {}  # {employee_id: display_name}
        
        # Tracking data
        self.last_seen = {}  # {employee_id: datetime}
        self.visit_counts = {}  # {employee_id: count}
        
        # Statistics
        self.total_detections = 0
        self.successful_recognitions = 0
        
        # Initialize
        self._setup_directories()
        self._setup_attendance_file()
        self.load_employee_faces()
        
        logger.info(f"Simple Attendance System initialized with {len(self.known_faces)} employees")
    
    def _setup_directories(self):
        """Create necessary directories."""
        self.face_dir.mkdir(exist_ok=True)
        self.attendance_file.parent.mkdir(exist_ok=True)
        
        # Create backup directory
        backup_dir = Path("backup")
        backup_dir.mkdir(exist_ok=True)
        
        logger.info("Directories set up successfully")
    
    def _setup_attendance_file(self):
        """Set up the attendance Excel file."""
        if not self.attendance_file.exists():
            # Create new file with headers
            df = pd.DataFrame(columns=[
                'Employee_ID', 'Employee_Name', 'Date', 'Time',
                'Visit_Type', 'Visit_Count', 'Confidence'
            ])
            df.to_excel(self.attendance_file, index=False)
            logger.info(f"Created new attendance file: {self.attendance_file}")
        else:
            # Load existing visit counts
            try:
                df = pd.read_excel(self.attendance_file)
                if not df.empty:
                    # Get the latest visit count for each employee
                    latest_counts = df.groupby('Employee_ID')['Visit_Count'].max()
                    self.visit_counts = latest_counts.to_dict()
                    logger.info(f"Loaded existing attendance data for {len(self.visit_counts)} employees")
            except Exception as e:
                logger.warning(f"Could not load existing attendance data: {e}")
    
    def load_employee_faces(self):
        """
        Load employee faces from the face directory.
        
        This method processes images one at a time to prevent memory issues.
        It's slower but much more reliable than batch processing.
        """
        logger.info("Loading employee faces...")
        
        # Clear existing data
        self.known_faces.clear()
        self.employee_names.clear()
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.face_dir.glob(f'*{ext}'))
            image_files.extend(self.face_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            logger.warning(f"No image files found in {self.face_dir}")
            logger.info("Please add employee photos to the faces/ directory")
            return False
        
        logger.info(f"Processing {len(image_files)} image files...")
        
        successful_loads = 0
        for i, image_path in enumerate(image_files):
            try:
                logger.info(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
                
                # Extract employee ID from filename
                employee_id = image_path.stem
                
                # Load and validate image
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                # Convert to RGB (face_recognition expects RGB)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Find face encodings
                face_encodings = face_recognition.face_encodings(rgb_image)
                
                if len(face_encodings) == 0:
                    logger.warning(f"No face found in {image_path.name}")
                    continue
                elif len(face_encodings) > 1:
                    logger.info(f"Multiple faces found in {image_path.name}, using the first one")
                
                # Store the encoding
                self.known_faces[employee_id] = face_encodings[0]
                self.employee_names[employee_id] = employee_id  # Use ID as name for now
                
                successful_loads += 1
                logger.info(f"Successfully loaded: {employee_id}")
                
                # Small delay to prevent system overload
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {successful_loads} employee faces")
        return successful_loads > 0
    
    def recognize_face(self, frame):
        """
        Recognize faces in a frame and return detection results.
        
        Args:
            frame: Video frame (numpy array)
            
        Returns:
            List of detection dictionaries
        """
        self.total_detections += 1
        
        if len(self.known_faces) == 0:
            logger.debug("No known faces loaded")
            return []
        
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in the frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            detections = []
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Scale back up the face location
                top, right, bottom, left = face_location
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Compare against known faces
                matches = face_recognition.compare_faces(
                    list(self.known_faces.values()), 
                    face_encoding, 
                    tolerance=0.5
                )
                
                employee_id = "Unknown"
                confidence = 0.0
                
                if True in matches:
                    # Find the best match
                    face_distances = face_recognition.face_distance(
                        list(self.known_faces.values()), 
                        face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        employee_id = list(self.known_faces.keys())[best_match_index]
                        confidence = 1.0 - face_distances[best_match_index]
                        self.successful_recognitions += 1
                
                detection = {
                    'employee_id': employee_id,
                    'employee_name': self.employee_names.get(employee_id, employee_id),
                    'confidence': confidence,
                    'bbox': (left, top, right, bottom),
                    'timestamp': datetime.now()
                }
                
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return []
    
    def log_attendance(self, employee_id, employee_name, confidence):
        """
        Log attendance for an employee.
        
        Args:
            employee_id: Employee identifier
            employee_name: Employee display name
            confidence: Recognition confidence (0.0 to 1.0)
            
        Returns:
            Dictionary with logging result
        """
        current_time = datetime.now()
        
        # Check cooldown period
        if employee_id in self.last_seen:
            time_since_last = current_time - self.last_seen[employee_id]
            if time_since_last.total_seconds() < self.cooldown_seconds:
                return {
                    'success': False,
                    'message': f'Cooldown active ({self.cooldown_seconds//60} min)',
                    'employee_id': employee_id
                }
        
        try:
            # Determine visit type (IN/OUT alternating)
            current_count = self.visit_counts.get(employee_id, 0)
            new_count = current_count + 1
            visit_type = "IN" if new_count % 2 == 1 else "OUT"
            
            # Create attendance record
            record = {
                'Employee_ID': employee_id,
                'Employee_Name': employee_name,
                'Date': current_time.strftime('%Y-%m-%d'),
                'Time': current_time.strftime('%H:%M:%S'),
                'Visit_Type': visit_type,
                'Visit_Count': new_count,
                'Confidence': round(confidence, 3)
            }
            
            # Save to Excel file
            success = self._save_attendance_record(record)
            
            if success:
                # Update tracking data
                self.last_seen[employee_id] = current_time
                self.visit_counts[employee_id] = new_count
                
                logger.info(f"Attendance logged: {employee_name} ({employee_id}) - {visit_type} #{new_count}")
                
                return {
                    'success': True,
                    'message': f'Attendance logged: {visit_type}',
                    'employee_id': employee_id,
                    'visit_type': visit_type,
                    'visit_count': new_count
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to save to file',
                    'employee_id': employee_id
                }
                
        except Exception as e:
            logger.error(f"Error logging attendance for {employee_id}: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'employee_id': employee_id
            }
    
    def _save_attendance_record(self, record):
        """
        Save attendance record to Excel file safely.
        
        This method uses atomic operations to prevent file corruption.
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Create backup before modifying
                if attempt == 0 and self.attendance_file.exists():
                    backup_path = Path("backup") / f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    shutil.copy2(self.attendance_file, backup_path)
                
                # Load existing data
                if self.attendance_file.exists():
                    df = pd.read_excel(self.attendance_file)
                else:
                    df = pd.DataFrame(columns=list(record.keys()))
                
                # Add new record
                new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                
                # Save to temporary file first
                temp_file = self.attendance_file.with_suffix('.tmp')
                new_df.to_excel(temp_file, index=False)
                
                # Atomic rename (this prevents corruption)
                temp_file.replace(self.attendance_file)
                
                return True
                
            except Exception as e:
                logger.warning(f"Save attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("All save attempts failed")
                    return False
                time.sleep(0.5)  # Wait before retry
        
        return False
    
    def process_frame(self, frame):
        """
        Process a video frame for attendance tracking.
        
        Args:
            frame: Video frame (numpy array)
            
        Returns:
            Tuple of (annotated_frame, attendance_results)
        """
        # Recognize faces in the frame
        detections = self.recognize_face(frame)
        
        # Process each detection
        attendance_results = []
        annotated_frame = frame.copy()
        
        for detection in detections:
            employee_id = detection['employee_id']
            
            # Only log attendance for recognized employees
            if employee_id != "Unknown":
                result = self.log_attendance(
                    employee_id,
                    detection['employee_name'],
                    detection['confidence']
                )
                
                if result['success']:
                    detection['attendance_logged'] = True
                    detection['visit_type'] = result['visit_type']
                    detection['visit_count'] = result['visit_count']
                else:
                    detection['attendance_logged'] = False
                    detection['message'] = result['message']
                
                attendance_results.append(detection)
            
            # Draw detection on frame
            annotated_frame = self._draw_detection(annotated_frame, detection)
        
        return annotated_frame, attendance_results
    
    def _draw_detection(self, frame, detection):
        """Draw detection information on the frame."""
        bbox = detection['bbox']
        employee_id = detection['employee_id']
        confidence = detection['confidence']
        
        left, top, right, bottom = bbox
        
        # Choose color based on recognition
        if employee_id == "Unknown":
            color = (0, 0, 255)  # Red for unknown
            label = "Unknown"
        else:
            color = (0, 255, 0)  # Green for recognized
            visit_info = detection.get('visit_type', '')
            label = f"{employee_id} - {visit_info}"
        
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (left, top - label_size[1] - 10), (left + label_size[0], top), color, -1)
        cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw confidence
        if employee_id != "Unknown":
            conf_text = f"{confidence:.2f}"
            cv2.putText(frame, conf_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def get_statistics(self):
        """Get system statistics."""
        recognition_rate = (self.successful_recognitions / self.total_detections * 100) if self.total_detections > 0 else 0
        
        return {
            'total_employees': len(self.known_faces),
            'total_detections': self.total_detections,
            'successful_recognitions': self.successful_recognitions,
            'recognition_rate': round(recognition_rate, 2),
            'employees_today': len([emp for emp, last_time in self.last_seen.items() 
                                  if last_time.date() == datetime.now().date()]),
            'attendance_file': str(self.attendance_file),
            'face_directory': str(self.face_dir)
        }
    
    def export_attendance_report(self, output_file=None, days=30):
        """
        Export attendance report to Excel.
        
        Args:
            output_file: Output file path (auto-generated if None)
            days: Number of days to include in report
            
        Returns:
            Boolean indicating success
        """
        try:
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"attendance_report_{timestamp}.xlsx"
            
            # Load attendance data
            if not self.attendance_file.exists():
                logger.warning("No attendance data to export")
                return False
            
            df = pd.read_excel(self.attendance_file)
            
            if df.empty:
                logger.warning("Attendance file is empty")
                return False
            
            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days)
            df['Date'] = pd.to_datetime(df['Date'])
            recent_df = df[df['Date'] >= cutoff_date]
            
            # Create summary statistics
            summary_stats = {
                'Total Records': len(recent_df),
                'Unique Employees': recent_df['Employee_ID'].nunique(),
                'Date Range': f"{cutoff_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
                'Report Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Employee summary
            employee_summary = recent_df.groupby(['Employee_ID', 'Employee_Name']).agg({
                'Visit_Count': 'max',
                'Date': ['min', 'max'],
                'Visit_Type': 'count'
            }).round(2)
            
            # Export to Excel with multiple sheets
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                recent_df.to_excel(writer, sheet_name='Attendance_Log', index=False)
                
                # Summary sheet
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Employee summary
                employee_summary.to_excel(writer, sheet_name='Employee_Summary')
            
            logger.info(f"Attendance report exported: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting attendance report: {e}")
            return False
    
    def add_employee(self, employee_id, image_path, employee_name=None):
        """
        Add a new employee to the system.
        
        Args:
            employee_id: Unique employee identifier
            image_path: Path to employee photo
            employee_name: Display name (optional)
            
        Returns:
            Boolean indicating success
        """
        try:
            if employee_name is None:
                employee_name = employee_id
            
            # Validate image path
            if not Path(image_path).exists():
                logger.error(f"Image file not found: {image_path}")
                return False
            
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_image)
            
            if len(face_encodings) == 0:
                logger.error(f"No face found in image: {image_path}")
                return False
            
            # Copy image to face directory
            target_path = self.face_dir / f"{employee_id}{Path(image_path).suffix}"
            shutil.copy2(image_path, target_path)
            
            # Add to known faces
            self.known_faces[employee_id] = face_encodings[0]
            self.employee_names[employee_id] = employee_name
            
            logger.info(f"Added employee: {employee_name} ({employee_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding employee {employee_id}: {e}")
            return False
    
    def remove_employee(self, employee_id):
        """
        Remove an employee from the system.
        
        Args:
            employee_id: Employee to remove
            
        Returns:
            Boolean indicating success
        """
        try:
            if employee_id in self.known_faces:
                del self.known_faces[employee_id]
                del self.employee_names[employee_id]
                
                # Remove image file if it exists
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_path = self.face_dir / f"{employee_id}{ext}"
                    if image_path.exists():
                        image_path.unlink()
                        break
                
                logger.info(f"Removed employee: {employee_id}")
                return True
            else:
                logger.warning(f"Employee not found: {employee_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing employee {employee_id}: {e}")
            return False


def test_attendance_system():
    """Test function to verify the attendance system works."""
    print("🧪 Testing Simple Attendance System")
    print("=" * 40)
    
    try:
        # Initialize system
        attendance = SimpleAttendanceSystem(
            face_dir="faces",
            attendance_file="test_attendance.xlsx",
            cooldown_minutes=1  # Short cooldown for testing
        )
        
        print(f"✅ System initialized successfully")
        
        # Get statistics
        stats = attendance.get_statistics()
        print(f"📊 Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test with webcam if available
        print(f"\n📸 Testing camera access...")
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✅ Camera opened successfully")
            print("Press 'q' to quit, 's' for statistics, 'e' to export report")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 10th frame to avoid overwhelming the system
                if frame_count % 10 == 0:
                    annotated_frame, results = attendance.process_frame(frame)
                    
                    if results:
                        print(f"🔍 Detections: {len(results)}")
                        for result in results:
                            emp_id = result['employee_id']
                            conf = result['confidence']
                            logged = result.get('attendance_logged', False)
                            print(f"   {emp_id}: {conf:.2f} ({'✓' if logged else '✗'})")
                else:
                    annotated_frame = frame
                
                # Display frame
                cv2.imshow('Simple Attendance Test', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    stats = attendance.get_statistics()
                    print(f"\n📊 Current Statistics:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                elif key == ord('e'):
                    if attendance.export_attendance_report():
                        print("📁 Report exported successfully")
                    else:
                        print("❌ Export failed")
            
            cap.release()
            cv2.destroyAllWindows()
            
        else:
            print("⚠️  Camera not available, skipping video test")
        
        # Clean up test file
        test_file = Path("test_attendance.xlsx")
        if test_file.exists():
            test_file.unlink()
            print("🧹 Cleaned up test files")
        
        print("✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test
        test_attendance_system()
    else:
        # Show usage information
        print("Simple Employee Attendance System")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python simple_attendance.py test    # Run test mode")
        print()
        print("Integration example:")
        print("""
# Basic usage
attendance = SimpleAttendanceSystem()

# Process video frame
annotated_frame, results = attendance.process_frame(frame)

# Get statistics
stats = attendance.get_statistics()

# Export report
attendance.export_attendance_report()
""")
        print()
        print("Setup instructions:")
        print("1. Create 'faces/' directory")
        print("2. Add employee photos: EMP001.jpg, JOHN_DOE.png, etc.")
        print("3. Run: python simple_attendance.py test")
        print()
        print("Requirements:")
        print("- opencv-python")
        print("- numpy") 
        print("- pandas")
        print("- face-recognition")
        print("- openpyxl")