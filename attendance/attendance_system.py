import sqlite3
import datetime
from typing import Dict, List, Optional, Tuple
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class AttendanceSystem:
    def __init__(self, db_path: str = "attendance.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with attendance tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create attendance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT NOT NULL,
                    employee_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    time_in DATETIME,
                    time_out DATETIME,
                    total_hours REAL DEFAULT 0,
                    visit_type TEXT NOT NULL CHECK (visit_type IN ('IN', 'OUT')),
                    confidence REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(employee_id, date, visit_type)
                )
            ''')
            
            # Create daily summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT NOT NULL,
                    employee_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    first_in DATETIME,
                    last_out DATETIME,
                    total_hours REAL DEFAULT 0,
                    visit_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'PRESENT',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(employee_id, date)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Attendance database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing attendance database: {e}")
            
    def can_log_visit(self, employee_id: str, current_time: datetime.datetime) -> Tuple[bool, str]:
        """
        Check if employee can be logged (10 minute minimum difference between visits).
        Returns (can_log, visit_type)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = current_time.date()
            
            # Get last visit for this employee today
            cursor.execute('''
                SELECT time_in, time_out, visit_type, created_at
                FROM attendance 
                WHERE employee_id = ? AND date = ?
                ORDER BY created_at DESC 
                LIMIT 1
            ''', (employee_id, today))
            
            last_visit = cursor.fetchone()
            
            if not last_visit:
                # First visit today - should be IN
                conn.close()
                return True, "IN"
            
            # Parse last visit time
            last_visit_time = None
            last_visit_type = last_visit[2]
            
            if last_visit[3]:  # created_at timestamp
                last_visit_time = datetime.datetime.fromisoformat(last_visit[3].replace('Z', '+00:00'))
            
            if last_visit_time:
                time_diff = (current_time - last_visit_time).total_seconds() / 60  # minutes
                
                if time_diff < 10:  # Less than 10 minutes
                    conn.close()
                    return False, last_visit_type
            
            # Determine next visit type
            next_visit_type = "OUT" if last_visit_type == "IN" else "IN"
            
            conn.close()
            return True, next_visit_type
            
        except Exception as e:
            logger.error(f"Error checking visit eligibility: {e}")
            return False, "IN"
    
    def log_attendance(self, employee_id: str, employee_name: str, confidence: float) -> Dict:
        """
        Log employee attendance with time tracking and visit restrictions.
        """
        current_time = datetime.datetime.now()
        today = current_time.date()
        
        # Check if visit can be logged
        can_log, visit_type = self.can_log_visit(employee_id, current_time)
        
        if not can_log:
            return {
                "success": False,
                "message": f"Cannot log visit. Minimum 10 minutes required between visits.",
                "employee_id": employee_id,
                "last_visit_type": visit_type
            }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert attendance record
            if visit_type == "IN":
                cursor.execute('''
                    INSERT OR REPLACE INTO attendance 
                    (employee_id, employee_name, date, time_in, visit_type, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (employee_id, employee_name, today, current_time, visit_type, confidence))
            else:
                cursor.execute('''
                    INSERT OR REPLACE INTO attendance 
                    (employee_id, employee_name, date, time_out, visit_type, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (employee_id, employee_name, today, current_time, visit_type, confidence))
            
            # Update daily summary
            self._update_daily_summary(cursor, employee_id, employee_name, today, current_time, visit_type)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Logged {visit_type} for {employee_name} ({employee_id}) at {current_time}")
            
            return {
                "success": True,
                "message": f"Successfully logged {visit_type}",
                "employee_id": employee_id,
                "employee_name": employee_name,
                "visit_type": visit_type,
                "time": current_time.isoformat(),
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error logging attendance: {e}")
            return {
                "success": False,
                "message": f"Database error: {str(e)}",
                "employee_id": employee_id
            }
    
    def _update_daily_summary(self, cursor, employee_id: str, employee_name: str, 
                            date, current_time: datetime.datetime, visit_type: str):
        """Update the daily attendance summary."""
        try:
            # Get existing daily record
            cursor.execute('''
                SELECT first_in, last_out, visit_count, total_hours
                FROM daily_attendance 
                WHERE employee_id = ? AND date = ?
            ''', (employee_id, date))
            
            existing = cursor.fetchone()
            
            if existing:
                first_in, last_out, visit_count, total_hours = existing
                
                if visit_type == "IN":
                    if not first_in:
                        first_in = current_time
                    visit_count += 1
                else:  # OUT
                    last_out = current_time
                    
                    # Calculate total hours if we have both in and out times
                    if first_in:
                        first_in_dt = datetime.datetime.fromisoformat(first_in) if isinstance(first_in, str) else first_in
                        total_hours = (current_time - first_in_dt).total_seconds() / 3600
                
                cursor.execute('''
                    UPDATE daily_attendance 
                    SET first_in = ?, last_out = ?, visit_count = ?, 
                        total_hours = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE employee_id = ? AND date = ?
                ''', (first_in, last_out, visit_count, total_hours, employee_id, date))
                
            else:
                # Create new daily record
                if visit_type == "IN":
                    cursor.execute('''
                        INSERT INTO daily_attendance 
                        (employee_id, employee_name, date, first_in, visit_count)
                        VALUES (?, ?, ?, ?, 1)
                    ''', (employee_id, employee_name, date, current_time))
                else:
                    cursor.execute('''
                        INSERT INTO daily_attendance 
                        (employee_id, employee_name, date, last_out, visit_count)
                        VALUES (?, ?, ?, ?, 1)
                    ''', (employee_id, employee_name, date, current_time))
                    
        except Exception as e:
            logger.error(f"Error updating daily summary: {e}")
    
    def get_daily_attendance(self, date: Optional[str] = None) -> List[Dict]:
        """Get daily attendance summary."""
        if not date:
            date = datetime.date.today()
        elif isinstance(date, str):
            date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT employee_id, employee_name, first_in, last_out, 
                       total_hours, visit_count, status
                FROM daily_attendance 
                WHERE date = ?
                ORDER BY first_in
            ''', (date,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "employee_id": row[0],
                    "employee_name": row[1],
                    "first_in": row[2],
                    "last_out": row[3],
                    "total_hours": round(row[4], 2) if row[4] else 0,
                    "visit_count": row[5],
                    "status": row[6]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting daily attendance: {e}")
            return []
    
    def get_employee_history(self, employee_id: str, days: int = 30) -> List[Dict]:
        """Get attendance history for specific employee."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = datetime.date.today() - datetime.timedelta(days=days)
            
            cursor.execute('''
                SELECT date, first_in, last_out, total_hours, visit_count, status
                FROM daily_attendance 
                WHERE employee_id = ? AND date >= ?
                ORDER BY date DESC
            ''', (employee_id, start_date))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "date": row[0],
                    "first_in": row[1],
                    "last_out": row[2],
                    "total_hours": round(row[3], 2) if row[3] else 0,
                    "visit_count": row[4],
                    "status": row[5]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting employee history: {e}")
            return []


class AttendanceModule:
    """Integration module for surveillance system."""
    
    def __init__(self, db_path: str = "attendance.db"):
        self.attendance_system = AttendanceSystem(db_path)
        self.last_detections = {}  # Cache to prevent rapid duplicate detections
        
    def process_frame(self, frame, face_detections: List = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process frame for attendance detection.
        Args:
            frame: Current video frame
            face_detections: List of face detection objects with face_id, name, confidence
        Returns:
            Tuple of (annotated_frame, attendance_detections)
        """
        if face_detections is None:
            face_detections = []
            
        current_time = datetime.datetime.now()
        attendance_detections = []
        annotated_frame = frame.copy()
        
        for detection in face_detections:
            try:
                # Extract detection information
                employee_id = getattr(detection, 'face_id', None) or getattr(detection, 'id', 'Unknown')
                employee_name = getattr(detection, 'name', employee_id)
                confidence = getattr(detection, 'confidence', 0.0)
                
                # Skip unknown faces
                if employee_id == "Unknown" or employee_id is None:
                    continue
                
                # Log attendance
                result = self.attendance_system.log_attendance(
                    employee_id, employee_name, confidence
                )
                
                if result['success']:
                    attendance_detections.append({
                        'employee_id': employee_id,
                        'employee_name': employee_name,
                        'visit_type': result['visit_type'],
                        'time': result['time'],
                        'confidence': confidence,
                        'visit_count': 1  # For compatibility with existing code
                    })
                    
                    # Draw attendance annotation on frame
                    self._draw_attendance_annotation(
                        annotated_frame, detection, result['visit_type'], employee_name
                    )
                
            except Exception as e:
                logger.error(f"Error processing attendance detection: {e}")
        
        return annotated_frame, attendance_detections
    
    def _draw_attendance_annotation(self, frame, detection, visit_type, employee_name):
        """Draw attendance information on the frame."""
        try:
            # Get bounding box if available
            if hasattr(detection, 'box'):
                x, y, w, h = detection.box
            elif hasattr(detection, 'bbox'):
                x, y, w, h = detection.bbox
            else:
                # Default position if no bounding box
                x, y, w, h = 50, 50, 200, 30
            
            # Colors for IN/OUT
            color = (0, 255, 0) if visit_type == "IN" else (0, 165, 255)  # Green for IN, Orange for OUT
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw attendance info
            text = f"{employee_name} - {visit_type}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
            
            # Draw text
            cv2.putText(frame, text, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
            
            # Draw timestamp
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (x, y + h + 20), font, 0.4, color, 1)
            
        except Exception as e:
            logger.error(f"Error drawing attendance annotation: {e}")


# Utility functions for integration
def convert_attendance_to_face_detections(attendance_detections: List[Dict]) -> List:
    """Convert attendance detections to face detection format for anomaly detector."""
    face_detections = []
    
    for detection in attendance_detections:
        # Create a simple object with required attributes
        class FaceDetection:
            def __init__(self, employee_id, confidence):
                self.face_id = employee_id
                self.confidence = confidence
                self.name = detection.get('employee_name', employee_id)
        
        face_detections.append(FaceDetection(
            detection['employee_id'], 
            detection['confidence']
        ))
    
    return face_detections


# Example usage and testing
if __name__ == "__main__":
    # Initialize attendance system
    attendance = AttendanceSystem("test_attendance.db")
    
    # Test logging
    print("Testing attendance system...")
    
    # First visit (should be IN)
    result1 = attendance.log_attendance("EMP001", "John Doe", 0.95)
    print(f"Visit 1: {result1}")
    
    # Try immediate second visit (should fail - less than 10 minutes)
    result2 = attendance.log_attendance("EMP001", "John Doe", 0.92)
    print(f"Visit 2 (immediate): {result2}")
    
    # Get daily attendance
    daily = attendance.get_daily_attendance()
    print(f"Daily attendance: {daily}")
    
    # Test attendance module
    module = AttendanceModule("test_attendance.db")
    
    # Simulate frame processing
    import numpy as np
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    class MockDetection:
        def __init__(self):
            self.face_id = "EMP002"
            self.name = "Jane Smith"
            self.confidence = 0.89
            self.box = (100, 100, 150, 200)
    
    mock_detections = [MockDetection()]
    annotated_frame, detections = module.process_frame(test_frame)
    print(f"Processed detections: {detections}") 