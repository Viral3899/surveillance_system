"""
Attendance tracking system for surveillance application.

This module provides:
- SQLite-based attendance logging with time tracking
- 10-minute minimum interval between visits
- Automatic IN/OUT detection
- Daily attendance summaries
- Employee attendance history
"""

from .attendance_system import AttendanceSystem, AttendanceModule, convert_attendance_to_face_detections

__version__ = "1.0.0"
__author__ = "Surveillance System"

__all__ = [
    'AttendanceSystem',
    'AttendanceModule', 
    'convert_attendance_to_face_detections'
] 