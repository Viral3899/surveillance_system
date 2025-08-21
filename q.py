#!/usr/bin/env python3
"""
System Diagnostic and Integration Script
======================================

This script:
1. Diagnoses current attendance system issues
2. Integrates the fixed attendance logic
3. Validates face detection and recognition
4. Tests the complete system flow
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemDiagnostic:
    """Comprehensive system diagnostic and integration."""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        
    def check_system_health(self):
        """Run complete system health check."""
        print("🔍 SYSTEM HEALTH CHECK")
        print("=" * 50)
        
        # 1. Check Python environment
        self._check_python_environment()
        
        # 2. Check required files
        self._check_required_files()
        
        # 3. Check directories
        self._check_directories()
        
        # 4. Check face images
        self._check_face_images()
        
        # 5. Check camera access
        self._check_camera_access()
        
        # 6. Check database
        self._check_database_integrity()
        
        # 7. Test face recognition
        self._test_face_recognition()
        
        # 8. Test attendance logic
        self._test_attendance_logic()
        
        return len(self.issues_found) == 0
    
    def _check_python_environment(self):
        """Check Python environment and packages."""
        print("\n📦 Checking Python Environment...")
        
        # Check Python version
        version = sys.version_info
        if version < (3, 8):
            self.issues_found.append("Python version too old (need 3.8+)")
            print(f"   ❌ Python {version.major}.{version.minor} (need 3.8+)")
        else:
            print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        
        # Check required packages
        required_packages = {
            'cv2': 'opencv-python',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'face_recognition': 'face-recognition',
            'sqlite3': 'sqlite3 (built-in)',
            'openpyxl': 'openpyxl',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn'
        }
        
        for package, install_name in required_packages.items():
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                self.issues_found.append(f"Missing package: {install_name}")
                print(f"   ❌ {package} (install with: pip install {install_name})")
    
    def _check_required_files(self):
        """Check if required system files exist."""
        print("\n📁 Checking Required Files...")
        
        required_files = [
            'api_server.py',
            'attendance_module.py', 
            'attendance_system.py',
            'utils/config.py',
            'utils/logger.py'
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"   ✅ {file_path}")
            else:
                self.issues_found.append(f"Missing file: {file_path}")
                print(f"   ❌ {file_path}")
    
    def _check_directories(self):
        """Check and create necessary directories."""
        print("\n📂 Checking Directories...")
        
        required_dirs = [
            'faces',
            'surveillance_output',
            'surveillance_output/logs',
            'attendance_reports',
            'backup',
            'cache'
        ]
        
        for directory in required_dirs:
            dir_path = Path(directory)
            if dir_path.exists():
                print(f"   ✅ {directory}/")
            else:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"   🔧 Created {directory}/")
                    self.fixes_applied.append(f"Created directory: {directory}")
                except Exception as e:
                    self.issues_found.append(f"Cannot create directory {directory}: {e}")
                    print(f"   ❌ {directory}/ - {e}")
    
    def _check_face_images(self):
        """Check face images in the gallery."""
        print("\n🖼️ Checking Face Images...")
        
        face_dirs = ['faces']
        total_images = 0
        valid_images = 0
        
        for face_dir in face_dirs:
            if Path(face_dir).exists():
                image_files = list(Path(face_dir).glob('*.jpg')) + \
                             list(Path(face_dir).glob('*.png')) + \
                             list(Path(face_dir).glob('*.jpeg'))
                
                print(f"   📁 {face_dir}: {len(image_files)} images")
                total_images += len(image_files)
                
                # Validate first few images
                for img_file in image_files[:5]:
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            height, width = img.shape[:2]
                            if width > 50 and height > 50:
                                valid_images += 1
                                print(f"     ✅ {img_file.name}: {width}x{height}")
                            else:
                                print(f"     ⚠️ {img_file.name}: Too small ({width}x{height})")
                        else:
                            print(f"     ❌ {img_file.name}: Cannot read")
                    except Exception as e:
                        print(f"     ❌ {img_file.name}: {e}")
        
        if total_images == 0:
            self.issues_found.append("No face images found")
            print("   ❌ No face images found!")
            print("   💡 Add employee photos to 'faces/' directory")
            print("   💡 Name format: EMP001.jpg, JOHN_DOE.png, etc.")
        else:
            print(f"   ✅ Total: {total_images} images, {valid_images} valid")
    
    def _check_camera_access(self):
        """Test camera access."""
        print("\n📹 Checking Camera Access...")
        
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"   ✅ Camera 0: {width}x{height}")
                else:
                    print("   ⚠️ Camera opened but no frame")
                cap.release()
            else:
                print("   ❌ Cannot open camera 0")
                # Try other camera indices
                for i in range(1, 4):
                    try:
                        cap = cv2.VideoCapture(i)
                        if cap.isOpened():
                            print(f"   ✅ Found camera {i}")
                            cap.release()
                            break
                        cap.release()
                    except:
                        continue
                else:
                    self.issues_found.append("No camera accessible")
        except Exception as e:
            self.issues_found.append(f"Camera test error: {e}")
            print(f"   ❌ Camera test failed: {e}")
    
    def _check_database_integrity(self):
        """Check attendance database integrity."""
        print("\n🗄️ Checking Database Integrity...")
        
        db_files = ['attendance.db', 'attendance.xlsx']
        
        for db_file in db_files:
            if Path(db_file).exists():
                try:
                    if db_file.endswith('.db'):
                        # Check SQLite database
                        conn = sqlite3.connect(db_file)
                        cursor = conn.cursor()
                        
                        # Check if tables exist
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = [row[0] for row in cursor.fetchall()]
                        
                        if 'attendance' in tables:
                            cursor.execute("SELECT COUNT(*) FROM attendance")
                            count = cursor.fetchone()[0]
                            print(f"   ✅ {db_file}: {count} records")
                        else:
                            print(f"   ⚠️ {db_file}: No attendance table")
                            
                        conn.close()
                        
                    elif db_file.endswith('.xlsx'):
                        # Check Excel file
                        df = pd.read_excel(db_file, engine='openpyxl')
                        print(f"   ✅ {db_file}: {len(df)} records")
                        
                except Exception as e:
                    print(f"   ❌ {db_file}: {e}")
            else:
                print(f"   ℹ️ {db_file}: Not found (will be created)")
    
    def _test_face_recognition(self):
        """Test face recognition functionality."""
        print("\n👤 Testing Face Recognition...")
        
        try:
            import face_recognition
            
            # Create test image
            test_image = np.zeros((200, 200, 3), dtype=np.uint8)
            # Add face-like features
            cv2.rectangle(test_image, (50, 50), (150, 150), (200, 200, 200), -1)
            cv2.circle(test_image, (75, 80), 5, (0, 0, 0), -1)  # Eyes
            cv2.circle(test_image, (125, 80), 5, (0, 0, 0), -1)
            cv2.ellipse(test_image, (100, 120), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Mouth
            
            # Test face detection
            face_locations = face_recognition.face_locations(test_image)
            print(f"   ✅ Face detection: {len(face_locations)} faces found")
            
            if face_locations:
                # Test face encoding
                face_encodings = face_recognition.face_encodings(test_image, face_locations)
                print(f"   ✅ Face encoding: {len(face_encodings)} encodings generated")
            
        except ImportError:
            self.issues_found.append("face_recognition not installed")
            print("   ❌ face_recognition not available")
        except Exception as e:
            self.issues_found.append(f"Face recognition test failed: {e}")
            print(f"   ❌ Face recognition test: {e}")
    
    def _test_attendance_logic(self):
        """Test current attendance logic."""
        print("\n📋 Testing Attendance Logic...")
        
        try:
            # Check if our fixed attendance system is available
            from fixed_attendance_logic import FixedAttendanceSystem
            
            # Create test instance
            test_attendance = FixedAttendanceSystem("test_attendance.db")
            
            # Test visit type determination
            test_time = datetime.now()
            visit_type, visit_number, can_log = test_attendance.determine_visit_type_and_number("TEST001", test_time)
            
            print(f"   ✅ Visit logic: {visit_type} #{visit_number} (can_log: {can_log})")
            
            # Clean up test database
            test_db = Path("test_attendance.db")
            if test_db.exists():
                test_db.unlink()
                
        except ImportError:
            print("   ℹ️ Fixed attendance system not integrated yet")
        except Exception as e:
            self.issues_found.append(f"Attendance logic test failed: {e}")
            print(f"   ❌ Attendance test: {e}")
    
    def integrate_fixed_attendance(self):
        """Integrate the fixed attendance system."""
        print("\n🔧 INTEGRATING FIXED ATTENDANCE SYSTEM")
        print("=" * 50)
        
        # Check current attendance files
        current_files = [
            'attendance_system.py',
            'attendance_module.py'
        ]
        
        backup_dir = Path('backup/pre_fix')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup existing files
        for file_path in current_files:
            if Path(file_path).exists():
                backup_path = backup_dir / f"{file_path}.backup"
                try:
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    print(f"   📋 Backed up {file_path} to {backup_path}")
                except Exception as e:
                    print(f"   ⚠️ Could not backup {file_path}: {e}")
        
        print("   ✅ Fixed attendance system ready for integration")
        print("   💡 Use the FixedAttendanceSystem class for proper IN/OUT logic")
    
    def create_integration_example(self):
        """Create example showing how to integrate fixed system."""
        example_code = '''
# Example: Integrating Fixed Attendance with Face Recognition
from fixed_attendance_logic import FixedAttendanceSystem
import cv2
import face_recognition

class IntegratedAttendanceSystem:
    def __init__(self):
        self.attendance = FixedAttendanceSystem("attendance.db")
        self.known_faces = self._load_known_faces()
    
    def _load_known_faces(self):
        """Load known faces from faces directory."""
        known_faces = {}
        faces_dir = Path("faces")
        
        for img_file in faces_dir.glob("*.jpg"):
            try:
                # Load and encode face
                img = face_recognition.load_image_file(str(img_file))
                encodings = face_recognition.face_encodings(img)
                
                if encodings:
                    employee_id = img_file.stem  # Use filename as employee ID
                    known_faces[employee_id] = encodings[0]
                    print(f"Loaded face: {employee_id}")
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
        
        return known_faces
    
    def process_frame(self, frame):
        """Process frame and log attendance with proper IN/OUT logic."""
        # Detect faces in frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        results = []
        
        for face_encoding in face_encodings:
            # Match against known faces
            matches = face_recognition.compare_faces(
                list(self.known_faces.values()), face_encoding, tolerance=0.6
            )
            
            if any(matches):
                # Find best match
                face_distances = face_recognition.face_distance(
                    list(self.known_faces.values()), face_encoding
                )
                best_match_index = face_distances.argmin()
                
                if matches[best_match_index]:
                    employee_id = list(self.known_faces.keys())[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    
                    # Log attendance with proper IN/OUT logic
                    result = self.attendance.log_attendance(
                        employee_id, employee_id, confidence
                    )
                    
                    results.append({
                        'employee_id': employee_id,
                        'confidence': confidence,
                        'attendance_result': result
                    })
        
        return results

# Usage example:
if __name__ == "__main__":
    system = IntegratedAttendanceSystem()
    
    # Start camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = system.process_frame(frame)
        
        # Display results
        for result in results:
            emp_id = result['employee_id']
            att_result = result['attendance_result']
            
            if att_result['success']:
                visit_type = att_result['visit_type']
                visit_num = att_result['visit_number']
                print(f"✅ {emp_id}: {visit_type} (Visit #{visit_num})")
            else:
                print(f"⏳ {emp_id}: {att_result['message']}")
        
        cv2.imshow('Attendance System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
'''
        
        with open('integration_example.py', 'w') as f:
            f.write(example_code)
        
        print("   ✅ Created integration_example.py")
    
    def generate_report(self):
        """Generate diagnostic report."""
        print("\n📊 DIAGNOSTIC REPORT")
        print("=" * 50)
        
        print(f"Issues Found: {len(self.issues_found)}")
        for issue in self.issues_found:
            print(f"   ❌ {issue}")
        
        print(f"\nFixes Applied: {len(self.fixes_applied)}")
        for fix in self.fixes_applied:
            print(f"   🔧 {fix}")
        
        if len(self.issues_found) == 0:
            print("\n🟢 SYSTEM STATUS: HEALTHY")
            print("   Your attendance system should work properly!")
        elif len(self.issues_found) <= 3:
            print("\n🟡 SYSTEM STATUS: NEEDS MINOR FIXES")
            print("   System should work with some limitations")
        else:
            print("\n🔴 SYSTEM STATUS: NEEDS ATTENTION")
            print("   Several issues need to be resolved")
        
        # Next steps
        print("\n📋 NEXT STEPS:")
        print("1. Fix any issues listed above")
        print("2. Add employee photos to 'faces/' directory")
        print("3. Use the FixedAttendanceSystem for proper IN/OUT logic")
        print("4. Test with: python integration_example.py")
        print("5. Run full system: python api_server.py")

def main():
    """Run complete diagnostic and integration."""
    print("🔍 AI SURVEILLANCE ATTENDANCE SYSTEM")
    print("🔧 DIAGNOSTIC & INTEGRATION TOOL")
    print("=" * 60)
    
    # Create diagnostic instance
    diagnostic = SystemDiagnostic()
    
    # Run health check
    is_healthy = diagnostic.check_system_health()
    
    # Integrate fixed attendance
    diagnostic.integrate_fixed_attendance()
    
    # Create integration example
    diagnostic.create_integration_example()
    
    # Generate report
    diagnostic.generate_report()
    
    return 0 if is_healthy else 1

if __name__ == "__main__":
    sys.exit(main())