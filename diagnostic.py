#!/usr/bin/env python3
"""
Attendance System Diagnostic Script
==================================

This script diagnoses common issues with the attendance system
and provides step-by-step fixes.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import importlib

def check_python_environment():
    """Check if Python environment is set up correctly."""
    print("=== Python Environment Check ===")
    
    # Check Python version
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check required packages
    required_packages = [
        'cv2', 'numpy', 'pandas', 'face_recognition', 
        'openpyxl', 'datetime', 'threading'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nTo install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_directory_structure():
    """Check if required directories exist."""
    print("\n=== Directory Structure Check ===")
    
    required_dirs = [
        'faces',           # Employee photos
        'faces',    # Recognition photos (alternative)
        'surveillance_output', # System outputs
        'attendance_reports',  # Report exports
        'models',          # AI models
        'cache'            # Temporary cache
    ]
    
    all_good = True
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/ exists")
        else:
            print(f"❌ {directory}/ missing")
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"   Created {directory}/")
            except Exception as e:
                print(f"   Failed to create: {e}")
                all_good = False
    
    return all_good

def check_face_images():
    """Check if face images are properly set up."""
    print("\n=== Face Images Check ===")
    
    face_dirs = ['faces', 'faces']
    total_images = 0
    
    for face_dir in face_dirs:
        if Path(face_dir).exists():
            image_files = list(Path(face_dir).glob('*.jpg')) + \
                         list(Path(face_dir).glob('*.png')) + \
                         list(Path(face_dir).glob('*.jpeg'))
            
            print(f"📁 {face_dir}: {len(image_files)} images")
            total_images += len(image_files)
            
            # Check a few images for validity
            for img_file in image_files[:3]:
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        height, width = img.shape[:2]
                        print(f"   ✅ {img_file.name}: {width}x{height}")
                    else:
                        print(f"   ❌ {img_file.name}: Cannot read image")
                except Exception as e:
                    print(f"   ❌ {img_file.name}: {e}")
    
    if total_images == 0:
        print("❌ No face images found!")
        print("\nTo add employee photos:")
        print("1. Place photos in the 'faces/' directory")
        print("2. Name them with employee ID: EMP001.jpg, JOHN_DOE.png, etc.")
        print("3. Use clear, front-facing photos")
        return False
    
    print(f"✅ Total face images: {total_images}")
    return True

def test_camera_access():
    """Test if camera can be accessed."""
    print("\n=== Camera Access Test ===")
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"✅ Camera working: {width}x{height}")
                cap.release()
                return True
            else:
                print("❌ Camera opened but no frame received")
        else:
            print("❌ Cannot open camera")
        
        cap.release()
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
    
    print("ℹ️  You can still use the system with video files")
    return False

def test_face_recognition():
    """Test face recognition functionality."""
    print("\n=== Face Recognition Test ===")
    
    try:
        import face_recognition
        
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add a simple face-like pattern
        cv2.rectangle(test_image, (30, 30), (70, 70), (255, 255, 255), -1)
        cv2.circle(test_image, (40, 45), 3, (0, 0, 0), -1)  # Eyes
        cv2.circle(test_image, (60, 45), 3, (0, 0, 0), -1)
        cv2.ellipse(test_image, (50, 60), (8, 4), 0, 0, 180, (0, 0, 0), 1)  # Mouth
        
        # Try to detect faces
        face_locations = face_recognition.face_locations(test_image)
        print(f"✅ Face recognition library working")
        print(f"   Test detected {len(face_locations)} faces")
        
        return True
        
    except ImportError:
        print("❌ face_recognition library not installed")
        print("   Install with: pip install face_recognition")
        return False
    except Exception as e:
        print(f"❌ Face recognition test failed: {e}")
        return False

def test_attendance_module():
    """Test the attendance module functionality."""
    print("\n=== Attendance Module Test ===")
    
    try:
        # Try to import the attendance module
        from attendance_module import EmployeeAttendanceModule
        
        # Create a test instance
        attendance = EmployeeAttendanceModule(
            face_dir="faces",
            attendance_file="test_attendance.xlsx",
            cooldown_seconds=1,  # Short cooldown for testing
            tolerance=0.5
        )
        
        print("✅ Attendance module imported successfully")
        
        # Get statistics
        stats = attendance.get_statistics()
        print(f"   Total employees loaded: {stats.get('total_employees', 0)}")
        print(f"   Safe mode: {stats.get('safe_mode', False)}")
        
        # Clean up test file
        test_file = Path("test_attendance.xlsx")
        if test_file.exists():
            test_file.unlink()
        
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import attendance module: {e}")
        return False
    except Exception as e:
        print(f"❌ Attendance module test failed: {e}")
        return False

def generate_fix_recommendations(test_results):
    """Generate specific fix recommendations based on test results."""
    print("\n" + "="*50)
    print("🔧 FIX RECOMMENDATIONS")
    print("="*50)
    
    if not test_results['environment']:
        print("\n1. INSTALL MISSING DEPENDENCIES:")
        print("   pip install opencv-python numpy pandas face-recognition openpyxl")
        print("   pip install dlib  # May require compilation tools")
    
    if not test_results['directories']:
        print("\n2. CREATE MISSING DIRECTORIES:")
        print("   The script attempted to create them automatically.")
        print("   If failed, create manually: faces/, surveillance_output/, etc.")
    
    if not test_results['face_images']:
        print("\n3. ADD EMPLOYEE PHOTOS:")
        print("   • Place clear photos in the 'faces/' directory")
        print("   • Name format: EMP001.jpg, JOHN_DOE.png, etc.")
        print("   • One face per image, good lighting")
        print("   • Supported formats: .jpg, .png, .jpeg")
    
    if not test_results['camera']:
        print("\n4. CAMERA ISSUES:")
        print("   • Check if camera is connected and not used by other apps")
        print("   • Try different camera ID: --camera 1")
        print("   • Use video files instead of live camera")
    
    if not test_results['face_recognition']:
        print("\n5. FACE RECOGNITION ISSUES:")
        print("   • Install dlib: pip install dlib")
        print("   • On Windows: pip install dlib or use conda")
        print("   • On Linux: apt-get install cmake")
    
    if not test_results['attendance_module']:
        print("\n6. ATTENDANCE MODULE ISSUES:")
        print("   • Ensure attendance_module.py is in the current directory")
        print("   • Check file permissions")
        print("   • Verify all dependencies are installed")

def main():
    """Run complete diagnostic."""
    print("🔍 AI SURVEILLANCE ATTENDANCE SYSTEM DIAGNOSTIC")
    print("=" * 60)
    
    # Run all tests
    test_results = {
        'environment': check_python_environment(),
        'directories': check_directory_structure(),
        'face_images': check_face_images(),
        'camera': test_camera_access(),
        'face_recognition': test_face_recognition(),
        'attendance_module': test_attendance_module()
    }
    
    # Calculate overall health
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    health_percentage = (passed_tests / total_tests) * 100
    
    print(f"\n" + "="*60)
    print(f"📊 DIAGNOSTIC RESULTS: {passed_tests}/{total_tests} tests passed ({health_percentage:.1f}%)")
    
    if health_percentage >= 80:
        print("🟢 SYSTEM STATUS: HEALTHY")
        print("Your attendance system should work properly!")
    elif health_percentage >= 60:
        print("🟡 SYSTEM STATUS: PARTIAL")
        print("System may work with limited functionality.")
    else:
        print("🔴 SYSTEM STATUS: NEEDS ATTENTION")
        print("Several issues need to be resolved.")
    
    # Generate specific recommendations
    generate_fix_recommendations(test_results)
    
    print(f"\n" + "="*60)
    print("💡 NEXT STEPS:")
    print("1. Fix the issues listed above")
    print("2. Re-run this diagnostic: python diagnostic.py")
    print("3. Start the attendance system: python api_server.py")
    print("4. Test with: python api_server.py --validate-system")

if __name__ == "__main__":
    main()