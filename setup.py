# #!/usr/bin/env python3
# """
# Setup script for AI-Powered Surveillance System.
# Downloads required models and creates necessary directories.
# """
# import os
# import urllib.request
# import bz2
# import shutil
# from pathlib import Path

# def create_directories():
#     """Create necessary directories."""
#     directories = [
#         'face_gallery',
#         'surveillance_output',
#         'surveillance_output/clips',
#         'surveillance_output/images', 
#         'surveillance_output/logs',
#         'models',
#         'camera',
#         'detection',
#         'face_recognition_s',
#         'anomaly',
#         'utils'
#     ]
    
#     for directory in directories:
#         Path(directory).mkdir(parents=True, exist_ok=True)
#         print(f"✓ Created directory: {directory}")

# def create_init_files():
#     """Create __init__.py files for packages."""
#     init_files = {
#         'camera/__init__.py': '''"""Camera handling module for surveillance system."""
# from .stream_handler import CameraStream, MultiCameraManager
# __all__ = ['CameraStream', 'MultiCameraManager']''',
        
#         'detection/__init__.py': '''"""Motion detection module for surveillance system."""
# from .motion_detector import MotionDetector, MotionDetectionMethod, MotionEvent
# __all__ = ['MotionDetector', 'MotionDetectionMethod', 'MotionEvent']''',
        
#         'face_recognition_s/__init__.py': '''"""Face detection and recognition module for surveillance system."""
# from .face_detector import FaceDetector, FaceDetection
# from .face_matcher import FaceMatcher, KnownFace
# __all__ = ['FaceDetector', 'FaceDetection', 'FaceMatcher', 'KnownFace']''',
        
#         'anomaly/__init__.py': '''"""Anomaly detection module for surveillance system."""
# from .anomaly_detector import AnomalyDetector, AnomalyEvent, AnomalyType, BaselineProfile
# __all__ = ['AnomalyDetector', 'AnomalyEvent', 'AnomalyType', 'BaselineProfile']''',
        
#         'utils/__init__.py': '''"""Utility modules for surveillance system."""
# from .config import config, Config
# from .logger import logger, SurveillanceLogger
# __all__ = ['config', 'Config', 'logger', 'SurveillanceLogger']'''
#     }
    
#     for file_path, content in init_files.items():
#         with open(file_path, 'w',encoding='utf-8') as f:
#             f.write(content)
#         print(f"✓ Created: {file_path}")

# def download_face_landmarks():
#     """Download the face landmarks predictor model."""
#     model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
#     model_path = "models/shape_predictor_68_face_landmarks.dat"
#     compressed_path = "models/shape_predictor_68_face_landmarks.dat.bz2"
    
#     if os.path.exists(model_path):
#         print("✓ Face landmarks model already exists")
#         return
    
#     try:
#         print("📥 Downloading face landmarks predictor (this may take a few minutes)...")
#         urllib.request.urlretrieve(model_url, compressed_path)
        
#         print("📦 Extracting model...")
#         with bz2.BZ2File(compressed_path, 'rb') as f_in:
#             with open(model_path, 'wb',encoding='utf-8') as f_out:
#                 shutil.copyfileobj(f_in, f_out)
        
#         os.remove(compressed_path)
#         print(f"✓ Face landmarks model downloaded: {model_path}")
        
#     except Exception as e:
#         print(f"⚠️  Failed to download face landmarks model: {e}")
#         print("   The system will work without it, but facial landmarks won't be available")

# def create_sample_config():
#     """Create a sample configuration file."""
#     sample_config = '''# AI-Powered Surveillance System Configuration
# # Copy this file to config.ini and modify as needed

# [camera]
# device_id = 0
# resolution_width = 640  
# resolution_height = 480
# fps = 30

# [detection]  
# method = background_subtraction  # background_subtraction, frame_differencing, optical_flow
# threshold = 25
# min_area = 500

# [face]
# model = hog  # hog or cnn (cnn requires more GPU memory)
# tolerance = 0.6
# detection_scale = 0.5

# [anomaly]
# motion_threshold = 0.3
# min_confidence = 0.7
# learning_period = 3600  # seconds

# [gpu]
# use_cuda = true
# memory_fraction = 0.7
# cache_cleanup_interval = 100

# [logging]
# log_level = INFO
# save_clips = true
# save_images = true
# clip_duration = 10
# max_storage_gb = 5.0
# '''

#     with open('sample_config.ini', 'w',encoding='utf-8') as f:
#         f.write(sample_config)
#     print("✓ Created sample_config.ini")

# def create_readme_for_face_gallery():
#     """Create README for face gallery."""
#     readme_content = '''# Face Gallery

# This directory contains known face images for recognition.

# ## Adding Face Images:

# 1. **One face per image**: Each image should contain only one person's face
# 2. **Good quality**: Use clear, well-lit photos  
# 3. **Naming convention**: Use the person's name as the filename
#    - Example: `john_doe.jpg`, `jane_smith.png`
# 4. **Supported formats**: JPG, JPEG, PNG, BMP

# ## Examples:

# ```
# face_gallery/
# ├── alice_johnson.jpg
# ├── bob_wilson.png
# ├── charlie_brown.jpg
# └── diana_prince.jpeg
# ```

# ## Tips:

# - Use multiple photos of the same person with different angles/lighting
# - Name them: `alice_01.jpg`, `alice_02.jpg`, etc.
# - Avoid group photos or multiple faces in one image
# - Face should be clearly visible and not obscured

# The system will automatically process these images and create face encodings for recognition.
# '''

#     with open('face_gallery/README.md', 'w',encoding='utf-8') as f:
#         f.write(readme_content)
#     print("✓ Created face_gallery/README.md")

# def main():
#     """Main setup function."""
#     print("🔍 AI-Powered Surveillance System Setup")
#     print("=" * 50)
    
#     print("\n1. Creating directories...")
#     create_directories()
    
#     print("\n2. Creating package files...")
#     create_init_files()
    
#     print("\n3. Creating configuration files...")
#     create_sample_config()
#     create_readme_for_face_gallery()
    
#     print("\n4. Downloading models (optional)...")
#     response = input("Download face landmarks model? (y/N): ").lower().strip()
#     if response == 'y':
#         download_face_landmarks()
#     else:
#         print("⏭️  Skipped face landmarks download")
#         print("   You can download it later by running this script again")
    
#     print("\n" + "=" * 50)
#     print("✅ Setup complete!")
#     print("\n📋 Next steps:")
#     print("1. Add face images to the 'face_gallery/' directory")
#     print("2. Install dependencies: pip install -r requirements.txt") 
#     print("3. Run the system: python main.py")
#     print("\n💡 For help: python main.py --help")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Complete Setup and Installation Script for AI Surveillance System
================================================================

This script sets up the entire surveillance system with all dependencies,
creates necessary directories, downloads models, and validates the installation.

Features:
- Dependency validation and installation
- Directory structure creation
- Model downloads and verification
- System validation and testing
- Configuration file generation
- Database setup (optional)

Author: AI Surveillance System
Version: 2.0
"""

import os
import sys
import subprocess
import urllib.request
import bz2
import shutil
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import platform

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message: str, color: str = Colors.WHITE):
    """Print colored message."""
    print(f"{color}{message}{Colors.END}")

def print_header(title: str):
    """Print section header."""
    print("\n" + "="*70)
    print_colored(title.center(70), Colors.BOLD + Colors.CYAN)
    print("="*70)

def print_success(message: str):
    """Print success message."""
    print_colored(f"✅ {message}", Colors.GREEN)

def print_warning(message: str):
    """Print warning message."""
    print_colored(f"⚠️  {message}", Colors.YELLOW)

def print_error(message: str):
    """Print error message."""
    print_colored(f"❌ {message}", Colors.RED)

def print_info(message: str):
    """Print info message."""
    print_colored(f"ℹ️  {message}", Colors.BLUE)

def run_command(command: str, description: str = None) -> bool:
    """Run a command and return success status."""
    if description:
        print_info(f"{description}...")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            if description:
                print_success(f"{description} completed")
            return True
        else:
            print_error(f"Command failed: {command}")
            print_error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"Exception running command: {e}")
        return False

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print_header("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    required_version = (3, 8)
    
    print_info(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    print_info(f"Required Python version: {required_version[0]}.{required_version[1]}+")
    
    if version >= required_version:
        print_success("Python version is compatible")
        return True
    else:
        print_error(f"Python {required_version[0]}.{required_version[1]}+ is required")
        return False

def check_system_requirements() -> Dict[str, bool]:
    """Check system requirements."""
    print_header("CHECKING SYSTEM REQUIREMENTS")
    
    requirements = {}
    
    # Check operating system
    os_name = platform.system()
    print_info(f"Operating System: {os_name}")
    requirements['os_supported'] = os_name in ['Linux', 'Windows', 'Darwin']
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print_info(f"Available RAM: {memory_gb:.1f} GB")
        requirements['memory_sufficient'] = memory_gb >= 4.0
        if memory_gb >= 4.0:
            print_success("Memory requirement met (4GB+)")
        else:
            print_warning("Low memory detected (recommended: 4GB+)")
    except ImportError:
        print_warning("Cannot check memory (psutil not installed)")
        requirements['memory_sufficient'] = True
    
    # Check disk space
    try:
        statvfs = os.statvfs('.')
        free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        print_info(f"Available disk space: {free_gb:.1f} GB")
        requirements['disk_sufficient'] = free_gb >= 2.0
        if free_gb >= 2.0:
            print_success("Disk space requirement met (2GB+)")
        else:
            print_warning("Low disk space (recommended: 2GB+)")
    except:
        print_warning("Cannot check disk space")
        requirements['disk_sufficient'] = True
    
    # Check for camera
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print_success("Camera detected")
            requirements['camera_available'] = True
            cap.release()
        else:
            print_warning("No camera detected (can use video files)")
            requirements['camera_available'] = False
    except:
        print_warning("Cannot check camera (OpenCV not installed)")
        requirements['camera_available'] = False
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_success(f"GPU detected: {gpu_name}")
            requirements['gpu_available'] = True
        else:
            print_info("No GPU detected (CPU-only mode)")
            requirements['gpu_available'] = False
    except:
        print_info("Cannot check GPU (PyTorch not installed)")
        requirements['gpu_available'] = False
    
    return requirements

def create_directories():
    """Create necessary directories."""
    print_header("CREATING DIRECTORY STRUCTURE")
    
    directories = [
        # Core directories
        'face_gallery',
        'faces',
        'surveillance_output',
        'surveillance_output/clips',
        'surveillance_output/images',
        'surveillance_output/logs',
        'models',
        'cache',
        'backup',
        'backup/daily',
        'backup/weekly',
        'backup/monthly',
        'attendance_reports',
        
        # Module directories
        'camera',
        'detection',
        'face_recognition_s',
        'anomaly',
        'utils',
        
        # Configuration directories
        'config',
        'logs',
        'temp'
    ]
    
    created_count = 0
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print_success(f"Created directory: {directory}")
            created_count += 1
        except Exception as e:
            print_error(f"Failed to create directory {directory}: {e}")
    
    print_info(f"Created {created_count}/{len(directories)} directories")

def create_init_files():
    """Create __init__.py files for packages."""
    print_header("CREATING PACKAGE INITIALIZATION FILES")
    
    init_files = {
        'camera/__init__.py': '''"""Camera handling module for surveillance system."""
from .stream_handler import CameraStream
__all__ = ['CameraStream']''',
        
        'detection/__init__.py': '''"""Motion detection module for surveillance system."""
from .motion_detector import MotionDetector, MotionDetectionMethod, MotionEvent
__all__ = ['MotionDetector', 'MotionDetectionMethod', 'MotionEvent']''',
        
        'face_recognition_s/__init__.py': '''"""Face detection and recognition module for surveillance system."""
from .face_detector import FaceDetector, FaceDetection
from .face_matcher import FaceMatcher
__all__ = ['FaceDetector', 'FaceDetection', 'FaceMatcher']''',
        
        'anomaly/__init__.py': '''"""Anomaly detection module for surveillance system."""
from .anomaly_detector import AnomalyDetector, AnomalyEvent, AnomalyType
__all__ = ['AnomalyDetector', 'AnomalyEvent', 'AnomalyType']''',
        
        'utils/__init__.py': '''"""Utility modules for surveillance system."""
from .config import config
from .logger import logger
__all__ = ['config', 'logger']'''
    }
    
    for file_path, content in init_files.items():
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print_success(f"Created: {file_path}")
        except Exception as e:
            print_error(f"Failed to create {file_path}: {e}")

def install_dependencies():
    """Install Python dependencies."""
    print_header("INSTALLING PYTHON DEPENDENCIES")
    
    # Check if pip is available
    if not run_command("pip --version", "Checking pip availability"):
        print_error("pip is not available. Please install pip first.")
        return False
    
    # Upgrade pip
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Install core dependencies
    core_deps = [
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "openpyxl>=3.0.0",
        "face-recognition>=1.3.0",
        "dlib>=19.22.0",
        "fastapi>=0.95.0",
        "uvicorn[standard]>=0.20.0",
        "streamlit>=1.20.0",
        "plotly>=5.0.0",
        "pydantic>=1.10.0",
        "requests>=2.28.0",
        "python-dateutil>=2.8.0",
        "Pillow>=8.0.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.60.0"
    ]
    
    print_info("Installing core dependencies...")
    for dep in core_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}"):
            print_warning(f"Failed to install {dep}")
    
    # Optional GPU dependencies
    print_info("Installing optional GPU dependencies...")
    gpu_deps = [
        "torch>=1.12.0",
        "torchvision>=0.13.0"
    ]
    
    for dep in gpu_deps:
        run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]} (optional)")
    
    print_success("Dependency installation completed")

def download_models():
    """Download required models."""
    print_header("DOWNLOADING REQUIRED MODELS")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download face landmarks model
    landmarks_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    landmarks_file = models_dir / "shape_predictor_68_face_landmarks.dat"
    compressed_file = models_dir / "shape_predictor_68_face_landmarks.dat.bz2"
    
    if landmarks_file.exists():
        print_success("Face landmarks model already exists")
    else:
        try:
            print_info("Downloading face landmarks predictor (this may take a few minutes)...")
            urllib.request.urlretrieve(landmarks_url, compressed_file)
            
            print_info("Extracting model...")
            with bz2.BZ2File(compressed_file, 'rb') as f_in:
                with open(landmarks_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            compressed_file.unlink()
            print_success(f"Face landmarks model downloaded: {landmarks_file}")
            
        except Exception as e:
            print_error(f"Failed to download face landmarks model: {e}")
            print_warning("The system will work without it, but facial landmarks won't be available")

def create_configuration_files():
    """Create configuration files."""
    print_header("CREATING CONFIGURATION FILES")
    
    # Main configuration file
    config_content = '''"""
AI-Powered Surveillance System Configuration
===========================================
"""

[camera]
device_id = 0
resolution_width = 640
resolution_height = 480
fps = 30
buffer_size = 1

[detection]
method = background_subtraction
background_subtractor = MOG2
threshold = 25
min_area = 500
learning_rate = 0.01

[face]
model = hog
tolerance = 0.6
detection_scale = 0.5
face_gallery_path = face_gallery
encodings_file = known_faces.pkl

[attendance]
enabled = true
face_gallery_path = faces
attendance_file = attendance.xlsx
face_tolerance = 0.5
cooldown_seconds = 600
auto_backup = true
retention_days = 90

[anomaly]
motion_threshold = 0.3
min_confidence = 0.7
time_window = 30

[logging]
log_level = INFO
output_dir = surveillance_output
save_clips = true
save_images = true
clip_duration = 10
max_storage_gb = 5.0

[gpu]
use_cuda = true
memory_fraction = 0.7
cache_cleanup_interval = 100
'''
    
    try:
        with open('config.ini', 'w', encoding='utf-8') as f:
            f.write(config_content)
        print_success("Created config.ini")
    except Exception as e:
        print_error(f"Failed to create config.ini: {e}")
    
    # Environment file
    env_content = '''# AI Surveillance System Environment Variables

# Camera settings
CAMERA_ID=0
CAMERA_RESOLUTION=640x480

# API settings
API_HOST=0.0.0.0
API_PORT=8080

# Attendance settings
ATTENDANCE_ENABLED=true
FACE_GALLERY_DIR=faces
ATTENDANCE_FILE=attendance.xlsx
FACE_TOLERANCE=0.5
COOLDOWN_SECONDS=600

# Logging
LOG_LEVEL=INFO
ENABLE_DEBUG=false

# GPU
ENABLE_GPU=true

# Security (optional)
# API_SECRET_KEY=your-secret-key-here
# ENABLE_API_AUTH=false
'''
    
    try:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print_success("Created .env file")
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")

def create_readme_files():
    """Create README files for various directories."""
    print_header("CREATING README FILES")
    
    readme_files = {
        'face_gallery/README.md': '''# Face Gallery for Recognition

This directory contains face images for the face recognition system.

## Adding Face Images:

1. **One face per image**: Each image should contain only one person's face
2. **Good quality**: Use clear, well-lit photos
3. **Naming convention**: Use the person's name as the filename
   - Example: `john_doe.jpg`, `jane_smith.png`
4. **Supported formats**: JPG, JPEG, PNG, BMP

## Examples:

```
face_gallery/
├── alice_johnson.jpg
├── bob_wilson.png
├── charlie_brown.jpg
└── diana_prince.jpeg
```

## Tips:

- Use multiple photos of the same person with different angles/lighting
- Name them: `alice_01.jpg`, `alice_02.jpg`, etc.
- Avoid group photos or multiple faces in one image
- Face should be clearly visible and not obscured

The system will automatically process these images and create face encodings for recognition.
''',

        'faces/README.md': '''# Employee Face Gallery for Attendance

This directory contains employee face images for attendance tracking.

## File Naming Convention:
- Use employee ID as filename: EMP001.jpg, JOHN_DOE.png, etc.
- Supported formats: .jpg, .jpeg, .png, .bmp

## Guidelines:
- One face per image
- Clear, well-lit photos
- Face should be clearly visible
- Avoid group photos

## Examples:
- EMP001.jpg
- JOHN_SMITH.png
- JANE_DOE.jpeg

The system will automatically process these images and create face encodings for attendance tracking.
''',

        'surveillance_output/README.md': '''# Surveillance Output Directory

This directory stores all surveillance system outputs:

- `clips/` - Video recordings of events
- `images/` - Captured images and snapshots
- `logs/` - System log files

Files are automatically organized by date and event type.
''',

        'attendance_reports/README.md': '''# Attendance Reports

This directory contains exported attendance reports in Excel format.

Reports include:
- Daily attendance summaries
- Employee visit logs
- Statistical analysis
- Backup files

Files are automatically timestamped for easy tracking.
''',

        'README.md': '''# AI-Powered Surveillance System with Employee Attendance

A comprehensive surveillance solution with real-time face recognition and attendance tracking.

## Features

- **Real-time Video Processing**: Motion detection, face recognition, anomaly detection
- **Employee Attendance**: Automatic attendance tracking with face recognition
- **Web Dashboard**: Real-time monitoring and control interface
- **REST API**: Complete API for integration and remote control
- **Multi-Camera Support**: Handle multiple camera feeds
- **GPU Acceleration**: Optional CUDA support for better performance

## Quick Start

1. **Installation**:
   ```bash
   python setup.py
   ```

2. **Add Employee Photos**:
   - Place employee photos in the `faces/` directory
   - Use employee ID as filename (e.g., `EMP001.jpg`)

3. **Run the System**:
   ```bash
   python api_server.py
   ```

4. **Access Web Interface**:
   - Open http://localhost:8080/docs for API documentation
   - Use the web interface for monitoring

## Configuration

Edit `config.ini` to customize system behavior:
- Camera settings
- Detection parameters
- Attendance settings
- GPU options

## Usage Examples

```bash
# Start with default settings
python api_server.py

# Use different camera
python api_server.py --camera 1

# Run without GUI
python api_server.py --headless

# API server only
python api_server.py --api-only

# Export attendance report
python api_server.py --export-report --days 30
```

## Directory Structure

```
├── api_server.py              # Main application
├── attendance_module.py       # Attendance tracking
├── camera/                    # Camera handling
├── detection/                 # Motion detection
├── face_recognition_s/        # Face recognition
├── anomaly/                   # Anomaly detection
├── utils/                     # Utilities
├── faces/                     # Employee photos
├── face_gallery/              # Recognition photos
├── surveillance_output/       # System outputs
└── attendance_reports/        # Attendance reports
```

## API Endpoints

- `GET /health` - System health check
- `GET /api/status` - System status
- `POST /api/control` - System control
- `GET /api/attendance/status` - Attendance status
- `POST /api/attendance/export` - Export reports

See `/docs` for complete API documentation.

## Requirements

- Python 3.8+
- OpenCV
- dlib
- face_recognition
- FastAPI
- And more (see requirements.txt)

## Support

For issues and questions, check the logs in `surveillance_output/logs/`.
'''
    }
    
    for file_path, content in readme_files.items():
        try:
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path_obj, 'w', encoding='utf-8') as f:
                f.write(content)
            print_success(f"Created: {file_path}")
        except Exception as e:
            print_error(f"Failed to create {file_path}: {e}")

def create_startup_scripts():
    """Create startup scripts for different platforms."""
    print_header("CREATING STARTUP SCRIPTS")
    
    # Windows batch script
    windows_script = '''@echo off
echo Starting AI Surveillance System...
python api_server.py %*
pause
'''
    
    try:
        with open('start_surveillance.bat', 'w', encoding='utf-8') as f:
            f.write(windows_script)
        print_success("Created start_surveillance.bat (Windows)")
    except Exception as e:
        print_error(f"Failed to create Windows script: {e}")
    
    # Linux/Mac shell script
    unix_script = '''#!/bin/bash
echo "Starting AI Surveillance System..."
python3 api_server.py "$@"
'''
    
    try:
        with open('start_surveillance.sh', 'w', encoding='utf-8') as f:
            f.write(unix_script)
        os.chmod('start_surveillance.sh', 0o755)
        print_success("Created start_surveillance.sh (Linux/Mac)")
    except Exception as e:
        print_error(f"Failed to create Unix script: {e}")

def validate_installation():
    """Validate the installation."""
    print_header("VALIDATING INSTALLATION")
    
    validation_results = {}
    
    # Check Python imports
    modules_to_check = [
        'cv2', 'numpy', 'pandas', 'face_recognition', 
        'fastapi', 'uvicorn', 'streamlit', 'plotly'
    ]
    
    print_info("Checking Python module imports...")
    for module in modules_to_check:
        try:
            __import__(module)
            print_success(f"✓ {module}")
            validation_results[f'import_{module}'] = True
        except ImportError as e:
            print_error(f"✗ {module}: {e}")
            validation_results[f'import_{module}'] = False
    
    # Check file structure
    print_info("Checking file structure...")
    required_files = [
        'api_server.py',
        'attendance_module.py',
        'utils/config.py',
        'utils/logger.py',
        'config.ini',
        '.env'
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(f"✓ {file_path}")
            validation_results[f'file_{file_path}'] = True
        else:
            print_error(f"✗ {file_path}")
            validation_results[f'file_{file_path}'] = False
    
    # Check directories
    print_info("Checking directory structure...")
    required_dirs = [
        'faces', 'face_gallery', 'surveillance_output',
        'attendance_reports', 'models', 'cache'
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_success(f"✓ {dir_path}/")
            validation_results[f'dir_{dir_path}'] = True
        else:
            print_error(f"✗ {dir_path}/")
            validation_results[f'dir_{dir_path}'] = False
    
    # Test basic functionality
    print_info("Testing basic functionality...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print_success("✓ Camera access")
            validation_results['camera_test'] = True
            cap.release()
        else:
            print_warning("✗ Camera access (not critical)")
            validation_results['camera_test'] = False
    except Exception as e:
        print_warning(f"✗ Camera test failed: {e}")
        validation_results['camera_test'] = False
    
    # Calculate success rate
    total_checks = len(validation_results)
    passed_checks = sum(validation_results.values())
    success_rate = (passed_checks / total_checks) * 100
    
    print_info(f"Validation Results: {passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print_success("Installation validation PASSED")
        return True
    else:
        print_error("Installation validation FAILED")
        return False

def create_sample_data():
    """Create sample configuration and test data."""
    print_header("CREATING SAMPLE DATA")
    
    # Create sample face images info
    sample_info = '''
# Sample Employee Setup

To test the attendance system, you can:

1. Add sample employee photos to the `faces/` directory:
   - faces/EMP001.jpg (John Doe)
   - faces/EMP002.jpg (Jane Smith)
   - faces/EMP003.jpg (Bob Wilson)

2. Use clear, front-facing photos with good lighting

3. Start the system and test face recognition:
   ```bash
   python api_server.py --validate-system
   python api_server.py --list-employees
   ```

4. Test attendance export:
   ```bash
   python api_server.py --export-report --days 7
   ```
'''
    
    try:
        with open('SAMPLE_SETUP.md', 'w', encoding='utf-8') as f:
            f.write(sample_info)
        print_success("Created SAMPLE_SETUP.md")
    except Exception as e:
        print_error(f"Failed to create sample setup file: {e}")

def main():
    """Main setup function."""
    print_colored("🔍 AI-POWERED SURVEILLANCE SYSTEM SETUP", Colors.BOLD + Colors.CYAN)
    print_colored("=" * 70, Colors.CYAN)
    
    print_info("This script will set up the complete surveillance system with all dependencies.")
    print_info("The setup process may take several minutes depending on your internet connection.")
    
    # Confirm setup
    try:
        response = input("\nDo you want to continue with the setup? (y/N): ").lower().strip()
        if response != 'y':
            print_info("Setup cancelled by user")
            return 0
    except KeyboardInterrupt:
        print_info("\nSetup cancelled by user")
        return 0
    
    setup_start_time = time.time()
    
    # Step 1: Check Python version
    if not check_python_version():
        print_error("Setup cannot continue with incompatible Python version")
        return 1
    
    # Step 2: Check system requirements
    requirements = check_system_requirements()
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Create package files
    create_init_files()
    
    # Step 5: Install dependencies
    print_info("Installing dependencies (this may take several minutes)...")
    install_dependencies()
    
    # Step 6: Download models
    download_models()
    
    # Step 7: Create configuration files
    create_configuration_files()
    
    # Step 8: Create documentation
    create_readme_files()
    
    # Step 9: Create startup scripts
    create_startup_scripts()
    
    # Step 10: Create sample data
    create_sample_data()
    
    # Step 11: Validate installation
    validation_success = validate_installation()
    
    # Setup completion
    setup_time = time.time() - setup_start_time
    
    print_header("SETUP COMPLETION SUMMARY")
    
    print_info(f"Setup completed in {setup_time:.1f} seconds")
    
    if validation_success:
        print_success("✅ SETUP COMPLETED SUCCESSFULLY!")
        print_info("\nNext steps:")
        print_info("1. Add employee photos to the 'faces/' directory")
        print_info("2. Run the system: python api_server.py")
        print_info("3. Access the web interface at http://localhost:8080/docs")
        print_info("4. Check the README.md file for detailed usage instructions")
        
        # Quick start suggestions
        print_info("\nQuick commands to try:")
        print_info("• python api_server.py --validate-system")
        print_info("• python api_server.py --list-employees")
        print_info("• python api_server.py --help")
        
        return 0
    else:
        print_warning("⚠️  Setup completed with some issues")
        print_info("Please check the validation results above and resolve any missing dependencies")
        print_info("The system may still work with reduced functionality")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_info("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)