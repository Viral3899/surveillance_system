#!/usr/bin/env python3
"""
Setup script for AI-Powered Surveillance System.
Downloads required models and creates necessary directories.
"""
import os
import urllib.request
import bz2
import shutil
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    directories = [
        'face_gallery',
        'surveillance_output',
        'surveillance_output/clips',
        'surveillance_output/images', 
        'surveillance_output/logs',
        'models',
        'camera',
        'detection',
        'face_recognition_s',
        'anomaly',
        'utils'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for packages."""
    init_files = {
        'camera/__init__.py': '''"""Camera handling module for surveillance system."""
from .stream_handler import CameraStream, MultiCameraManager
__all__ = ['CameraStream', 'MultiCameraManager']''',
        
        'detection/__init__.py': '''"""Motion detection module for surveillance system."""
from .motion_detector import MotionDetector, MotionDetectionMethod, MotionEvent
__all__ = ['MotionDetector', 'MotionDetectionMethod', 'MotionEvent']''',
        
        'face_recognition_s/__init__.py': '''"""Face detection and recognition module for surveillance system."""
from .face_detector import FaceDetector, FaceDetection
from .face_matcher import FaceMatcher, KnownFace
__all__ = ['FaceDetector', 'FaceDetection', 'FaceMatcher', 'KnownFace']''',
        
        'anomaly/__init__.py': '''"""Anomaly detection module for surveillance system."""
from .anomaly_detector import AnomalyDetector, AnomalyEvent, AnomalyType, BaselineProfile
__all__ = ['AnomalyDetector', 'AnomalyEvent', 'AnomalyType', 'BaselineProfile']''',
        
        'utils/__init__.py': '''"""Utility modules for surveillance system."""
from .config import config, Config
from .logger import logger, SurveillanceLogger
__all__ = ['config', 'Config', 'logger', 'SurveillanceLogger']'''
    }
    
    for file_path, content in init_files.items():
        with open(file_path, 'w',encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created: {file_path}")

def download_face_landmarks():
    """Download the face landmarks predictor model."""
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    model_path = "models/shape_predictor_68_face_landmarks.dat"
    compressed_path = "models/shape_predictor_68_face_landmarks.dat.bz2"
    
    if os.path.exists(model_path):
        print("✓ Face landmarks model already exists")
        return
    
    try:
        print("📥 Downloading face landmarks predictor (this may take a few minutes)...")
        urllib.request.urlretrieve(model_url, compressed_path)
        
        print("📦 Extracting model...")
        with bz2.BZ2File(compressed_path, 'rb') as f_in:
            with open(model_path, 'wb',encoding='utf-8') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        os.remove(compressed_path)
        print(f"✓ Face landmarks model downloaded: {model_path}")
        
    except Exception as e:
        print(f"⚠️  Failed to download face landmarks model: {e}")
        print("   The system will work without it, but facial landmarks won't be available")

def create_sample_config():
    """Create a sample configuration file."""
    sample_config = '''# AI-Powered Surveillance System Configuration
# Copy this file to config.ini and modify as needed

[camera]
device_id = 0
resolution_width = 640  
resolution_height = 480
fps = 30

[detection]  
method = background_subtraction  # background_subtraction, frame_differencing, optical_flow
threshold = 25
min_area = 500

[face]
model = hog  # hog or cnn (cnn requires more GPU memory)
tolerance = 0.6
detection_scale = 0.5

[anomaly]
motion_threshold = 0.3
min_confidence = 0.7
learning_period = 3600  # seconds

[gpu]
use_cuda = true
memory_fraction = 0.7
cache_cleanup_interval = 100

[logging]
log_level = INFO
save_clips = true
save_images = true
clip_duration = 10
max_storage_gb = 5.0
'''

    with open('sample_config.ini', 'w',encoding='utf-8') as f:
        f.write(sample_config)
    print("✓ Created sample_config.ini")

def create_readme_for_face_gallery():
    """Create README for face gallery."""
    readme_content = '''# Face Gallery

This directory contains known face images for recognition.

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
'''

    with open('face_gallery/README.md', 'w',encoding='utf-8') as f:
        f.write(readme_content)
    print("✓ Created face_gallery/README.md")

def main():
    """Main setup function."""
    print("🔍 AI-Powered Surveillance System Setup")
    print("=" * 50)
    
    print("\n1. Creating directories...")
    create_directories()
    
    print("\n2. Creating package files...")
    create_init_files()
    
    print("\n3. Creating configuration files...")
    create_sample_config()
    create_readme_for_face_gallery()
    
    print("\n4. Downloading models (optional)...")
    response = input("Download face landmarks model? (y/N): ").lower().strip()
    if response == 'y':
        download_face_landmarks()
    else:
        print("⏭️  Skipped face landmarks download")
        print("   You can download it later by running this script again")
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Add face images to the 'face_gallery/' directory")
    print("2. Install dependencies: pip install -r requirements.txt") 
    print("3. Run the system: python main.py")
    print("\n💡 For help: python main.py --help")

if __name__ == "__main__":
    main()