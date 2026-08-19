#!/usr/bin/env python3
"""CLI entry point for the AI-powered surveillance system."""

import argparse
import os
import sys

from utils.config import config
from utils.logger import logger
from surveillance.system import (
    SurveillanceController,
    SurveillanceSystem,
    SurveillanceWebUI,
    get_cuda_device_name,
    is_cuda_ready,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Powered Surveillance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Start with default camera and GUI
  python main.py --camera 1 --headless     # Use camera 1 in headless mode
  python main.py --web                     # Launch Streamlit dashboard
  python main.py --face-gallery ./faces    # Custom face gallery path
        """,
    )
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--headless", "-hl", action="store_true", help="Run without GUI windows")
    parser.add_argument("--web", "-w", action="store_true", help="Launch Streamlit web interface")
    parser.add_argument("--face-gallery", "-fg", type=str, help="Path to face gallery folder")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory for recordings and logs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    return parser.parse_args()


def _validate_camera(camera_id: int):
    if os.name == "nt":
        return
    device_path = f"/dev/video{camera_id}"
    if not os.path.exists(device_path) and camera_id != 0:
        print(f"Warning: Camera device {device_path} may not exist")


def _configure_runtime(args: argparse.Namespace):
    if args.face_gallery:
        if not os.path.exists(args.face_gallery):
            raise FileNotFoundError(f"Face gallery path does not exist: {args.face_gallery}")
        config.face.face_gallery_path = args.face_gallery
        config._create_directories()
    
    if args.output_dir:
        config.logging.output_dir = args.output_dir
        config._create_directories()
    
    if args.verbose:
        config.logging.log_level = "DEBUG"
    
    if args.no_gpu:
        config.gpu.use_cuda = False
    

def _print_startup(args: argparse.Namespace):
    print("=" * 60)
    print("AI-Powered Surveillance System")
    print("=" * 60)
    print(f"Camera: {args.camera}")
    print(f"Display: {'Headless' if args.headless else 'GUI'}")
    print(f"Web Interface: {'Enabled' if args.web else 'Disabled'}")
    print(f"Face Gallery: {config.face.face_gallery_path}")
    print(f"Output Directory: {config.logging.output_dir}")
    gpu_active = config.gpu.use_cuda and is_cuda_ready()
    print(f"GPU Acceleration: {'Enabled' if gpu_active else 'Disabled'}")
    if gpu_active:
        device_name = get_cuda_device_name()
        if device_name:
            print(f"GPU Device: {device_name}")
    print("=" * 60)
    

def main() -> int:
    args = parse_args()
    _validate_camera(args.camera)

    try:
        _configure_runtime(args)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    _print_startup(args)
    logger.info(
        "Starting AI-Powered Surveillance System",
        extra={"camera": args.camera, "headless": args.headless, "web": args.web},
    )

    controller = SurveillanceController(
            camera_id=args.camera,
        gui_mode=not args.headless,
        enable_signal_handlers=True,
        )
        
    try:
        if args.web:
            print("\nStarting web interface.")
            print("Tip: Run 'streamlit run main.py --web' for the best experience")
            system = SurveillanceSystem(camera_id=args.camera, gui_mode=not args.headless)
            web_ui = SurveillanceWebUI(system)
            web_ui.create_streamlit_app()
            return 0

        if not args.headless:
            print("\n" + "="*60)
            print("UNIFIED SINGLE-SCREEN DISPLAY")
            print("="*60)
            print("All features shown in one window:")
            print("  - Live camera feed with motion detection")
            print("  - Face recognition with attendance tracking")
            print("  - System status and statistics")
            print("  - Attendance statistics and visitor list")
            print("\nControls:")


            
            print("  Q - Quit system")
            print("  R - Reset detectors")
            print("  L - Toggle learning mode")
            print("  S - Save current frame")
            print("\nStarting surveillance. Press Q to quit.")
            print("="*60)
        else:
            print("\nStarting headless surveillance. Press Ctrl+C to quit.")
            
        controller.start(background=False)
    
    except KeyboardInterrupt:
        print("\nSurveillance system interrupted by user")
        logger.info("Surveillance system interrupted by user")
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        logger.error(f"Fatal error: {exc}")
        return 1
    finally:
        controller.stop()
    
    print("Surveillance system shutdown complete")
    logger.info("Surveillance system shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
