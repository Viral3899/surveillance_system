#!/usr/bin/env python3
"""FastAPI server for controlling the surveillance system and attendance data."""
from __future__ import annotations

import os
import threading
import time
import psutil
from typing import List, Optional, Dict, Any
from datetime import datetime

import uvicorn      
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from attendance.storage import AttendanceRepository
from surveillance.system import (
    SurveillanceController,
    get_cuda_device_name,
    is_cuda_ready,
)
from utils.config import config
from utils.logger import logger


class ControlRequest(BaseModel):
    action: str = Field(..., pattern="^(start|stop|reset|restart)$")
    gui_mode: Optional[bool] = None


class SettingsRequest(BaseModel):
    motion_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    learning_mode: Optional[bool] = None


class EmployeeRequest(BaseModel):
    employee_id: str = Field(..., min_length=1, max_length=64)
    employee_name: Optional[str] = Field(None, max_length=128)
    image_path: Optional[str] = None


controller = SurveillanceController(
    camera_id=config.camera.device_id,
    gui_mode=False,
    enable_signal_handlers=False,
)

# Initialize attendance repository with MySQL support
if config.attendance.database_type == "mysql":
    mysql_config = {
        "host": config.attendance.mysql_host,
        "port": config.attendance.mysql_port,
        "user": config.attendance.mysql_user,
        "password": config.attendance.mysql_password,
        "database": config.attendance.mysql_database,
    }
    attendance_repo = AttendanceRepository(
        database_type="mysql",
        mysql_config=mysql_config
    )
    logger.info(f"API server using MySQL: {config.attendance.mysql_database}")
else:
    attendance_repo = AttendanceRepository(
        database_path=config.attendance.database_file,
        database_type="sqlite"
    )
    logger.info(f"API server using SQLite: {config.attendance.database_file}")

_repo_lock = threading.Lock()

# Performance metrics tracking
_metrics = {
    "start_time": time.time(),
    "request_count": 0,
    "error_count": 0,
    "last_request_time": None,
    "endpoint_stats": {}
}

app = FastAPI(
    title="AI Surveillance API",
    version="1.0.0",
    description="REST API for controlling the AI-powered surveillance system and managing attendance data",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    _metrics["error_count"] += 1
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    _metrics["error_count"] += 1
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track API requests for metrics."""
    start_time = time.time()
    _metrics["request_count"] += 1
    _metrics["last_request_time"] = datetime.now().isoformat()
    
    endpoint = request.url.path
    if endpoint not in _metrics["endpoint_stats"]:
        _metrics["endpoint_stats"][endpoint] = {"count": 0, "total_time": 0.0}
    _metrics["endpoint_stats"][endpoint]["count"] += 1
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    _metrics["endpoint_stats"][endpoint]["total_time"] += process_time
    
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/health", tags=["System"])
def health_check():
    """
    Health check endpoint.
    
    Returns basic system health status.
    """
    try:
        system_status = controller.get_status()
        is_healthy = True
        
        # Check if system is responsive
        if "status" in system_status and system_status["status"] == "error":
            is_healthy = False
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - _metrics["start_time"],
            "system_running": controller.is_running
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/status", tags=["System"])
def get_status():
    """
    Get comprehensive system status.
    
    Returns detailed information about:
    - Surveillance system state
    - Attendance summary
    - GPU availability
    - System performance metrics
    """
    try:
        system_status = controller.get_status()
        summary = attendance_repo.get_summary()
        
        # Get system resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        status_data = {
            **system_status,
            "attendance_summary": summary,
            "gpu": {
                "available": is_cuda_ready(),
                "device": get_cuda_device_name(),
            },
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "memory_total_mb": memory.total / (1024 * 1024),
            },
            "api_metrics": {
                "uptime_seconds": time.time() - _metrics["start_time"],
                "total_requests": _metrics["request_count"],
                "total_errors": _metrics["error_count"],
                "last_request": _metrics["last_request_time"],
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return status_data
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system status: {str(e)}"
        )


@app.post("/control", tags=["Control"])
def control_system(request: ControlRequest):
    """
    Control the surveillance system.
    
    Actions:
    - start: Start the surveillance system in background mode
    - stop: Stop the surveillance system
    - reset: Reset all detectors (motion, face, anomaly)
    - restart: Restart the surveillance system
    
    Args:
        request: ControlRequest with action and optional gui_mode
    """
    try:
        action = request.action.lower()
        
        if action == "start":
            if controller.is_running:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="System is already running"
                )
            started = controller.start(background=True)
            if not started:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to start system"
                )
            logger.info("Surveillance system started via API")
            return {"status": "starting", "message": "System is starting in background"}
            
        elif action == "stop":
            if not controller.is_running:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="System is not running"
                )
            controller.stop()
            logger.info("Surveillance system stopped via API")
            return {"status": "stopped", "message": "System has been stopped"}
            
        elif action == "reset":
            if not controller.is_running:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="System must be running to reset detectors"
                )
            controller.reset_detectors()
            logger.info("Detectors reset via API")
            return {"status": "detectors_reset", "message": "All detectors have been reset"}
            
        elif action == "restart":
            if request.gui_mode is not None:
                controller.set_gui_mode(request.gui_mode)
            controller.restart()
            logger.info("Surveillance system restarted via API")
            return {"status": "restarting", "message": "System is restarting"}
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown action: {action}. Valid actions: start, stop, reset, restart"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error controlling system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to control system: {str(e)}"
        )


@app.post("/settings", tags=["Control"])
def update_settings(settings: SettingsRequest):
    """
    Update surveillance system settings.
    
    Args:
        settings: SettingsRequest with optional motion_threshold and learning_mode
    """
    try:
        system = controller.get_system()
        if not system:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="System must be running to update settings"
            )
        
        updated = []
        
        if settings.motion_threshold is not None:
            old_value = config.anomaly.motion_threshold
            config.anomaly.motion_threshold = settings.motion_threshold
            if hasattr(system, 'anomaly_detector') and system.anomaly_detector:
                system.anomaly_detector.update_thresholds(motion_threshold=settings.motion_threshold)
            updated.append(f"motion_threshold: {old_value} -> {settings.motion_threshold}")
            logger.info(f"Motion threshold updated: {old_value} -> {settings.motion_threshold}")
        
        if settings.learning_mode is not None:
            if hasattr(system, 'anomaly_detector') and system.anomaly_detector:
                system.anomaly_detector.set_learning_mode(settings.learning_mode)
            updated.append(f"learning_mode: {settings.learning_mode}")
            logger.info(f"Learning mode updated: {settings.learning_mode}")
        
        if not updated:
            return {"status": "no_changes", "message": "No settings were updated"}
        
        return {
            "status": "updated",
            "message": "Settings updated successfully",
            "updated": updated
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update settings: {str(e)}"
        )


@app.get("/attendance/logs", tags=["Attendance"])
def list_attendance_logs(limit: int = 100, employee_id: Optional[str] = None):
    """
    Get attendance logs.
    
    Args:
        limit: Maximum number of logs to return (default: 100, max: 1000)
        employee_id: Optional filter by employee ID
    """
    try:
        # Validate limit
        limit = max(1, min(limit, 1000))
        
        with _repo_lock:
            logs = attendance_repo.get_recent_logs(limit=limit, employee_id=employee_id)
        
        return {
            "results": logs,
            "count": len(logs),
            "limit": limit,
            "filtered_by_employee": employee_id is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error retrieving attendance logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve attendance logs: {str(e)}"
        )


@app.get("/attendance/employees", tags=["Attendance"])
def list_employees():
    """
    List all registered employees.
    
    Returns a list of all employees in the attendance system.
    """
    try:
        with _repo_lock:
            employees = attendance_repo.list_employees()
        
        return {
            "results": employees,
            "count": len(employees),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing employees: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list employees: {str(e)}"
        )


@app.post("/attendance/employees", tags=["Attendance"])
def register_employee(request: EmployeeRequest):
    """
    Register or update an employee.
    
    Args:
        request: EmployeeRequest with employee_id, optional name and image_path
    """
    try:
        # Validate image path if provided
        if request.image_path and not os.path.exists(request.image_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image file not found: {request.image_path}"
            )
        
        with _repo_lock:
            attendance_repo.upsert_employee(
                employee_id=request.employee_id,
                employee_name=request.employee_name or request.employee_id,
                image_path=request.image_path,
            )
        
        logger.info(f"Employee registered/updated: {request.employee_id}")
        return {
            "status": "saved",
            "message": f"Employee {request.employee_id} saved successfully",
            "employee_id": request.employee_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering employee: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register employee: {str(e)}"
        )


@app.get("/metrics", tags=["System"])
def get_metrics():
    """
    Get API performance metrics.
    
    Returns detailed metrics about API usage and performance.
    """
    try:
        # Calculate average response times
        endpoint_avg_times = {}
        for endpoint, stats in _metrics["endpoint_stats"].items():
            if stats["count"] > 0:
                endpoint_avg_times[endpoint] = {
                    "count": stats["count"],
                    "avg_time_seconds": stats["total_time"] / stats["count"],
                    "total_time_seconds": stats["total_time"]
                }
        
        return {
            "uptime_seconds": time.time() - _metrics["start_time"],
            "uptime_hours": (time.time() - _metrics["start_time"]) / 3600,
            "total_requests": _metrics["request_count"],
            "total_errors": _metrics["error_count"],
            "error_rate": _metrics["error_count"] / max(_metrics["request_count"], 1),
            "last_request_time": _metrics["last_request_time"],
            "endpoint_statistics": endpoint_avg_times,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


def run():
    """Run the API server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    run()
