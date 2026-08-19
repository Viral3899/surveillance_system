
# Quick Start Guide

## Running the Surveillance System

### Option 1: Run the Main System
```bash
# With GUI (default)
python main.py

# Headless mode (no GUI)
python main.py --headless

# With web interface
python main.py --web
```

### Option 2: Run the API Server
```bash
# Start the API server
python api_server.py

# The API will be available at:
# - API: http://localhost:8080
# - Docs: http://localhost:8080/docs
# - Health: http://localhost:8080/health
```

### Option 3: Run Both (Recommended for Production)
```bash
# Terminal 1: Start API server
python api_server.py

# Terminal 2: Start surveillance system
python main.py --headless
```

## API Usage Examples

### Check System Health
```bash
curl http://localhost:8080/health
```

### Get System Status
```bash
curl http://localhost:8080/status
```

### Start Surveillance System via API
```bash
curl -X POST http://localhost:8080/control \
  -H "Content-Type: application/json" \
  -d '{"action": "start"}'
```

### Stop Surveillance System
```bash
curl -X POST http://localhost:8080/control \
  -H "Content-Type: application/json" \
  -d '{"action": "stop"}'
```

### Get Attendance Logs
```bash
curl http://localhost:8080/attendance/logs?limit=10
```

### Get API Metrics
```bash
curl http://localhost:8080/metrics
```

## Configuration

The system uses environment variables for configuration. Key settings:

- `CAMERA_ID`: Camera device ID (default: 0)
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8080)
- `ATTENDANCE_ENABLED`: Enable attendance module (default: true)
- `ENABLE_GPU`: Enable GPU acceleration (default: true)
- `LOG_LEVEL`: Logging level (default: INFO)

## Features

✅ Real-time face recognition
✅ Motion detection
✅ Anomaly detection
✅ Attendance tracking
✅ REST API for control
✅ Performance monitoring
✅ Health checks
✅ Comprehensive error handling

## Troubleshooting

### Camera Not Detected
- Check camera permissions
- Try different camera IDs: `python main.py --camera 1`

### API Server Not Starting
- Check if port 8080 is available
- Change port: `export API_PORT=8081`

### GPU Not Working
- Verify CUDA is installed
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

## Next Steps

1. Add employee faces to `faces/` directory
2. Configure settings via environment variables
3. Access API documentation at http://localhost:8080/docs
4. Monitor system via `/status` and `/metrics` endpoints

