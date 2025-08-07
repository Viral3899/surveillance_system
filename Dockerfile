# Multi-stage build for AI Surveillance System
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV OPENCV_LOG_LEVEL=ERROR

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    dlib \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    # Face recognition dependencies
    libboost-all-dev \
    # Video/camera dependencies
    v4l-utils \
    ffmpeg \
    # Utilities
    wget \
    curl \
    unzip \
    bzip2 \
    # For GUI support (optional)
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download and setup dlib face landmarks model
RUN mkdir -p /app/models && \
    wget -O /tmp/shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bunzip2 /tmp/shape_predictor_68_face_landmarks.dat.bz2 && \
    mv /tmp/shape_predictor_68_face_landmarks.dat /app/models/

# Create app user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p face_gallery surveillance_output/clips surveillance_output/images surveillance_output/logs && \
    chown -R appuser:appuser face_gallery surveillance_output

# Switch to non-root user
USER appuser

# Run setup script
RUN python setup.py

# Expose ports for web interface
EXPOSE 8501 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import cv2; print('OpenCV available')" || exit 1

# Default command
CMD ["python", "main.py", "--headless"]

# Development stage
FROM base as development
USER root
RUN pip install --no-cache-dir jupyter notebook ipython
USER appuser
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage  
FROM base as production
# Copy only necessary files for production
COPY --from=base --chown=appuser:appuser /app /app
# Production optimizations
ENV PYTHONOPTIMIZE=1
CMD ["python", "main.py", "--headless", "--no-gpu"]