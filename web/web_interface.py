#!/usr/bin/env python3
"""
Streamlit web interface for AI Surveillance System.
Provides real-time monitoring and control dashboard.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Optional
import base64

# Page configuration
st.set_page_config(
    page_title="AI Surveillance Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ff4b4b;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .alert-medium {
        background-color: #ffa500;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .alert-low {
        background-color: #32cd32;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = os.getenv("SURVEILLANCE_API_URL", "http://localhost:8080")
REFRESH_INTERVAL = 2  # seconds

class SurveillanceAPI:
    """API client for surveillance system."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_status(self) -> Dict:
        """Get system status."""
        try:
            response = self.session.get(f"{self.base_url}/api/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def get_statistics(self) -> Dict:
        """Get system statistics."""
        try:
            response = self.session.get(f"{self.base_url}/api/stats", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        """Get recent events."""
        try:
            response = self.session.get(f"{self.base_url}/api/events?limit={limit}", timeout=5)
            response.raise_for_status()
            return response.json().get("events", [])
        except Exception as e:
            return []
    
    def get_face_database(self) -> Dict:
        """Get face database information."""
        try:
            response = self.session.get(f"{self.base_url}/api/faces", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"total_faces": 0, "faces": []}
    
    def control_system(self, action: str) -> Dict:
        """Control system (start/stop/reset)."""
        try:
            response = self.session.post(f"{self.base_url}/api/control", 
                                       json={"action": action}, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def update_settings(self, settings: Dict) -> Dict:
        """Update system settings."""
        try:
            response = self.session.put(f"{self.base_url}/api/settings", 
                                      json=settings, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "success": False}

# Initialize API client
@st.cache_resource
def get_api_client():
    return SurveillanceAPI(API_BASE_URL)

api = get_api_client()

def main():
    """Main dashboard application."""
    
    # Title and header
    st.title("🔍 AI Surveillance Dashboard")
    
    # Check API connectivity
    status = api.get_status()
    if "error" in status:
        st.error(f"⚠️ Unable to connect to surveillance system: {status['error']}")
        st.info("Please ensure the surveillance system is running and accessible.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ System Controls")
        
        # System status indicator
        system_running = status.get("status") == "running"
        status_color = "🟢" if system_running else "🔴"
        st.markdown(f"**Status:** {status_color} {'Running' if system_running else 'Stopped'}")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start" if not system_running else "⏹️ Stop"):
                action = "start" if not system_running else "stop"
                result = api.control_system(action)
                if result.get("success"):
                    st.success(f"System {action}ed successfully")
                    st.rerun()
                else:
                    st.error(f"Failed to {action} system: {result.get('error', 'Unknown error')}")
        
        with col2:
            if st.button("🔄 Reset"):
                result = api.control_system("reset")
                if result.get("success"):
                    st.success("System reset successfully")
                else:
                    st.error(f"Failed to reset system: {result.get('error', 'Unknown error')}")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        
        # Motion threshold
        motion_threshold = st.slider(
            "Motion Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.3, 
            step=0.1,
            help="Sensitivity for motion detection"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for anomaly detection"
        )
        
        # Learning mode
        learning_mode = st.checkbox(
            "Learning Mode",
            value=status.get("learning_mode", False),
            help="Enable learning mode for baseline establishment"
        )
        
        # Apply settings button
        if st.button("💾 Apply Settings"):
            settings = {
                "motion_threshold": motion_threshold,
                "confidence_threshold": confidence_threshold,
                "learning_mode": learning_mode
            }
            result = api.update_settings(settings)
            if result.get("success"):
                st.success("Settings updated successfully")
            else:
                st.error(f"Failed to update settings: {result.get('error', 'Unknown error')}")
        
        st.divider()
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
        refresh_rate = st.selectbox(
            "Refresh Rate (seconds)",
            options=[1, 2, 5, 10, 30],
            index=1
        )
    
    # Main dashboard content
    if system_running:
        display_running_dashboard(status, auto_refresh, refresh_rate)
    else:
        display_stopped_dashboard()
    
    # Auto-refresh mechanism
    if auto_refresh and system_running:
        time.sleep(refresh_rate)
        st.rerun()

def display_running_dashboard(status: Dict, auto_refresh: bool, refresh_rate: int):
    """Display dashboard when system is running."""
    
    # Get additional data
    stats = api.get_statistics()
    recent_events = api.get_recent_events(limit=20)
    face_db = api.get_face_database()
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        fps = status.get("current_fps", 0)
        st.metric(
            label="📹 FPS",
            value=f"{fps:.1f}",
            delta=f"{fps-30:.1f}" if fps > 0 else None
        )
    
    with col2:
        frames = status.get("frames_processed", 0)
        st.metric(
            label="🎞️ Frames",
            value=f"{frames:,}"
        )
    
    with col3:
        runtime = status.get("runtime_seconds", 0)
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        st.metric(
            label="⏱️ Runtime",
            value=f"{hours:02d}:{minutes:02d}"
        )
    
    with col4:
        anomalies = stats.get("anomaly_stats", {}).get("total_anomalies_detected", 0)
        recent_anomalies = len([e for e in recent_events if e.get("type", "").startswith("anomaly")])
        st.metric(
            label="⚠️ Anomalies",
            value=str(anomalies),
            delta=f"+{recent_anomalies}" if recent_anomalies > 0 else None
        )
    
    with col5:
        faces = face_db.get("total_faces", 0)
        st.metric(
            label="👤 Known Faces",
            value=str(faces)
        )
    
    st.divider()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Monitoring", 
        "🚨 Recent Events", 
        "👥 Face Database",
        "📈 Analytics",
        "🎥 Live Feed"
    ])
    
    with tab1:
        display_live_monitoring(status, stats)
    
    with tab2:
        display_recent_events(recent_events)
    
    with tab3:
        display_face_database(face_db)
    
    with tab4:
        display_analytics(stats, recent_events)
    
    with tab5:
        display_live_feed()

def display_stopped_dashboard():
    """Display dashboard when system is stopped."""
    st.info("🔴 Surveillance system is currently stopped")
    st.markdown("""
    ### Quick Start Guide:
    1. Click **Start** in the sidebar to begin surveillance
    2. Adjust settings as needed (motion threshold, confidence, etc.)
    3. Monitor live feed and events in real-time
    4. Manage known faces in the Face Database tab
    """)
    
    # Show offline statistics if available
    st.subheader("📊 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**GPU Support**: Available" if status.get("gpu_available") else "**GPU Support**: Not available")
    
    with col2:
        st.info("**Camera**: Ready")
    
    with col3:
        st.info("**Storage**: Ready")

def display_live_monitoring(status: Dict, stats: Dict):
    """Display live monitoring tab."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 System Performance")
        
        # Performance metrics
        motion_stats = stats.get("motion_stats", {})
        face_stats = stats.get("face_stats", {})
        anomaly_stats = stats.get("anomaly_stats", {})
        
        # Create performance chart
        perf_data = {
            "Metric": ["Motion Detection", "Face Detection", "Anomaly Detection"],
            "FPS": [
                motion_stats.get("fps", 0),
                face_stats.get("detection_fps", 0),
                anomaly_stats.get("detection_fps", 0)
            ],
            "Avg Time (ms)": [
                motion_stats.get("average_processing_time_ms", 0),
                face_stats.get("average_detection_time_ms", 0),
                anomaly_stats.get("average_detection_time_ms", 0)
            ]
        }
        
        df_perf = pd.DataFrame(perf_data)
        
        fig = px.bar(
            df_perf, 
            x="Metric", 
            y="FPS", 
            title="Processing Performance",
            color="FPS",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("🎯 Detection Summary")
        
        # Detection counts
        motion_events = motion_stats.get("average_events_per_frame", 0)
        face_detections = face_stats.get("total_detections", 0)
        total_anomalies = anomaly_stats.get("total_anomalies_detected", 0)
        
        # Progress bars for activity levels
        st.markdown("**Motion Activity**")
        st.progress(min(motion_events / 10, 1.0))
        st.caption(f"{motion_events:.2f} events/frame")
        
        st.markdown("**Face Detections**")
        face_rate = min(face_detections / 1000, 1.0) if face_detections > 0 else 0
        st.progress(face_rate)
        st.caption(f"{face_detections} total detections")
        
        st.markdown("**Anomaly Rate**")
        anomaly_rate = min(total_anomalies / 100, 1.0) if total_anomalies > 0 else 0
        st.progress(anomaly_rate)
        st.caption(f"{total_anomalies} total anomalies")
        
        # System health indicators
        st.subheader("💊 System Health")
        
        gpu_status = "🟢 Available" if status.get("gpu_available") else "🟡 Not Available"
        st.markdown(f"**GPU**: {gpu_status}")
        
        learning_status = "🟢 Active" if status.get("learning_mode") else "🔴 Disabled"
        st.markdown(f"**Learning**: {learning_status}")
        
        camera_status = "🟢 Connected" if status.get("status") == "running" else "🔴 Disconnected"
        st.markdown(f"**Camera**: {camera_status}")

def display_recent_events(events: List[Dict]):
    """Display recent events tab."""
    
    st.subheader("🚨 Recent Events")
    
    if not events:
        st.info("No recent events to display")
        return
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        event_types = list(set(e.get("type", "unknown") for e in events))
        selected_types = st.multiselect("Event Types", event_types, default=event_types)
    
    with col2:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)
    
    with col3:
        max_events = st.number_input("Max Events", 1, 100, 20)
    
    # Filter events
    filtered_events = [
        e for e in events 
        if e.get("type", "unknown") in selected_types
        and e.get("confidence", 0) >= min_confidence
    ][:max_events]
    
    # Display events
    for i, event in enumerate(filtered_events):
        event_type = event.get("type", "Unknown")
        confidence = event.get("confidence", 0)
        timestamp = event.get("timestamp", "")
        description = event.get("description", "No description")
        
        # Color code by confidence
        if confidence >= 0.8:
            alert_class = "alert-high"
        elif confidence >= 0.5:
            alert_class = "alert-medium"
        else:
            alert_class = "alert-low"
        
        st.markdown(f"""
        <div class="{alert_class}">
            <strong>{event_type}</strong> - Confidence: {confidence:.2f}<br>
            <small>{timestamp}</small><br>
            {description}
        </div>
        """, unsafe_allow_html=True)
    
    # Events timeline chart
    if filtered_events:
        st.subheader("📈 Events Timeline")
        
        # Convert to DataFrame for plotting
        df_events = pd.DataFrame(filtered_events)
        if 'timestamp' in df_events.columns:
            df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
            
            # Group by hour and type
            df_hourly = df_events.groupby([
                df_events['timestamp'].dt.floor('H'), 
                'type'
            ]).size().reset_index(name='count')
            
            fig = px.bar(
                df_hourly,
                x='timestamp',
                y='count',
                color='type',
                title="Events Over Time",
                labels={'timestamp': 'Time', 'count': 'Event Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

def display_face_database(face_db: Dict):
    """Display face database tab."""
    
    st.subheader("👥 Face Database")
    
    faces = face_db.get("faces", [])
    total_faces = face_db.get("total_faces", 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Faces", total_faces)
    with col2:
        active_faces = len([f for f in faces if f.get("recognition_count", 0) > 0])
        st.metric("Active Faces", active_faces)
    with col3:
        avg_recognitions = sum(f.get("recognition_count", 0) for f in faces) / len(faces) if faces else 0
        st.metric("Avg Recognitions", f"{avg_recognitions:.1f}")
    
    if faces:
        # Face database table
        df_faces = pd.DataFrame(faces)
        
        # Sort by recognition count
        df_faces = df_faces.sort_values("recognition_count", ascending=False)
        
        # Display table
        st.dataframe(
            df_faces[["name", "recognition_count", "last_seen", "added_date"]],
            use_container_width=True
        )
        
        # Recognition distribution chart
        if len(faces) > 1:
            fig = px.pie(
                df_faces.head(10),
                values="recognition_count",
                names="name",
                title="Top 10 Most Recognized Faces"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No faces in database. Add face images to the face_gallery folder.")

def display_analytics(stats: Dict, events: List[Dict]):
    """Display analytics tab."""
    
    st.subheader("📈 System Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly types distribution
        anomaly_events = [e for e in events if e.get("type", "").startswith("anomaly")]
        if anomaly_events:
            anomaly_types = [e.get("type", "").replace("anomaly_", "") for e in anomaly_events]
            anomaly_counts = pd.Series(anomaly_types).value_counts()
            
            fig = px.pie(
                values=anomaly_counts.values,
                names=anomaly_counts.index,
                title="Anomaly Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomaly data available")
    
    with col2:
        # Performance trends
        motion_stats = stats.get("motion_stats", {})
        face_stats = stats.get("face_stats", {})
        
        performance_data = {
            "Component": ["Motion Detection", "Face Detection", "Face Recognition"],
            "Processing Time (ms)": [
                motion_stats.get("average_processing_time_ms", 0),
                face_stats.get("average_detection_time_ms", 0),
                face_stats.get("average_encoding_time_ms", 0)
            ]
        }
        
        df_performance = pd.DataFrame(performance_data)
        
        fig = px.bar(
            df_performance,
            x="Component",
            y="Processing Time (ms)",
            title="Average Processing Times"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # System resource usage (if available)
    st.subheader("🔧 Resource Usage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CPU usage placeholder
        cpu_usage = 0.6  # This would come from actual system monitoring
        st.metric("CPU Usage", f"{cpu_usage*100:.1f}%")
        st.progress(cpu_usage)
    
    with col2:
        # Memory usage placeholder
        memory_usage = 0.4
        st.metric("Memory Usage", f"{memory_usage*100:.1f}%")
        st.progress(memory_usage)
    
    with col3:
        # GPU usage placeholder (if available)
        gpu_usage = 0.3 if stats.get("gpu_available") else 0
        st.metric("GPU Usage", f"{gpu_usage*100:.1f}%" if gpu_usage > 0 else "N/A")
        if gpu_usage > 0:
            st.progress(gpu_usage)

def display_live_feed():
    """Display live camera feed tab."""
    
    st.subheader("🎥 Live Camera Feed")
    st.info("Live feed display requires additional implementation for streaming video frames")
    
    # Placeholder for live feed
    # In a real implementation, you would:
    # 1. Set up WebSocket or WebRTC connection
    # 2. Stream frames from the surveillance system
    # 3. Display them in real-time
    
    st.markdown("""
    **Note**: To implement live feed display, you would need to:
    - Add video streaming endpoint to the API
    - Use WebSocket or WebRTC for real-time video transmission
    - Implement client-side video decoding and display
    
    For now, you can view the camera feed directly from the main surveillance application.
    """)
    
    # Show camera information
    camera_info = api.get_status().get("camera_info", {})
    if camera_info:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Resolution", f"{camera_info.get('width', 0)}x{camera_info.get('height', 0)}")
        
        with col2:
            st.metric("Camera FPS", f"{camera_info.get('fps', 0)}")
        
        with col3:
            st.metric("Processing FPS", f"{camera_info.get('processing_fps', 0):.1f}")

if __name__ == "__main__":
    main()