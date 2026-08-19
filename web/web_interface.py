#!/usr/bin/env python3
"""
Streamlit web interface for AI Surveillance System.
Connects to the surveillance API for status, control, and attendance data.
"""
import os
import time
import requests
import streamlit as st

API_URL = os.environ.get("SURVEILLANCE_API_URL", "http://localhost:8080")


def fetch_json(endpoint: str, method: str = "GET", json_data: dict = None) -> dict | None:
    """Fetch JSON from API endpoint."""
    url = f"{API_URL.rstrip('/')}{endpoint}"
    try:
        resp = requests.request(
            method,
            url,
            json=json_data,
            timeout=10,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


def main():
    st.set_page_config(
        page_title="AI Surveillance System",
        page_icon="📹",
        layout="wide",
    )

    st.title("🔍 AI-Powered Surveillance System")
    st.caption(f"API: {API_URL}")

    # Sidebar controls
    with st.sidebar:
        st.header("System Controls")

        status = fetch_json("/status")
        running = status.get("status") == "running" if status else False

        action = "stop" if running else "start"
        label = "Stop System" if running else "Start System"
        if st.button(label):
            result = fetch_json("/control", method="POST", json_data={"action": action})
            if result:
                st.success(result.get("message", "OK"))
            st.rerun()

        if st.button("Reset Detectors"):
            result = fetch_json("/control", method="POST", json_data={"action": "reset"})
            if result:
                st.info(result.get("message", "Reset"))
            st.rerun()

        if st.button("Restart System"):
            result = fetch_json("/control", method="POST", json_data={"action": "restart"})
            if result:
                st.warning(result.get("message", "Restarting"))
            st.rerun()

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("System Status")
        status = fetch_json("/status")
        if status:
            running = status.get("status") == "running"
            if running:
                st.success("🟢 System Running")
                st.metric("FPS", f"{status.get('current_fps', 0):.1f}")
                st.metric("Frames Processed", status.get("frames_processed", 0))
                st.metric("Runtime (s)", f"{status.get('runtime_seconds', 0):.0f}")
            else:
                st.error("🔴 System Stopped")

            gpu = status.get("gpu", {})
            if gpu.get("available"):
                st.metric("GPU", gpu.get("device") or gpu.get("device_name", "N/A"))
            else:
                st.metric("GPU", "Not available")

            # Attendance summary
            att = status.get("attendance_summary", {})
            if att:
                st.subheader("Attendance Today")
                st.metric("Unique", att.get("unique_today", 0))
                st.metric("Total Logins", att.get("total_today", 0))
        else:
            st.error("Could not reach surveillance API. Ensure the API server is running.")

    with col2:
        st.header("Attendance Logs")
        logs = fetch_json("/attendance/logs?limit=10")
        if logs and isinstance(logs, dict) and "results" in logs:
            for log in logs.get("results", [])[:5]:
                name = log.get("employee_name", log.get("employee_id", "Unknown"))
                dt = log.get("timestamp", "")[:19] if log.get("timestamp") else ""
                st.write(f"**{name}** @ {dt}")
        else:
            st.caption("No recent logs")

        st.header("Health")
        health = fetch_json("/health")
        if health:
            st.success("API healthy" if health.get("status") == "healthy" else "API degraded")
            st.caption(f"Uptime: {health.get('uptime_seconds', 0):.0f}s")
        else:
            st.error("API unhealthy")

    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    main()
