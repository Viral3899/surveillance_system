from datetime import datetime, timezone
from pathlib import Path

from attendance.storage import AttendanceRepository


def test_repository_records_and_fetches_entries(tmp_path: Path):
    repo = AttendanceRepository(tmp_path / "attendance.db")
    record = {
        "Employee_ID": "EMP001",
        "Employee_Name": "Test User",
        "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "Visit_Type": "FIRST_VISIT",
        "Visit_Count": 1,
        "Confidence": 0.95,
    }
    repo.record_attendance(record)

    logs = repo.get_recent_logs(limit=5)
    assert len(logs) == 1
    assert logs[0]["employee_id"] == "EMP001"

    summary = repo.get_summary(hours=24)
    assert summary["total_events"] == 1
    assert summary["unique_employees"] == 1
