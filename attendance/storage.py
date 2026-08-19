"""
Unified database storage for attendance and employee metadata.
Supports both SQLite and MySQL databases.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

# Try to import database drivers
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    logger.warning("SQLite not available")

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logger.warning("MySQL connector not available. Install with: pip install mysql-connector-python")


class AttendanceRepository:
    """Thread-safe repository for attendance logs supporting SQLite and MySQL."""

    def __init__(self, database_path: str = None, database_type: str = "mysql", 
                 mysql_config: Dict[str, Any] = None):
        """
        Initialize attendance repository.
        
        Args:
            database_path: Path for SQLite database (if using SQLite)
            database_type: "sqlite" or "mysql"
            mysql_config: MySQL connection config dict with keys: host, port, user, password, database
        """
        self.database_type = database_type.lower()
        self._lock = RLock()
        
        if self.database_type == "mysql":
            if not MYSQL_AVAILABLE:
                raise ImportError("MySQL connector not available. Install with: pip install mysql-connector-python")
            
            # MySQL configuration
            mysql_config = mysql_config or {}
            self.mysql_host = mysql_config.get("host", "localhost")
            self.mysql_port = mysql_config.get("port", 3306)
            self.mysql_user = mysql_config.get("user", "root")
            self.mysql_password = mysql_config.get("password", "root")
            self.mysql_database = mysql_config.get("database", "attendance_db")
            self.connection = None
            
            logger.info(f"Using MySQL database: {self.mysql_user}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}")
            
        elif self.database_type == "sqlite":
            if not SQLITE_AVAILABLE:
                raise ImportError("SQLite not available")
            
            # SQLite configuration
            self.database_path = Path(database_path or "attendance_reports/attendance.db")
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Using SQLite database: {self.database_path}")
        else:
            raise ValueError(f"Unsupported database type: {database_type}. Use 'sqlite' or 'mysql'")
        
        self._initialize()

    def _get_mysql_connection(self):
        """Get MySQL database connection."""
        try:
            if self.connection is None or not self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    host=self.mysql_host,
                    port=self.mysql_port,
                    user=self.mysql_user,
                    password=self.mysql_password,
                    database=self.mysql_database,
                    autocommit=False,
                    charset='utf8mb4',
                    collation='utf8mb4_unicode_ci',
                    connection_timeout=5
                )
            return self.connection
        except MySQLError as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return None

    def _get_sqlite_connection(self):
        """Get SQLite database connection."""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _connect(self):
        """Get database connection based on type."""
        if self.database_type == "mysql":
            return self._get_mysql_connection()
        else:
            return self._get_sqlite_connection()

    def _execute(self, query: str, params: tuple = None, fetch: bool = False):
        """Execute query with proper error handling."""
        try:
            conn = self._connect()
            if conn is None:
                return None
            
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                result = cursor.fetchall()
                cursor.close()
                if self.database_type == "sqlite":
                    conn.commit()
                    conn.close()
                return result
            else:
                if self.database_type == "mysql":
                    conn.commit()
                else:
                    conn.commit()
                    conn.close()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"Database error: {e}")
            if self.database_type == "mysql" and conn:
                conn.rollback()
            elif self.database_type == "sqlite" and conn:
                conn.rollback()
            return None

    def _initialize(self):
        """Initialize database with proper schema."""
        if self.database_type == "mysql":
            self._initialize_mysql()
        else:
            self._initialize_sqlite()

    def _initialize_mysql(self):
        """Initialize MySQL database schema."""
        conn = self._get_mysql_connection()
        if conn is None:
            logger.error("Failed to connect to MySQL for initialization")
            return
        
        try:
            cursor = conn.cursor()
            
            # Create employees table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id VARCHAR(64) PRIMARY KEY,
                    employee_name VARCHAR(128),
                    image_path TEXT,
                    metadata TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create attendance_logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    employee_id VARCHAR(64),
                    employee_name VARCHAR(128),
                    timestamp DATETIME,
                    visit_type VARCHAR(20),
                    visit_count INT DEFAULT 1,
                    confidence FLOAT,
                    metadata TEXT,
                    INDEX idx_employee_id (employee_id),
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_employee_timestamp (employee_id, timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            conn.commit()
            cursor.close()
            logger.info("MySQL database initialized successfully")
            
        except MySQLError as e:
            logger.error(f"Error initializing MySQL database: {e}")
            conn.rollback()

    def _initialize_sqlite(self):
        """Initialize SQLite database schema."""
        with self._get_sqlite_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id TEXT PRIMARY KEY,
                    employee_name TEXT,
                    image_path TEXT,
                    metadata TEXT,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT,
                    employee_name TEXT,
                    timestamp TEXT,
                    visit_type TEXT,
                    visit_count INTEGER,
                    confidence REAL,
                    metadata TEXT
                )
            """)
            conn.commit()

    def record_attendance(self, record: Dict) -> None:
        """Persist a new attendance record."""
        with self._lock:
            if self.database_type == "mysql":
                self._record_attendance_mysql(record)
            else:
                self._record_attendance_sqlite(record)

    def _record_attendance_mysql(self, record: Dict) -> None:
        """Record attendance in MySQL."""
        conn = self._get_mysql_connection()
        if conn is None:
            return
        
        try:
            cursor = conn.cursor()
            
            # Insert attendance log
            cursor.execute("""
                INSERT INTO attendance_logs (
                    employee_id, employee_name, timestamp, visit_type, 
                    visit_count, confidence, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                record.get("Employee_ID"),
                record.get("Employee_Name"),
                record.get("Timestamp"),
                record.get("Visit_Type"),
                record.get("Visit_Count", 1),
                record.get("Confidence"),
                json.dumps(record, default=str),
            ))
            
            # Upsert employee
            employee_id = record.get("Employee_ID")
            if employee_id and employee_id != "Unknown":
                cursor.execute("""
                    INSERT INTO employees (employee_id, employee_name, image_path, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        employee_name = VALUES(employee_name),
                        image_path = COALESCE(VALUES(image_path), image_path),
                        metadata = VALUES(metadata),
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    employee_id,
                    record.get("Employee_Name", employee_id),
                    record.get("Image_Path"),
                    json.dumps({
                        "last_seen": record.get("Timestamp"),
                        "visit_count": record.get("Visit_Count"),
                    }, default=str),
                ))
            
            conn.commit()
            cursor.close()
            
        except MySQLError as e:
            logger.error(f"Error recording attendance in MySQL: {e}")
            conn.rollback()

    def _record_attendance_sqlite(self, record: Dict) -> None:
        """Record attendance in SQLite."""
        with self._get_sqlite_connection() as conn:
            conn.execute("""
                INSERT INTO attendance_logs (
                    employee_id, employee_name, timestamp, visit_type, 
                    visit_count, confidence, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("Employee_ID"),
                record.get("Employee_Name"),
                record.get("Timestamp"),
                record.get("Visit_Type"),
                record.get("Visit_Count", 1),
                record.get("Confidence"),
                json.dumps(record, default=str),
            ))
            
            employee_id = record.get("Employee_ID")
            if employee_id and employee_id != "Unknown":
                conn.execute("""
                    INSERT INTO employees (employee_id, employee_name, image_path, metadata)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(employee_id) DO UPDATE SET
                        employee_name=excluded.employee_name,
                        image_path=COALESCE(excluded.image_path, employees.image_path),
                        metadata=excluded.metadata,
                        updated_at=CURRENT_TIMESTAMP
                """, (
                    employee_id,
                    record.get("Employee_Name", employee_id),
                    record.get("Image_Path"),
                    json.dumps({
                        "last_seen": record.get("Timestamp"),
                        "visit_count": record.get("Visit_Count"),
                    }, default=str),
                ))
            conn.commit()

    def upsert_employee(self, employee_id: str, employee_name: str, image_path: Optional[str] = None):
        """Upsert employee information."""
        with self._lock:
            if self.database_type == "mysql":
                conn = self._get_mysql_connection()
                if conn:
                    try:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO employees (employee_id, employee_name, image_path, metadata)
                            VALUES (%s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                employee_name = VALUES(employee_name),
                                image_path = COALESCE(VALUES(image_path), image_path),
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            employee_id,
                            employee_name,
                            image_path,
                            json.dumps({"manual_update": datetime.utcnow().isoformat()}),
                        ))
                        conn.commit()
                        cursor.close()
                    except MySQLError as e:
                        logger.error(f"Error upserting employee in MySQL: {e}")
                        conn.rollback()
            else:
                with self._get_sqlite_connection() as conn:
                    conn.execute("""
                        INSERT INTO employees (employee_id, employee_name, image_path, metadata)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(employee_id) DO UPDATE SET
                            employee_name=excluded.employee_name,
                            image_path=COALESCE(excluded.image_path, employees.image_path),
                            updated_at=CURRENT_TIMESTAMP
                    """, (
                        employee_id,
                        employee_name,
                        image_path,
                        json.dumps({"manual_update": datetime.utcnow().isoformat()}),
                    ))
                    conn.commit()

    def list_employees(self) -> List[Dict]:
        """List all employees."""
        if self.database_type == "mysql":
            conn = self._get_mysql_connection()
            if conn:
                try:
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute("""
                        SELECT employee_id, employee_name, image_path, 
                               added_at, updated_at
                        FROM employees ORDER BY employee_name
                    """)
                    result = cursor.fetchall()
                    cursor.close()
                    return result
                except MySQLError as e:
                    logger.error(f"Error listing employees from MySQL: {e}")
                    return []
            return []
        else:
            with self._get_sqlite_connection() as conn:
                rows = conn.execute("""
                    SELECT employee_id, employee_name, image_path, added_at, updated_at
                    FROM employees ORDER BY employee_name
                """).fetchall()
                return [dict(row) for row in rows]

    def get_recent_logs(self, limit: int = 100, employee_id: Optional[str] = None) -> List[Dict]:
        """Get recent attendance logs."""
        if self.database_type == "mysql":
            conn = self._get_mysql_connection()
            if conn:
                try:
                    cursor = conn.cursor(dictionary=True)
                    if employee_id:
                        cursor.execute("""
                            SELECT employee_id, employee_name, timestamp, 
                                   visit_type, visit_count, confidence
                            FROM attendance_logs
                            WHERE employee_id = %s
                            ORDER BY timestamp DESC LIMIT %s
                        """, (employee_id, limit))
                    else:
                        cursor.execute("""
                            SELECT employee_id, employee_name, timestamp, 
                                   visit_type, visit_count, confidence
                            FROM attendance_logs
                            ORDER BY timestamp DESC LIMIT %s
                        """, (limit,))
                    result = cursor.fetchall()
                    cursor.close()
                    return result
                except MySQLError as e:
                    logger.error(f"Error getting recent logs from MySQL: {e}")
                    return []
            return []
        else:
            query = """
                SELECT employee_id, employee_name, timestamp, visit_type, visit_count, confidence
                FROM attendance_logs
            """
            params: List = []
            if employee_id:
                query += " WHERE employee_id = ?"
                params.append(employee_id)
            query += " ORDER BY datetime(timestamp) DESC LIMIT ?"
            params.append(limit)

            with self._get_sqlite_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [dict(row) for row in rows]

    def fetch_records(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Fetch attendance records within date range."""
        if self.database_type == "mysql":
            conn = self._get_mysql_connection()
            if conn:
                try:
                    cursor = conn.cursor(dictionary=True)
                    query = """
                        SELECT employee_id, employee_name, timestamp, 
                               visit_type, visit_count, confidence
                        FROM attendance_logs
                        WHERE 1=1
                    """
                    params = []
                    if start_date:
                        query += " AND DATE(timestamp) >= %s"
                        params.append(start_date)
                    if end_date:
                        query += " AND DATE(timestamp) <= %s"
                        params.append(end_date)
                    query += " ORDER BY timestamp ASC"
                    
                    cursor.execute(query, params)
                    result = cursor.fetchall()
                    cursor.close()
                    return result
                except MySQLError as e:
                    logger.error(f"Error fetching records from MySQL: {e}")
                    return []
            return []
        else:
            query = """
                SELECT employee_id, employee_name, timestamp, visit_type, visit_count, confidence
                FROM attendance_logs
                WHERE 1=1
            """
            params: List = []
            if start_date:
                query += " AND date(timestamp) >= date(?)"
                params.append(start_date)
            if end_date:
                query += " AND date(timestamp) <= date(?)"
                params.append(end_date)
            query += " ORDER BY datetime(timestamp) ASC"

            with self._get_sqlite_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [dict(row) for row in rows]

    def get_summary(self, hours: int = 24) -> Dict:
        """Get attendance summary for specified hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        
        if self.database_type == "mysql":
            conn = self._get_mysql_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT COUNT(*) FROM attendance_logs 
                        WHERE timestamp >= %s
                    """, (cutoff_iso,))
                    total = cursor.fetchone()[0]
                    
                    cursor.execute("""
                        SELECT COUNT(DISTINCT employee_id) FROM attendance_logs 
                        WHERE timestamp >= %s
                    """, (cutoff_iso,))
                    unique = cursor.fetchone()[0]
                    cursor.close()
                    
                    return {
                        "time_window_hours": hours,
                        "total_events": total,
                        "unique_employees": unique,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                except MySQLError as e:
                    logger.error(f"Error getting summary from MySQL: {e}")
                    return {
                        "time_window_hours": hours,
                        "total_events": 0,
                        "unique_employees": 0,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
            return {
                "time_window_hours": hours,
                "total_events": 0,
                "unique_employees": 0,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            with self._get_sqlite_connection() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM attendance_logs WHERE timestamp >= ?",
                    (cutoff_iso,),
                ).fetchone()[0]
                unique = conn.execute(
                    "SELECT COUNT(DISTINCT employee_id) FROM attendance_logs WHERE timestamp >= ?",
                    (cutoff_iso,),
                ).fetchone()[0]
            return {
                "time_window_hours": hours,
                "total_events": total,
                "unique_employees": unique,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    def close(self):
        """Close database connections."""
        if self.database_type == "mysql" and self.connection and self.connection.is_connected():
            self.connection.close()
            self.connection = None
            logger.info("MySQL connection closed")
