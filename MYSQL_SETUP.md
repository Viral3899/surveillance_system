# MySQL Database Setup Guide

## Overview

The surveillance system now supports **MySQL localhost database** for attendance tracking. You can choose between SQLite (file-based) or MySQL (server-based) databases.

## Quick Start

### 1. Install MySQL

**Windows:**
- Download MySQL from https://dev.mysql.com/downloads/installer/
- Install MySQL Server
- Remember the root password you set

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install mysql-server
sudo mysql_secure_installation
```

**macOS:**
```bash
brew install mysql
brew services start mysql
```

### 2. Create Database

Connect to MySQL:
```bash
mysql -u root -p
```

Create database and user:
```sql
CREATE DATABASE attendance_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'attendance_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON attendance_db.* TO 'attendance_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### 3. Install Python MySQL Connector

```bash
pip install mysql-connector-python
```

Or if using requirements.txt:
```bash
pip install -r requirements.txt
```

### 4. Configure the System

#### Option A: Environment Variables

Create a `.env` file or set environment variables:

```bash
# Database type: "mysql" or "sqlite"
DATABASE_TYPE=mysql

# MySQL connection settings
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=attendance_db
```

#### Option B: Edit Config File

Edit `utils/config.py` or set in code:

```python
config.attendance.database_type = "mysql"
config.attendance.mysql_host = "localhost"
config.attendance.mysql_port = 3306
config.attendance.mysql_user = "root"
config.attendance.mysql_password = "your_password"
config.attendance.mysql_database = "attendance_db"
```

### 5. Run the System

```bash
python main.py
```

The system will automatically:
- Connect to MySQL
- Create tables if they don't exist
- Start logging attendance to MySQL

## Configuration Options

### Default MySQL Settings

```python
database_type: str = "mysql"  # "sqlite" or "mysql"
mysql_host: str = "localhost"
mysql_port: int = 3306
mysql_user: str = "root"
mysql_password: str = "root"
mysql_database: str = "attendance_db"
```

### Switching Between Databases

**To use MySQL:**
```python
config.attendance.database_type = "mysql"
```

**To use SQLite:**
```python
config.attendance.database_type = "sqlite"
```

## Database Schema

The system automatically creates these tables:

### `employees` Table
- `employee_id` (VARCHAR, PRIMARY KEY)
- `employee_name` (VARCHAR)
- `image_path` (TEXT)
- `metadata` (TEXT, JSON)
- `added_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

### `attendance_logs` Table
- `id` (INT, AUTO_INCREMENT, PRIMARY KEY)
- `employee_id` (VARCHAR, INDEXED)
- `employee_name` (VARCHAR)
- `timestamp` (DATETIME, INDEXED)
- `visit_type` (VARCHAR)
- `visit_count` (INT)
- `confidence` (FLOAT)
- `metadata` (TEXT, JSON)

## Verification

### Check Database Connection

```python
from attendance.storage import AttendanceRepository

repo = AttendanceRepository(
    database_type="mysql",
    mysql_config={
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "your_password",
        "database": "attendance_db"
    }
)

# Test connection
employees = repo.list_employees()
print(f"Connected! Found {len(employees)} employees")
```

### View Data in MySQL

```bash
mysql -u root -p attendance_db
```

```sql
-- View all employees
SELECT * FROM employees;

-- View recent attendance logs
SELECT * FROM attendance_logs ORDER BY timestamp DESC LIMIT 10;

-- View today's attendance
SELECT employee_name, COUNT(*) as visits 
FROM attendance_logs 
WHERE DATE(timestamp) = CURDATE() 
GROUP BY employee_name;
```

## Troubleshooting

### Connection Errors

**Error: "Access denied for user"**
- Check username and password
- Verify user has privileges: `GRANT ALL PRIVILEGES ON attendance_db.* TO 'user'@'localhost';`

**Error: "Can't connect to MySQL server"**
- Check if MySQL is running: `sudo systemctl status mysql` (Linux) or check Services (Windows)
- Verify host and port (default: localhost:3306)
- Check firewall settings

**Error: "Unknown database"**
- Create the database: `CREATE DATABASE attendance_db;`
- Verify database name in config

### Import Errors

**Error: "No module named 'mysql.connector'"**
```bash
pip install mysql-connector-python
```

### Performance

- MySQL is faster for large datasets
- Use indexes for better query performance (automatically created)
- Consider connection pooling for high-traffic scenarios

## Benefits of MySQL

✅ **Better Performance**: Faster queries for large datasets
✅ **Concurrent Access**: Multiple applications can access the database
✅ **Scalability**: Can handle more data and connections
✅ **Backup & Recovery**: Better tools for database management
✅ **Remote Access**: Can access from other machines on network
✅ **Advanced Features**: Transactions, stored procedures, triggers

## Migration from SQLite to MySQL

If you have existing SQLite data:

1. Export data from SQLite:
```python
from attendance.storage import AttendanceRepository

# SQLite repo
sqlite_repo = AttendanceRepository(database_type="sqlite", database_path="attendance.db")
employees = sqlite_repo.list_employees()
logs = sqlite_repo.fetch_records()
```

2. Import to MySQL:
```python
# MySQL repo
mysql_repo = AttendanceRepository(
    database_type="mysql",
    mysql_config={"host": "localhost", "user": "root", "password": "pass", "database": "attendance_db"}
)

# Import employees
for emp in employees:
    mysql_repo.upsert_employee(emp['employee_id'], emp['employee_name'], emp.get('image_path'))

# Import logs
for log in logs:
    mysql_repo.record_attendance({
        "Employee_ID": log['employee_id'],
        "Employee_Name": log['employee_name'],
        "Timestamp": log['timestamp'],
        "Visit_Type": log['visit_type'],
        "Visit_Count": log['visit_count'],
        "Confidence": log['confidence']
    })
```

## Security Best Practices

1. **Use Strong Passwords**: Don't use default "root" password
2. **Create Dedicated User**: Don't use root user for application
3. **Limit Privileges**: Grant only necessary permissions
4. **Use SSL**: Enable SSL for remote connections
5. **Regular Backups**: Backup database regularly
6. **Update Regularly**: Keep MySQL updated

## Example Configuration

```python
# .env file
DATABASE_TYPE=mysql
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=attendance_user
MYSQL_PASSWORD=secure_password_123
MYSQL_DATABASE=attendance_db
```

The system will automatically use these settings when you run it!

