# MySQL Quick Setup - Root Credentials

## Default Configuration

The system is now configured to use MySQL with these default credentials:

- **Host**: localhost
- **Port**: 3306
- **User**: root
- **Password**: root
- **Database**: attendance_db

## Quick Setup Steps

### 1. Install MySQL (if not installed)

**Windows:**
- Download from: https://dev.mysql.com/downloads/installer/
- Install MySQL Server
- Set root password to: `root` (or update config if different)

**Linux:**
```bash
sudo apt-get update
sudo apt-get install mysql-server
sudo mysql_secure_installation
# Set root password to: root
```

### 2. Create Database

Connect to MySQL:
```bash
mysql -u root -p
# Enter password: root
```

Create database:
```sql
CREATE DATABASE attendance_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EXIT;
```

### 3. Install Python MySQL Connector

```bash
pip install mysql-connector-python
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 4. Run the System

The system will automatically:
- Connect to MySQL using root/root credentials
- Create tables if they don't exist
- Start logging attendance

```bash
python main.py
```

## Verify Connection

Test the connection:
```python
from attendance.storage import AttendanceRepository

repo = AttendanceRepository(
    database_type="mysql",
    mysql_config={
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "root",
        "database": "attendance_db"
    }
)

employees = repo.list_employees()
print(f"Connected! Found {len(employees)} employees")
```

## Change Credentials

If your MySQL root password is different, set environment variable:

```bash
# Windows
set MYSQL_PASSWORD=your_password

# Linux/Mac
export MYSQL_PASSWORD=your_password
```

Or edit `utils/config.py`:
```python
mysql_password: str = "your_password"
```

## Troubleshooting

**Connection Error:**
- Check MySQL is running: `sudo systemctl status mysql` (Linux)
- Verify credentials: `mysql -u root -p`
- Check database exists: `SHOW DATABASES;`

**Module Not Found:**
```bash
pip install mysql-connector-python
```

**Access Denied:**
- Verify root password is correct
- Check MySQL user permissions

## Default Settings Summary

```python
database_type = "mysql"
mysql_host = "localhost"
mysql_port = 3306
mysql_user = "root"
mysql_password = "root"
mysql_database = "attendance_db"
```

The system is ready to use with MySQL!

