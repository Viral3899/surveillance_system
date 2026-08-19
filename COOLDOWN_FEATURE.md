# 30-Minute Visit Cooldown Feature

## Overview

The attendance system now implements a **30-minute cooldown period** between visits. This means:

- ✅ A person must wait **30 minutes** after their last visit before a new visit is logged
- ✅ The system will still detect and display the person, but won't count it as a new visit
- ✅ Visit count only increments after the 30-minute cooldown period has passed

## How It Works

### Visit Logging Logic

1. **First Detection**: When a person is first detected, their visit is immediately logged
2. **Subsequent Detections**: 
   - If detected within 30 minutes → **No new visit logged** (cooldown active)
   - If detected after 30 minutes → **New visit logged** (cooldown expired)

### Visual Indicators

On the display, you'll see:

- **Normal Status**: `John Doe (Visits: 3) 0.95`
  - Person detected, visit count shown
  - Ready for new visit if 30 minutes have passed

- **Cooldown Status**: `John Doe (Visits: 3, Wait: 15m 30s) 0.95`
  - Person detected but in cooldown
  - Shows remaining time until next visit can be logged
  - Visit count does NOT increment

## Configuration

The cooldown period is configured in `utils/config.py`:

```python
cooldown_seconds: int = 1800  # 30 minutes
```

To change the cooldown period, modify this value:
- 600 seconds = 10 minutes
- 1800 seconds = 30 minutes (current)
- 3600 seconds = 60 minutes

## Behavior Details

### What Happens During Cooldown

1. **Face Detection**: Still works normally
2. **Display**: Person is shown with cooldown timer
3. **Database**: No new attendance record is created
4. **Visit Count**: Does not increment
5. **Last Seen**: Still updates for display purposes

### What Happens After Cooldown

1. **New Visit Logged**: Attendance record created in database
2. **Visit Count Increments**: Counter increases by 1
3. **Cooldown Resets**: 30-minute timer starts again
4. **Display Updates**: Shows new visit count

## Example Scenarios

### Scenario 1: Normal Visit Pattern
```
10:00 AM - John detected → Visit #1 logged
10:05 AM - John detected → Cooldown active (no new visit)
10:15 AM - John detected → Cooldown active (no new visit)
10:35 AM - John detected → Visit #2 logged (30+ minutes passed)
```

### Scenario 2: Multiple People
```
10:00 AM - John detected → Visit #1 logged
10:05 AM - Jane detected → Visit #1 logged (different person)
10:10 AM - John detected → Cooldown active (John's cooldown)
10:35 AM - John detected → Visit #2 logged (John's cooldown expired)
```

## Benefits

✅ **Prevents Duplicate Logs**: Avoids logging the same person multiple times in quick succession
✅ **Accurate Visit Tracking**: Only counts meaningful visits (30+ minutes apart)
✅ **Visual Feedback**: Shows cooldown status on screen
✅ **Configurable**: Easy to adjust cooldown period

## Technical Implementation

- **Tracking**: `last_visit_times` dictionary stores last visit time per employee
- **Check**: Compares current time with last visit time
- **Cooldown**: 1800 seconds (30 minutes) default
- **Reset**: Daily reset at midnight

## Logging

The system logs:
- ✅ New visits: `"New visit logged for John Doe (Visit #2)"`
- ⏳ Cooldown: `"John detected but in cooldown (1200s remaining)"` (debug level)

## Notes

- Cooldown is per employee (each person has their own timer)
- Cooldown resets daily at midnight
- Face detection continues normally during cooldown
- Display always shows current status

