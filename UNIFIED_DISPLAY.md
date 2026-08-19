# Unified Single-Screen Display

The surveillance system now displays **all features in a single unified screen** when running `main.py`.

## Features Displayed

### Single Window: "AI Surveillance System - All Features"

Everything is shown in one comprehensive display:

#### 1. **Live Camera Feed** (Center)
- Real-time video stream
- Motion detection overlays (highlighted regions)
- Face detection boxes:
  - **Green boxes** = Known faces (registered employees)
  - **Red boxes** = Unknown faces
- Anomaly detection indicators
- Face labels with:
  - Employee name
  - Visit count (for known faces)
  - Recognition confidence  

#### 2. **System Status Panel** (Top-Left)
- FPS (Frames Per Second)
- Total frames processed
- Runtime (HH:MM:SS)
- GPU status (ON/OFF)
- Learning mode status
- Detection statistics:
  - Motion events count
  - Faces detected count
  - Anomalies detected count

#### 3. **Attendance Tracking Panel** (Top-Right)
- Total detections today
- Known faces count
- Unknown faces count
- Today's unique visitors
- **Top 5 Recent Visitors** with visit counts

#### 4. **Controls Panel** (Bottom-Left)
- Q - Quit System
- R - Reset Detectors
- L - Toggle Learning Mode
- S - Save Current Frame
- A - Show Attendance Details

#### 5. **Timestamp** (Bottom-Right)
- Current date and time

## How to Run

```bash
# Run with unified display (default)
python main.py

# Or specify camera
python main.py --camera 0
```

## What You'll See

When you run the system, you'll see:

1. **One window** showing everything
2. **Real-time updates** of all statistics
3. **Color-coded face detection**:
   - Green = Known employee (attendance tracked)
   - Red = Unknown person
4. **Live attendance tracking** as faces are detected
5. **Motion detection** highlighted on the video
6. **All system metrics** updated in real-time

## Attendance Integration

The system automatically:
- Detects faces in the video stream
- Matches them against registered employees
- Logs attendance when known faces are detected
- Tracks visit counts per employee
- Displays attendance statistics in real-time
- Resets daily attendance at midnight

## Display Layout

```
┌─────────────────────────────────────────────────────────────┐
│ [System Status]        [Attendance Stats]                    │
│ FPS: 30.0            Total Detections: 45                   │
│ Frames: 1500         Known Faces: 38                        │
│ Runtime: 00:05:00    Unknown: 7                             │
│ GPU: ON              Today's Visitors: 5                     │
│                      Top Visitors:                           │
│                      1. John Doe: 12 visits                 │
│                      ...                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│              LIVE CAMERA FEED                               │
│         (with motion, faces, anomalies)                     │
│                                                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ [Controls]                                    [Timestamp]   │
│ Q - Quit       2025-12-01 12:30:45                         │
│ R - Reset                                                  │
│ L - Toggle Learning                                        │
│ S - Save Frame                                             │
└─────────────────────────────────────────────────────────────┘
```

## Benefits

✅ **Single Screen** - Everything in one place
✅ **Real-time Updates** - All stats update live
✅ **Comprehensive View** - See motion, faces, attendance, and system status
✅ **Easy Monitoring** - No need to switch between windows
✅ **Professional Display** - Clean, organized layout

## Keyboard Controls

- **Q** - Quit the system
- **R** - Reset all detectors (motion, face, anomaly)
- **L** - Toggle learning mode for anomaly detection
- **S** - Save current frame as image
- **ESC** - Also quits the system

## Requirements

- Camera connected and accessible
- Employee faces registered in `faces/` directory
- Attendance module enabled (default: enabled)

## Notes

- The window is resizable - drag corners to adjust size
- All overlays scale with window size
- Statistics update every frame
- Attendance resets automatically at midnight

