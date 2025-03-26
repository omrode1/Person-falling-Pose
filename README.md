# Fall Detection System

## Overview
This is a computer vision-based fall detection system to detect people falling in video streams. The system uses pose estimation algorithms to track human body keypoints and detect anomalies that indicate a fall.

## Features
- Real-time fall detection using YOLOv8 Pose estimation
- Alternative implementation using MediaPipe for pose tracking
- Multiple detection approaches available (angle-based, trajectory-based)
- Configurable detection parameters
- Visual output with skeleton overlay and angle measurements
- Performance up to 37 FPS on modern hardware
- Ability to process video files or live camera feeds
- Results visualization with head position analysis graphs

## Requirements
- Python 3.8+
- OpenCV 4.x
- PyTorch (for YOLOv8)
- MediaPipe
- Matplotlib
- NumPy
- Scipy
- Ultralytics YOLOv8

## Usage
The system offers two different implementation approaches:

### 1. YOLOv8-based Detection
```bash
python fall.py
```

### 2. MediaPipe-based Detection
```bash
python openpose.py
```

## Configuration Options
Both implementations allow for customizing detection parameters:

```python
# Detection parameters
FALL_ANGLE_THRESHOLD = 70  # Threshold in degrees
HEAD_DROP_THRESHOLD = 0.15  # Vertical position change threshold
STABILITY_FRAMES = 4  # Required consecutive detection frames
CONFIDENCE_THRESHOLD = 0.5  # Minimum pose confidence
```

Display options:
```python
# Display settings
DISPLAY_WIDTH = 800  # Output display width
SHOW_ANGLES = True  # Show angle measurements
SAVE_DETECTIONS = True  # Save frames when falls detected
DRAW_SKELETON = True  # Show pose skeleton
SAVE_OUTPUT_VIDEO = True  # Save results to video file
```

## Fall Detection Method
The system detects falls through multiple indicators:

1. **Torso Angle Analysis**: Measures the angle of the torso relative to the vertical axis. A large angle indicates the person is horizontal.
2. **Head Position Tracking**: Monitors the vertical position of the head over time. A rapid drop indicates falling.
3. **Temporal Analysis**: Requires detection criteria to be met for multiple consecutive frames to reduce false positives.

## Implementation Approaches
The project offers two different pose estimation approaches:

### YOLOv8 Implementation (fall.py)
- Uses YOLOv8 pose estimation model from Ultralytics
- Provides excellent accuracy and robust pose detection
- Requires more computational resources
- Higher precision in detecting complex poses

### MediaPipe Implementation (openpose.py)
- Utilizes Google's MediaPipe pose estimation
- Lighter weight solution with faster processing
- Works well on less powerful hardware
- Simpler to deploy with fewer dependencies

## Performance Optimization
- Frame skipping for higher processing rates
- Resolution adjustment for performance tuning
- GPU acceleration support
- Multi-threaded processing capability

## Output Analysis
The system generates visualization outputs:
- Processed video with skeleton overlay and fall detection markers
- Screenshots of detected fall events
- Head position graphs showing vertical movement over time

## Results
We have successfully implemented two different approaches for fall detection:

1. **YOLOv8 Implementation**: 
   - Achieved up to 11 FPS performance on the test video
   - Uses angle-based detection with temporal stability
   - Provides detailed skeletal visualization with angle measurements

2. **MediaPipe Implementation**:
   - Achieved up to 37 FPS performance on the test video
   - Added head position tracking for enhanced fall detection
   - Generates analytical graph showing the vertical head movement
   - Successfully detected fall at frame 24 in the test video

Both implementations successfully process the same test video (personfalling.mp4) and can work in headless environments, saving output videos for later analysis. The MediaPipe implementation provides faster processing speeds, while the YOLOv8 implementation may offer higher accuracy for complex poses and difficult scenarios.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Ultralytics for YOLOv8
- Google for MediaPipe
- OpenCV team for computer vision tools 