import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import mediapipe as mp

#-------------------------------------------------
# Fall Detection System using MediaPipe Pose
# For L&T Oman RnD - Alternative to OpenPose
#-------------------------------------------------

#---------------------
# USER CONFIGURATION
#---------------------
# Video source
VIDEO_SOURCE = "test_vdos/personfalling.mp4"

# Detection parameters
FALL_ANGLE_THRESHOLD = 70  # Larger angle = less sensitive (degrees)
HEAD_DROP_THRESHOLD = 0.15  # Head vertical position change threshold
STABILITY_FRAMES = 4  # Number of consecutive frames to confirm fall
CONFIDENCE_THRESHOLD = 0.5  # Minimum keypoint confidence

# Display settings
DISPLAY_WIDTH = 800  # Width of displayed window
SHOW_ANGLES = True  # Show angle measurements on screen
SAVE_DETECTIONS = True  # Save frames when falls are detected
DRAW_SKELETON = True  # Draw the skeleton pose
SAVE_OUTPUT_VIDEO = True  # Save the processed video
HEADLESS_MODE = True  # Set to True if running in a headless environment
#---------------------

# Print system info
print("\n===== Fall Detection System (MediaPipe) =====")
print("Running with the following settings:")
print(f"- Detection threshold: {FALL_ANGLE_THRESHOLD}° angle")
print(f"- Head drop threshold: {HEAD_DROP_THRESHOLD}")
print(f"- Stability frames: {STABILITY_FRAMES}")
print(f"- Headless mode: {HEADLESS_MODE}")
print(f"- Save output video: {SAVE_OUTPUT_VIDEO}")

# Create output directory for results if needed
os.makedirs("output", exist_ok=True)

# Initialize MediaPipe pose
print("Initializing MediaPipe Pose...")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open video source
print(f"Opening video: {VIDEO_SOURCE}")
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_SOURCE}")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} total frames")

# Initialize output video writer if requested
output_video = None
if SAVE_OUTPUT_VIDEO:
    output_path = "output/mediapipe_fall_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Output video will be saved to: {output_path}")

# Initialize variables
font = cv2.FONT_HERSHEY_SIMPLEX
frame_count = 0
process_count = 0
start_time = time.time()
fall_frames = 0
fall_detected = False
last_detection_time = 0
head_position_history = []  # Store head position history
torso_angle_history = []  # Store torso angles for smoothing

# Create window if not in headless mode
if not HEADLESS_MODE:
    cv2.namedWindow("Fall Detection (MediaPipe)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fall Detection (MediaPipe)", DISPLAY_WIDTH, int(DISPLAY_WIDTH * (height/width)))

def calculate_angle(a, b, c):
    """Calculate angle between three points in degrees"""
    if None in (a, b, c):
        return None
        
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
        
    ba = a - b
    bc = c - b
    
    # Calculate cosine of angle using dot product
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    # Avoid division by zero
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None
        
    cosine_angle = dot_product / (norm_ba * norm_bc)
    # Clip to handle numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def vector_angle_with_vertical(pt1, pt2):
    """Calculate angle between vector (pt1->pt2) and vertical axis"""
    vertical = np.array([0, -1])  # Image Y-axis points down
    
    vector = np.array([pt2.x - pt1.x, pt2.y - pt1.y])
    
    # Normalize vectors
    unit_vector = vector / (np.linalg.norm(vector) + 1e-6)
    
    # Calculate dot product and angle
    dot_product = np.dot(unit_vector, vertical)
    angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    
    return angle

def detect_fall_mediapipe(results, frame_height):
    """Detect if a person has fallen using MediaPipe pose landmarks"""
    if not results.pose_landmarks:
        return False, 0, 0
    
    landmarks = results.pose_landmarks.landmark
    
    # Get key landmarks
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    
    # Track head position (normalized by frame height)
    if nose.visibility > CONFIDENCE_THRESHOLD:
        head_position = nose.y
        head_position_history.append(head_position)
        if len(head_position_history) > 10:
            head_position_history.pop(0)
    
    # Calculate head drop (if we have enough history)
    head_drop = 0
    if len(head_position_history) > 5:
        head_drop = max(head_position_history) - min(head_position_history)
    
    # Calculate torso angle - angle between shoulders midpoint to hips midpoint vector and vertical
    torso_angle = 0
    shoulder_hip_ratio = 0
    
    if (left_shoulder.visibility > CONFIDENCE_THRESHOLD and 
        right_shoulder.visibility > CONFIDENCE_THRESHOLD and
        left_hip.visibility > CONFIDENCE_THRESHOLD and
        right_hip.visibility > CONFIDENCE_THRESHOLD):
        
        # Calculate torso angle
        # We'll use the midpoint of shoulders and midpoint of hips
        shoulders_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulders_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulders_mid = type('obj', (object,), {'x': shoulders_mid_x, 'y': shoulders_mid_y})
        
        hips_mid_x = (left_hip.x + right_hip.x) / 2
        hips_mid_y = (left_hip.y + right_hip.y) / 2
        hips_mid = type('obj', (object,), {'x': hips_mid_x, 'y': hips_mid_y})
        
        # Calculate angle with vertical
        torso_angle = vector_angle_with_vertical(hips_mid, shoulders_mid)
        
        # Calculate shoulder to hip width ratio (changes during fall)
        shoulder_width = np.sqrt((right_shoulder.x - left_shoulder.x)**2 + 
                                 (right_shoulder.y - left_shoulder.y)**2)
        hip_width = np.sqrt((right_hip.x - left_hip.x)**2 + 
                            (right_hip.y - left_hip.y)**2)
        
        if hip_width > 0.01:  # Avoid division by small numbers
            shoulder_hip_ratio = shoulder_width / hip_width
        
        # Store angle for smoothing
        torso_angle_history.append(torso_angle)
        if len(torso_angle_history) > 5:
            torso_angle_history.pop(0)
    
    # Smooth the angle
    smoothed_angle = sum(torso_angle_history) / max(1, len(torso_angle_history))
    
    # Detect fall based on angle and head drop
    fall_detected = (smoothed_angle > FALL_ANGLE_THRESHOLD) or (head_drop > HEAD_DROP_THRESHOLD)
    
    return fall_detected, smoothed_angle, head_drop

# Print startup message
print("\nStarting fall detection...")
print("Processing video frames...")

# For visualization of head position
head_positions = []
frame_numbers = []

# Progress reporting interval (every 10% of total frames)
progress_interval = max(1, total_frames // 10)

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Report progress periodically
    if frame_count % progress_interval == 0:
        percent_complete = (frame_count / total_frames) * 100
        elapsed = time.time() - start_time
        estimated_total = elapsed / (frame_count / total_frames) if frame_count > 0 else 0
        remaining = max(0, estimated_total - elapsed)
        fps_actual = process_count / elapsed if elapsed > 0 else 0
        print(f"Progress: {percent_complete:.1f}% ({frame_count}/{total_frames}) - "
              f"FPS: {fps_actual:.1f} - "
              f"Time remaining: {remaining:.1f}s")
    
    # Skip frames for better performance (optional)
    if frame_count % 2 != 0:
        continue
    
    process_count += 1
    
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = pose.process(image_rgb)
    
    # Create a copy for visualization
    annotated_frame = frame.copy()
    
    # Draw the pose landmarks
    if results.pose_landmarks and DRAW_SKELETON:
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # Detect fall
    fall_in_current_frame, torso_angle, head_drop = detect_fall_mediapipe(results, frame.shape[0])
    
    # Track for visualization
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
        if nose.visibility > CONFIDENCE_THRESHOLD:
            head_positions.append(nose.y)
            frame_numbers.append(frame_count)
    
    # Add visualization of measurements
    if SHOW_ANGLES and results.pose_landmarks:
        # Show torso angle
        if len(torso_angle_history) > 0:
            cv2.putText(annotated_frame, f"Torso angle: {torso_angle:.1f}°", 
                       (10, frame.shape[0] - 90), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Show head drop
        if len(head_position_history) > 5:
            cv2.putText(annotated_frame, f"Head drop: {head_drop:.3f}", 
                       (10, frame.shape[0] - 60), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Draw status
        if fall_in_current_frame:
            status = "POTENTIAL FALL"
            color = (0, 0, 255)
        else:
            status = "NORMAL"
            color = (0, 255, 0)
            
        # Show status
        cv2.putText(annotated_frame, status, (10, 90), font, 0.7, color, 2, cv2.LINE_AA)
    
    # Fall detection temporal logic
    if fall_in_current_frame:
        fall_frames += 1
        if fall_frames >= STABILITY_FRAMES and not fall_detected:
            fall_detected = True
            last_detection_time = time.time()
            print(f"FALL DETECTED at frame {frame_count}!")
            
            # Save fall detection frame
            if SAVE_DETECTIONS:
                cv2.imwrite(f"output/mediapipe_fall_detected_frame_{frame_count}.jpg", annotated_frame)
    else:
        # Gradually reduce fall frame counter if no detection
        fall_frames = max(0, fall_frames - 1)
    
    # Reset fall detection after some time
    if fall_detected and time.time() - last_detection_time > 3.0:
        fall_detected = False
    
    # Add fall alert to frame
    if fall_detected:
        cv2.putText(annotated_frame, "FALL DETECTED!", (frame.shape[1]//4, 50), 
                   font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    
    # Display performance metrics
    elapsed_time = time.time() - start_time
    fps_actual = process_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(annotated_frame, f"FPS: {fps_actual:.1f}", (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Write frame to output video if requested
    if SAVE_OUTPUT_VIDEO and output_video is not None:
        output_video.write(annotated_frame)
    
    # Display frame if not in headless mode
    if not HEADLESS_MODE:
        # Resize and display
        display_height = int(height * (DISPLAY_WIDTH / width))
        display_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, display_height))
        cv2.imshow("Fall Detection (MediaPipe)", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Save current frame
            cv2.imwrite(f"output/mediapipe_frame_{frame_count}.jpg", annotated_frame)
            print(f"Saved frame {frame_count}")

# Clean up
cap.release()
if output_video is not None:
    output_video.release()
pose.close()
if not HEADLESS_MODE:
    cv2.destroyAllWindows()

# Print summary
elapsed_time = time.time() - start_time
print(f"\nProcessing complete!")
print(f"Processed {frame_count} frames in {elapsed_time:.1f} seconds")
print(f"Average performance: {process_count/elapsed_time:.1f} FPS")

if SAVE_OUTPUT_VIDEO:
    print(f"Output video saved to: output/mediapipe_fall_detection.mp4")

# Plot head position changes
if len(head_positions) > 10 and len(frame_numbers) > 10:
    print("Generating head position analysis graph...")
    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, head_positions, 'b-', alpha=0.7, label='Raw Position')
    
    # Apply smoothing filter
    if len(head_positions) > 10:
        window_size = min(15, len(head_positions) - 2)
        if window_size % 2 == 0:
            window_size += 1  # Must be odd
        y_smooth = savgol_filter(head_positions, window_size, 3)
        plt.plot(frame_numbers, y_smooth, 'r-', linewidth=2, label='Smoothed')
    
    plt.title('Head Vertical Position Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Y Position (0=top, 1=bottom)')
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at fall threshold
    max_y = max(head_positions)
    threshold_y = max_y - HEAD_DROP_THRESHOLD
    plt.axhline(y=threshold_y, color='g', linestyle='--', alpha=0.8, 
               label=f'Fall Threshold (drop > {HEAD_DROP_THRESHOLD})')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/head_position_analysis.png')
    print("Head position analysis graph saved to output/head_position_analysis.png")

print("\nFall detection analysis complete.")
print("Check the 'output' directory for results.") 