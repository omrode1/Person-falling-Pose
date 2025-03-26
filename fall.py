from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch
import os

#-------------------------------------------------
# Fall Detection System using YOLOv8 Pose Estimation
# For L&T Oman RnD
#-------------------------------------------------

#---------------------
# USER CONFIGURATION
#---------------------
# Video source - choose a video file or set to 0 for webcam
VIDEO_SOURCE = "test_vdos/personfalling.mp4"  # Change to video file or 0 for webcam

# Detection parameters
FALL_ANGLE_THRESHOLD = 60  # Larger angle = less sensitive (degrees)
STABILITY_FRAMES = 3  # Number of consecutive frames to confirm fall
CONFIDENCE_THRESHOLD = 0.5  # Detection confidence threshold

# Performance settings
PROCESS_WIDTH = 320  # Width to resize input for processing (smaller = faster)
FRAME_SKIP = 3  # Process every Nth frame (higher = faster but less accurate)
USE_HALF_PRECISION = False  # Use FP16 for even better GPU performance (set to False if errors occur)

# Display settings
DISPLAY_WIDTH = 800  # Width of displayed window
SHOW_ANGLES = True  # Show angle measurements on screen
SAVE_DETECTIONS = True  # Save frames when falls are detected
SAVE_OUTPUT_VIDEO = True  # Save processed video output
HEADLESS_MODE = True  # Run without display windows
#---------------------

# Print system info
print("\n===== Fall Detection System =====")
print("Running with the following settings:")
print(f"- Detection threshold: {FALL_ANGLE_THRESHOLD}° angle")
print(f"- Processing resolution: {PROCESS_WIDTH}px wide")
print(f"- Frame skip: {FRAME_SKIP}")
print(f"- Headless mode: {HEADLESS_MODE}")
print(f"- Save output video: {SAVE_OUTPUT_VIDEO}")

# Ensure CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory for results if needed
os.makedirs("output", exist_ok=True)

# Load YOLOv8 pose model with GPU acceleration
print("Loading YOLOv8 pose model...")
model = YOLO("yolov8n-pose.pt").to(device)
if USE_HALF_PRECISION and device.type == 'cuda':
    try:
        model = model.half()  # Convert to half precision for faster inference
        print("Half precision enabled")
    except Exception as e:
        print(f"Could not enable half precision: {e}")
        USE_HALF_PRECISION = False
print(f"Model loaded on: {device}" + (" (half precision)" if USE_HALF_PRECISION and device.type == 'cuda' else ""))

# Open video source
print(f"Opening video: {VIDEO_SOURCE}")
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_SOURCE}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} total frames")

# Initialize output video writer if requested
output_video = None
if SAVE_OUTPUT_VIDEO:
    output_path = "output/yolov8_fall_detection.mp4"
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
angles_history = []  # Store recent angle history for smoothing

# Create window if not in headless mode
if not HEADLESS_MODE:
    cv2.namedWindow("Fall Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fall Detection", DISPLAY_WIDTH, int(DISPLAY_WIDTH * (height/width)))

def calculate_angle(pt1, pt2, pt3):
    """Calculate angle between three points in degrees"""
    if pt1 is None or pt2 is None or pt3 is None:
        return None
        
    if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)) or np.any(np.isnan(pt3)):
        return None
    
    a = np.array(pt1)
    b = np.array(pt2)
    c = np.array(pt3)
    
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

def vector_angle_with_vertical(vector):
    """Calculate angle between a vector and the vertical axis"""
    vertical = np.array([0, -1])  # Image Y-axis points down
    
    # Normalize vectors
    unit_vector = vector / (np.linalg.norm(vector) + 1e-6)
    
    # Calculate dot product and angle
    dot_product = np.dot(unit_vector, vertical)
    angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    
    return angle

@torch.no_grad()  # Decorator to disable gradient calculation for this function
def detect_fall(result, frame):
    """Detect if a person has fallen based on body orientation"""
    # Create a copy for visualization
    show_frame = frame.copy() if SHOW_ANGLES else None
    
    try:
        # Direct access to tensor data for speed
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return False, show_frame
        
        # Convert keypoints to numpy
        if hasattr(result.keypoints, 'cpu'):
            # PyTorch tensor
            keypoints_data = result.keypoints.cpu().numpy()
        else:
            # Already numpy
            keypoints_data = result.keypoints
            
        # Debug format of keypoints occasionally
        if frame_count % 300 == 0:
            print(f"Keypoints data shape: {keypoints_data.shape}")
            
        # Process detections
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            max_angle = 0
            shoulder_hip_ratio = 0.0
            
            # For each detected person
            for person_idx in range(len(result.boxes)):
                try:
                    # Try to access person's keypoints
                    if person_idx >= len(keypoints_data):
                        continue
                    
                    person_keypoints = keypoints_data[person_idx]
                    
                    # Skip if format is unexpected or there aren't enough points
                    if not isinstance(person_keypoints, np.ndarray) or person_keypoints.shape[0] < 17:
                        continue
                    
                    # Get keypoints with good confidence
                    if person_keypoints.shape[1] >= 3:
                        # Identify key body points with good confidence
                        left_shoulder = None if person_keypoints[5, 2] < CONFIDENCE_THRESHOLD else person_keypoints[5, :2]
                        right_shoulder = None if person_keypoints[6, 2] < CONFIDENCE_THRESHOLD else person_keypoints[6, :2]
                        left_hip = None if person_keypoints[11, 2] < CONFIDENCE_THRESHOLD else person_keypoints[11, :2]
                        right_hip = None if person_keypoints[12, 2] < CONFIDENCE_THRESHOLD else person_keypoints[12, :2]
                        left_knee = None if person_keypoints[13, 2] < CONFIDENCE_THRESHOLD else person_keypoints[13, :2]
                        right_knee = None if person_keypoints[14, 2] < CONFIDENCE_THRESHOLD else person_keypoints[14, :2]
                        left_ankle = None if person_keypoints[15, 2] < CONFIDENCE_THRESHOLD else person_keypoints[15, :2]
                        right_ankle = None if person_keypoints[16, 2] < CONFIDENCE_THRESHOLD else person_keypoints[16, :2]
                        
                        vertical_angles = []
                        
                        # Measure torso orientation (shoulders to hips)
                        if left_shoulder is not None and right_shoulder is not None and left_hip is not None and right_hip is not None:
                            # Calculate midpoints of shoulders and hips
                            shoulder_mid = (left_shoulder + right_shoulder) / 2
                            hip_mid = (left_hip + right_hip) / 2
                            
                            # Calculate torso vector and its angle with vertical
                            torso_vector = shoulder_mid - hip_mid
                            torso_angle = vector_angle_with_vertical(torso_vector)
                            vertical_angles.append(torso_angle)
                            
                            # Calculate width ratio (changes when person falls)
                            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
                            hip_width = np.linalg.norm(right_hip - left_hip)
                            if hip_width > 0:
                                shoulder_hip_ratio = shoulder_width / hip_width
                            
                            # Visualize on frame if requested
                            if SHOW_ANGLES:
                                # Draw torso line
                                cv2.line(show_frame, tuple(map(int, shoulder_mid)), tuple(map(int, hip_mid)), (0, 255, 0), 2)
                                # Display angle
                                cv2.putText(show_frame, f"{torso_angle:.1f}°", tuple(map(int, hip_mid + np.array([10, 0]))), 
                                           font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                        
                        # Measure lower body orientation (hips to ankles)
                        if left_hip is not None and right_hip is not None and left_ankle is not None and right_ankle is not None:
                            # Calculate midpoints
                            hip_mid = (left_hip + right_hip) / 2
                            ankle_mid = (left_ankle + right_ankle) / 2
                            
                            # Calculate lower body vector and angle
                            lower_vector = hip_mid - ankle_mid
                            lower_angle = vector_angle_with_vertical(lower_vector)
                            vertical_angles.append(lower_angle)
                            
                            # Visualize
                            if SHOW_ANGLES:
                                cv2.line(show_frame, tuple(map(int, hip_mid)), tuple(map(int, ankle_mid)), (255, 0, 0), 2)
                                cv2.putText(show_frame, f"{lower_angle:.1f}°", tuple(map(int, ankle_mid + np.array([10, 0]))), 
                                          font, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
                        
                        # If we have angles, evaluate fall detection
                        if vertical_angles:
                            current_max_angle = max(vertical_angles)
                            max_angle = max(max_angle, current_max_angle)  # Update global max
                            
                except Exception as e:
                    if frame_count % 100 == 0:
                        print(f"Error processing person {person_idx}: {e}")
                    continue
            
            # Keep a history of max angles for smoothing (reduces false positives)
            angles_history.append(max_angle)
            if len(angles_history) > 5:
                angles_history.pop(0)
            
            # Calculate smoothed angle
            smoothed_angle = sum(angles_history) / len(angles_history) if angles_history else 0
            
            # Determine if a fall is happening
            fall_happening = smoothed_angle > FALL_ANGLE_THRESHOLD
            
            # Display detection criterion
            if SHOW_ANGLES and max_angle > 0:
                cv2.putText(show_frame, f"Max angle: {max_angle:.1f}° (Threshold: {FALL_ANGLE_THRESHOLD}°)", 
                           (10, height - 60), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(show_frame, f"S/H ratio: {shoulder_hip_ratio:.2f}", 
                           (10, height - 30), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Return fall detection status
            return fall_happening, show_frame

    except Exception as e:
        if frame_count % 100 == 0:
            print(f"Error in fall detection: {e}")
        
    return False, show_frame

# Print startup message
print("\nStarting fall detection...")
if not HEADLESS_MODE:
    print("Press 'q' to quit, 's' to save current frame")
else:
    print("Processing video frames...")

# Progress reporting interval (every 10% of total frames)
progress_interval = max(1, total_frames // 10)

# Main processing loop
while cap.isOpened():
    # Skip frames if needed for better performance
    for _ in range(FRAME_SKIP):
        ret, _ = cap.read()
        if not ret:
            break
    
    # Read actual frame to process
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
    
    process_count += 1
    
    # Resize frame for faster processing
    process_height = int(height * (PROCESS_WIDTH / width))
    process_frame = cv2.resize(frame, (PROCESS_WIDTH, process_height))
    
    # Run pose detection with optimized settings
    results = model(process_frame, verbose=False)
    
    # Process detection results
    fall_in_current_frame = False
    
    if len(results) > 0:
        result = results[0]
        
        # First get the skeleton visualization with keypoints
        skeleton_frame = result.plot()
        
        # Run fall detection with angle visualization
        fall_detected_in_frame, angle_frame = detect_fall(result, frame)
        
        if fall_detected_in_frame:
            fall_in_current_frame = True
        
        # Combine visualizations: if we're showing angles, overlay skeleton on angle frame
        if SHOW_ANGLES and angle_frame is not None:
            # First resize skeleton if needed
            if skeleton_frame.shape[:2] != (height, width):
                skeleton_frame = cv2.resize(skeleton_frame, (width, height))
            
            # Simplify the approach for clearer visualization
            # Use a 50/50 blend of the angle visualization and the YOLOv8 skeleton visualization
            annotated_frame = cv2.addWeighted(angle_frame, 0.5, skeleton_frame, 0.5, 0)
            
            # Add additional information to the frame
            if fall_in_current_frame:
                status_color = (0, 0, 255)  # Red for potential fall
                status_text = "POTENTIAL FALL"
            else:
                status_color = (0, 255, 0)  # Green for normal
                status_text = "NORMAL"
                
            # Show person count
            cv2.putText(annotated_frame, f"Persons: {len(result.boxes)}", 
                       (10, 60), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                
            # Show status message
            cv2.putText(annotated_frame, status_text, 
                       (10, 90), font, 0.7, status_color, 2, cv2.LINE_AA)
        else:
            # Just use the skeleton frame
            annotated_frame = skeleton_frame
        
        # Ensure proper size
        if annotated_frame.shape[:2] != (height, width):
            annotated_frame = cv2.resize(annotated_frame, (width, height))
    else:
        # No detections
        annotated_frame = frame
    
    # Fall detection temporal logic - need multiple consecutive frames
    if fall_in_current_frame:
        fall_frames += 1
        
        # Confirm fall only after several consecutive detections
        if fall_frames >= STABILITY_FRAMES and not fall_detected:
            fall_detected = True
            last_detection_time = time.time()
            print(f"FALL DETECTED at frame {frame_count}!")
            
            # Save fall detection frame
            if SAVE_DETECTIONS:
                cv2.imwrite(f"output/fall_detected_frame_{frame_count}.jpg", annotated_frame)
    else:
        # Gradually reduce fall frame counter if no detection
        fall_frames = max(0, fall_frames - 1)
    
    # Reset fall detection after some time
    if fall_detected and time.time() - last_detection_time > 3.0:
        fall_detected = False
    
    # Add fall alert to frame
    if fall_detected:
        cv2.putText(annotated_frame, "FALL DETECTED!", (width//4, 50), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    
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
        cv2.imshow("Fall Detection", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Save current frame
            cv2.imwrite(f"output/frame_{frame_count}.jpg", annotated_frame)
            print(f"Saved frame {frame_count}")

# Clean up
cap.release()
if output_video is not None:
    output_video.release()
if not HEADLESS_MODE:
    cv2.destroyAllWindows()

# Print summary
elapsed_time = time.time() - start_time
print(f"\nProcessing complete!")
print(f"Processed {frame_count} frames in {elapsed_time:.1f} seconds")
print(f"Average performance: {process_count/elapsed_time:.1f} FPS")

if SAVE_OUTPUT_VIDEO:
    print(f"Output video saved to: output/yolov8_fall_detection.mp4")

print("\nFall detection analysis complete.")
print("Check the 'output' directory for results.")
