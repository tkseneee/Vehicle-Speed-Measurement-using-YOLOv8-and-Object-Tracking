import cv2
import numpy as np
from ultralytics import YOLO

# 1. LOAD YOLOv8 MODEL AND SET VIDEO FPS

model = YOLO("yolov8n.pt")  # or your custom fine-tuned model
video_path = "traffic1.mp4"  # Replace with your actual video

cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if video_fps == 0:
    video_fps = 30.0  # fallback if FPS metadata is missing

# Define output video writer (MP4 format)
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_width, frame_height))

cap.release()


# 2. CALIBRATION & THRESHOLD SETTINGS

scale_factor = 0.0113         # meters per pixel (adjust based on your calibration)
movement_threshold = 1      # pixel threshold to catch subtle movement
confidence_threshold = 0.5  # increased to reduce false positives

# Dictionary to store previous center positions for each tracked object (by track ID)
prev_positions = {}
# Dictionary to store speed history for each track ID (for averaging over a window)
speed_history = {}
window_size = 5  # number of frames over which to average the speed

# 3. TRACK & DETECT CARS, THEN CALCULATE SPEED
for result in model.track(
        source=video_path,
        conf=confidence_threshold,  # increased confidence threshold
        stream=True):
    
    # Get the original frame
    frame = result.orig_img

    # Process each detected bounding box in the current frame
    for box in result.boxes:
        # Filter for the 'car' class (COCO class ID = 2)
        if int(box.cls) == 2:
            # Get bounding box coordinates and compute the center
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Use the built-in tracker identifier. (Try box.id if box.track_id is not working)
            track_id = int(box.id) if hasattr(box, "id") and box.id is not None else None

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if track_id is not None:
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # SPEED CALCULATION WITH SMOOTHING
            if track_id is not None:
                if track_id in prev_positions:
                    prev_cx, prev_cy = prev_positions[track_id]
                    # Calculate the pixel displacement between frames
                    d_pixels = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

                    # Compute speed if displacement exceeds threshold; otherwise, set speed to zero
                    if d_pixels > movement_threshold:
                        d_meters = d_pixels * scale_factor  # Convert pixels to meters
                        speed_m_s = d_meters * video_fps     # Convert to m/s (distance per frame * FPS)
                        speed_km_h = speed_m_s * 3.6         # Convert m/s to km/h
                    else:
                        speed_km_h = 0.0

                    # Maintain a history of speed values for the current track_id
                    if track_id in speed_history:
                        speed_history[track_id].append(speed_km_h)
                        if len(speed_history[track_id]) > window_size:
                            speed_history[track_id].pop(0)
                    else:
                        speed_history[track_id] = [speed_km_h]
                    
                    # Compute the average speed over the window
                    avg_speed = np.mean(speed_history[track_id])
                    
                    # Display the averaged speed below the bounding box
                    cv2.putText(frame, f"{avg_speed:.2f} km/h", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Update the previous position for the track_id
                prev_positions[track_id] = (cx, cy)


    # 4. SHOW FRAME & SAVE TO OUTPUT VIDEO
    cv2.imshow("Car Speed Tracker", frame)
    out.write(frame)  # Save frame to video

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
out.release()
cv2.destroyAllWindows()

print(f"Process completed. Output saved as: {output_path}")
