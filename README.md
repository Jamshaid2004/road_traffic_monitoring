🚗 Intelligent Vehicle Counting System using YOLOv8 + SORT Tracker + OpenCV

This project implements a real-time vehicle detection and counting system using YOLOv8, OpenCV, and SORT (Simple Online and Realtime Tracking).
It can detect and track multiple types of vehicles (cars, buses, trucks, motorbikes, etc.) in video footage and count how many vehicles are entering and leaving a defined region of interest (ROI).

🎯 Features

✅ YOLOv8 Object Detection – Detects vehicles with high accuracy.
✅ SORT Tracker – Assigns a unique ID to each vehicle and maintains tracking across frames.
✅ Line Crossing Logic – Counts vehicles crossing pre-defined incoming and outgoing lines.
✅ Mask Support – Applies a custom mask to ignore irrelevant areas of the video.
✅ Overlap Filtering – Avoids duplicate detections using IoU (Intersection over Union) checks.
✅ Video Output – Exports processed video with bounding boxes, tracking IDs, and counts.
✅ Customizable FPS and Regions – Easily adjust frame rate, counting lines, and mask dimensions.

⚙️ How It Works

Video Input:
The system loads a road traffic video (road_traffic.mp4) and a mask image (mask.png).

Preprocessing:
The mask is resized to match the video dimensions.
The relevant region is extracted using cv2.bitwise_and().

Vehicle Detection (YOLOv8):
YOLO detects objects frame-by-frame.
Only vehicle classes (car, bus, truck, motorbike) are considered.

Overlap Filtering:
Detections overlapping more than 50% with previously detected boxes are ignored to prevent duplicates.

Tracking (SORT Algorithm):
The system tracks detected objects across frames and assigns a unique ID to each.

Counting Logic:
When a tracked object’s center crosses the incoming or outgoing line,
it increments the respective count and visually highlights the line.

Visualization:
Each tracked object is displayed with:

Bounding box

Object ID

Label and confidence

Colored center point

Live incoming/outgoing count on the frame

Video Output:
The processed video is saved as road_traffic_output.mp4.
