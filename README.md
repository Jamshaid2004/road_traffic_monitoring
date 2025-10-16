# Intelligent Vehicle Counting System using YOLOv8 + SORT Tracker + OpenCV 🚗

This project implements a **real-time vehicle detection and counting system** using **YOLOv8**, **OpenCV**, and **SORT (Simple Online and Realtime Tracking)**.  
It can detect and track multiple types of vehicles (cars, buses, trucks, motorbikes, etc.) in video footage and count how many vehicles are **entering** and **leaving** a defined **region of interest (ROI)**.

---

## 🎯 Features

✅ **YOLOv8 Object Detection** – Detects vehicles with high accuracy.  
✅ **SORT Tracker** – Assigns a unique ID to each vehicle and maintains tracking across frames.  
✅ **Line Crossing Logic** – Counts vehicles crossing predefined incoming and outgoing lines.  
✅ **Mask Support** – Applies a custom mask to ignore irrelevant areas of the video.  
✅ **Overlap Filtering** – Avoids duplicate detections using IoU (Intersection over Union) checks.  
✅ **Video Output** – Exports processed video with bounding boxes, tracking IDs, and counts.  
✅ **Customizable FPS and Regions** – Easily adjust frame rate, counting lines, and mask dimensions.

---

## ⚙️ How It Works

### 🖼️ 1. Video Input
- The system loads:
  - A **road traffic video** → `road_traffic.mp4`  
  - A **mask image** → `mask.png`

### 🧩 2. Preprocessing
- The mask is **resized** to match the video dimensions.  
- The relevant region is extracted using:
  ```python
  masked_frame = cv2.bitwise_and(img, mask)

### 🚘 3. Vehicle Detection (YOLOv8)

- YOLOv8 detects objects frame by frame.
- Only selected vehicle classes are considered:
- car, bus, truck, motorbike

### ⚠️ 4. Overlap Filtering

- Detections that overlap more than 50% with previously drawn boxes are ignored to prevent duplicates.

### 🧠 5. Tracking (SORT Algorithm)

- Each detected vehicle is assigned a unique ID using the SORT tracker.
- The ID remains consistent across frames, allowing smooth tracking.

### 🔢 6. Counting Logic

- Two virtual lines are drawn:
  - Incoming line
  - Outgoing line
- When a tracked object's center point crosses one of these lines:
- The corresponding counter (Incoming / Outgoing) is incremented.
- The line momentarily changes color for visualization.

### 🎨 7. Visualization

- Each tracked object is displayed with:
  - 🟦 Bounding Box
  - 🔢 Object ID
  - 🏷️ Label and Confidence
  - 🎯 Colored Center Point
  - 📊 Live Incoming/Outgoing Count on the frame

### 💾 8. Video Output

- The processed video is exported as: → `road_traffic_output.mp4`
