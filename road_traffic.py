import cvzone
import cv2
import math


from overlap_area_detector import *
from sort import *
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

source_path = "../Video/road_traffic.mp4"
output_path = "../Video/road_traffic_output.mp4"
mask_path = "mask.png"

source = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)

source_frame_width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
source_frame_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))

mask = cv2.imread(mask_path)
mask = cv2.resize(mask,(source_frame_width,source_frame_height))

# print(source_frame_width)
# print(source_frame_height)
# default_fps = cv2.CAP_PROP_FPS
fps = 60
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output = cv2.VideoWriter(output_path,fourcc,fps,(source_frame_width,source_frame_height))

if not source.isOpened():
    print("couldn't opened the file")
else:
    print("file is opened properly")

tracker = Sort(max_age=20,min_hits=5,iou_threshold=0.2)
tracking_results = []

limits_of_outgoing_traffic = [100,500,550,500]
limits_of_incoming_traffic = [720,500,1150,500]
total_count_of_incoming_traffic = []
total_count_of_outgoing_traffic = []

while True:
    success, img = source.read()
    if not success:
        print("no more frames available")
        break

    print("Video frame shape:", img.shape)
    print("Mask shape:", mask.shape)
    masked_frame = cv2.bitwise_and(img, mask)
    results = model(masked_frame,stream=True)

    detections = np.empty((0, 5))
    paintedBoxes = []
    for result in results:
        for box in result.boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = class_names[cls]
            if conf > 0.3 and label == "car" or label == "bus" or label == "motorbike" or label == "truck":
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if check_overlap_area((x1,y1,x2,y2),paintedBoxes):
                    print("This Box overlaps with some already existed box.")
                else:
                    w, h = x2 - x1, y2 - y1
                    bbox = x1, y1, w, h
                    cvzone.putTextRect(img,text=f'{label} {conf}',pos=(x1+10,y1-10),scale=1,thickness=1,offset=2)
                    cvzone.cornerRect(img,bbox=bbox,colorC=(255, 0, 0), l = 0, rt = 1)
                    paintedBoxes.append((x1, y1, x2, y2))
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections,currentArray))

    tracking_results = tracker.update(detections)

    for result in tracking_results:
        x1, y1, x2, y2, id = result
        w, h = x2 - x1, y2 - y1
        cvzone.putTextRect(img,f'{int(id)}',(int(max(0, x2-20)), int(max(35, y1-10))),scale=1,thickness=2,offset=10)
        cx, cy = int(x1 + w // 2), int(y1 + h // 2)
        cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        cv2.line(img,(limits_of_incoming_traffic[0],limits_of_incoming_traffic[1]),(limits_of_incoming_traffic[2],limits_of_incoming_traffic[3]),(250,250,250),2)
        cv2.line(img, (limits_of_outgoing_traffic[0], limits_of_outgoing_traffic[1]),(limits_of_outgoing_traffic[2], limits_of_outgoing_traffic[3]), (250, 250, 250), 2)

        if (limits_of_incoming_traffic[0] < cx < limits_of_incoming_traffic[2] and
                limits_of_incoming_traffic[1] - 15 < cy < limits_of_incoming_traffic[1] + 15):
            if id not in total_count_of_incoming_traffic:
                total_count_of_incoming_traffic.append(id)
                cv2.line(img, (limits_of_incoming_traffic[0], limits_of_incoming_traffic[1]),
                         (limits_of_incoming_traffic[2], limits_of_incoming_traffic[3]), (0, 255, 0), 7)

        if (limits_of_outgoing_traffic[0] < cx < limits_of_outgoing_traffic[2] and
                limits_of_outgoing_traffic[1] - 15 < cy < limits_of_outgoing_traffic[1] + 15):
            if id not in total_count_of_outgoing_traffic:
                total_count_of_outgoing_traffic.append(id)
                cv2.line(img, (limits_of_outgoing_traffic[0], limits_of_outgoing_traffic[1]),
                         (limits_of_outgoing_traffic[2], limits_of_outgoing_traffic[3]), (0, 0, 255), 7)

    cvzone.putTextRect(img, f'Incoming: {len(total_count_of_incoming_traffic)}', (50, 50), scale=1, thickness=2)
    cvzone.putTextRect(img, f'Outgoing: {len(total_count_of_outgoing_traffic)}', (50, 100), scale=1, thickness=2)

    cv2.imshow("output preview",img)
    # cv2.imshow("Image Region", masked_frame)
    output.write(img)
    cv2.waitKey(1)
source.release()
output.release()
cv2.destroyWindow("output preview")