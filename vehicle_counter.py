import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(r"C:\Users\mahes\Videos\cars.mp4.mp4")

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

# ---------------- MODEL ----------------
model = YOLO("../Yolo-Weights/yolov8l.pt")

# ---------------- CLASSES ----------------
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# ---------------- MASK ----------------
mask = cv2.imread("mask-950x480 (1).png")
if mask is None:
    print("❌ Error: mask.png not found.")
    exit()

# ---------------- TRACKER ----------------
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# ---------------- LINE ----------------
limits = [400, 297, 673, 297]
totalCount = []

# ---------------- LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        print("✅ Video ended or cannot read frame.")
        break

    # Resize mask to match frame
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)

    # Make mask binary (VERY IMPORTANT)
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    # Apply mask correctly
    imgRegion = cv2.bitwise_and(img, img, mask=mask_binary)

    # Optional: show masked region for debugging
    cv2.imshow("Masked Region", imgRegion)

    # ---------------- DETECTION ----------------
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Correct filtering
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                w, h = x2 - x1, y2 - y1

                # Draw detection (for debugging)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 255))
                cvzone.putTextRect(img, f'{currentClass} {conf:.2f}', (x1, y1 - 10), scale=1, thickness=1)

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # ---------------- TRACKING ----------------
    resultsTracker = tracker.update(detections)

    # Draw counting line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        w, h = x2 - x1, y2 - y1

        # Draw tracking box
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2)
        cvzone.putTextRect(img, f'ID {id}', (x1, y1 - 10), scale=1, thickness=2)

        # Center point
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Count vehicles crossing line
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if id not in totalCount:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # ---------------- DISPLAY COUNT ----------------
    cv2.putText(img, f'Count: {len(totalCount)}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # ---------------- SHOW OUTPUT ----------------
    cv2.imshow("Final Output", img)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()