"""
combined_autopilot.py  
Merged + optimized autopilot with:
- YOLOv8 detection (robust parsing)
- Lane detection
- Steering angle estimation
- Obstacle LEFT/CENTER/RIGHT classification
- VehicleController integration
- More UI, better stability, improved performance

Run: python combined_autopilot.py
"""

import cv2
import numpy as np
import math
import os
from ultralytics import YOLO
from vehicle_controller import VehicleController

# -----------------------------
# CONFIG
# -----------------------------
MODEL_LOCAL_PATH = os.path.join("models", "yolov8x.pt")
CONF_THRESH = 0.35
IOU_THRESH = 0.45
FRAME_WIDTH = 960
HAZARD_CLASSES = {"person", "car", "truck", "bus", "motorcycle"}
OBSTACLE_AREA_THRESH = 0.02
LANE_DY_FACTOR = 0.6

# optional steering smoothing
SMOOTHING_ALPHA = 0.25
last_angle = 0.0


# ============================================================
# LANE DETECTION
# ============================================================
def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


def region_of_interest(edges):
    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(w * 0.1), h),
        (int(w * 0.9), h),
        (int(w * 0.55), int(h * 0.55)),
        (int(w * 0.45), int(h * 0.55))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def detect_lines(cropped_edges):
    return cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=120
    )


def draw_lines(frame, lines):
    overlay = np.zeros_like(frame)
    if lines is None:
        return frame
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 6)
    return cv2.addWeighted(frame, 0.8, overlay, 1, 1)


# ============================================================
# STEERING
# ============================================================
def calculate_steering_angle(frame, lines):
    h, w = frame.shape[:2]
    if lines is None:
        return 0.0

    left_x = []
    right_x = []

    for l in lines:
        x1, y1, x2, y2 = l[0]
        if abs(y2 - y1) < 10:
            continue

        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        mid_x = (x1 + x2) / 2

        if slope < 0:
            left_x.append(mid_x)
        else:
            right_x.append(mid_x)

    if not left_x or not right_x:
        return 0.0

    lane_center = (np.mean(left_x) + np.mean(right_x)) / 2
    frame_center = w / 2
    dx = lane_center - frame_center
    dy = h * LANE_DY_FACTOR

    angle_deg = math.degrees(math.atan2(dx, dy))
    return float(angle_deg)


def smooth_angle(angle):
    global last_angle
    smoothed = last_angle * (1 - SMOOTHING_ALPHA) + angle * SMOOTHING_ALPHA
    last_angle = smoothed
    return smoothed


def draw_steering_indicator(frame, angle):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h - 20
    length = 120

    ex = int(cx + length * math.sin(math.radians(angle)))
    ey = int(cy - length * math.cos(math.radians(angle)))

    cv2.line(frame, (cx, cy), (ex, ey), (0, 0, 255), 6)
    cv2.putText(frame, f"Steering: {angle:.1f} deg", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    return frame


# ============================================================
# YOLO UTILITIES
# ============================================================
def load_yolo_model():
    if os.path.exists(MODEL_LOCAL_PATH):
        print(f"[INFO] Loading local YOLO model: {MODEL_LOCAL_PATH}")
        return YOLO(MODEL_LOCAL_PATH)
    else:
        print("[INFO] Local model missing. Auto-downloading YOLOv8x.")
        return YOLO("yolov8x.pt")


def parse_detections(results, frame_area):
    dets = []

    for r in results:
        if not hasattr(r, "boxes"):
            continue

        for box in r.boxes:
            # robust tensor-safe extraction
            conf = float(box.conf[0] if hasattr(box.conf, "__len__") else box.conf)
            cls = int(box.cls[0] if hasattr(box.cls, "__len__") else box.cls)

            if hasattr(model, "names") and cls in model.names:
                name = model.names[cls]
            else:
                name = str(cls)

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area_ratio = (w * h) / frame_area

            dets.append((name, conf, (x1, y1, x2, y2), area_ratio))

    return dets


# ============================================================
# MAIN AUTOPILOT PIPELINE
# ============================================================
def run(video_source=0):
    global model
    model = load_yolo_model()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source:", video_source)
        return

    controller = VehicleController()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # resize while keeping aspect ratio
        h0, w0 = frame.shape[:2]
        scale = FRAME_WIDTH / w0 if w0 > FRAME_WIDTH else 1.0
        frame_proc = cv2.resize(frame, (int(w0 * scale), int(h0 * scale))) if scale != 1.0 else frame.copy()

        fh, fw = frame_proc.shape[:2]
        frame_area = fw * fh

        # ------------- YOLO -------------
        results = model(frame_proc, conf=CONF_THRESH, iou=IOU_THRESH, stream=False)
        dets = parse_detections(results, frame_area)

        obstacle_detected = False
        obstacle_pos = None

        # Determine obstacle position
        for name, conf, (x1, y1, x2, y2), area_ratio in dets:
            if name in HAZARD_CLASSES and conf >= 0.4 and area_ratio >= OBSTACLE_AREA_THRESH:
                obstacle_detected = True
                cx = (x1 + x2) // 2

                if cx < fw * 0.33:
                    obstacle_pos = "LEFT"
                elif cx > fw * 0.66:
                    obstacle_pos = "RIGHT"
                else:
                    obstacle_pos = "CENTER"
                break

        # Draw bounding boxes
        for name, conf, (x1, y1, x2, y2), area_ratio in dets:
            cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0,165,255), 2)
            cv2.putText(frame_proc, f"{name} {conf:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

        # emergency visuals
        if obstacle_detected:
            cv2.putText(frame_proc, "<< EMERGENCY STOP >>", (fw//2 - 160, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)

        # ------------- LANES -------------
        edges = detect_edges(frame_proc)
        cropped = region_of_interest(edges)
        lines = detect_lines(cropped)
        lane_vis = draw_lines(frame_proc, lines)

        # ------------- STEERING -------------
        raw_angle = calculate_steering_angle(frame_proc, lines)
        smooth = smooth_angle(raw_angle)
        final_vis = draw_steering_indicator(lane_vis, smooth)

        # ------------- VEHICLE CONTROL -------------
        lane_available = lines is not None and len(lines) > 0
        state, speed = controller.update(smooth, obstacle_detected, lane_available, obstacle_pos)

        # HUD info
        cv2.putText(final_vis, f"Detections: {len(dets)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        cv2.putText(final_vis, f"State: {state}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(final_vis, f"Speed: {speed:.2f}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        if obstacle_pos:
            cv2.putText(final_vis, f"Obstacle: {obstacle_pos}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)

        cv2.imshow("Combined Autopilot", final_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("snapshot.png", final_vis)
            print("[INFO] Snapshot saved as snapshot.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(0)
