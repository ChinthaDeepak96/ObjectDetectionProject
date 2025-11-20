"""
combined_autopilot.py
Combines YOLOv8 object detection + lane detection + steering angle calculation
Run: python src/combined_autopilot.py
"""

import cv2
import numpy as np
import math
import os
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
MODEL_LOCAL_PATH = os.path.join("models", "yolov8x.pt")  # prefer local models/yolov8x.pt
CONF_THRESH = 0.35          # YOLO confidence threshold
IOU_THRESH = 0.45
FRAME_WIDTH = 960           # processing width (resize for speed)
HAZARD_CLASSES = {"person", "car", "truck", "bus", "motorcycle"}  # classes we treat as obstacle
OBSTACLE_AREA_THRESH = 0.02  # bbox area ratio (bbox_area / frame_area) considered "close"
LANE_DY_FACTOR = 0.6        # vertical offset used in steering calculation
# -----------------------------

# -----------------------------
# Helper: Lane detection utilities
# -----------------------------
def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(edges):
    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, h),
        (w, h),
        (int(w * 0.55), int(h * 0.6)),
        (int(w * 0.45), int(h * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped = cv2.bitwise_and(edges, mask)
    return cropped

def detect_lines(cropped_edges):
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=120
    )
    return lines

def draw_lines(frame, lines):
    line_img = np.zeros_like(frame)
    if lines is None:
        return frame
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 6)
    combined = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return combined

# -----------------------------
# Steering calculation
# -----------------------------
def calculate_steering_angle(frame, lines):
    """Return steering angle in degrees. Positive => turn right, Negative => turn left."""
    h, w = frame.shape[:2]
    if lines is None:
        return 0

    left_x = []
    right_x = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        # ignore nearly horizontal lines
        if abs(y2 - y1) < 10:
            continue
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if slope < 0:
            left_x.append((x1 + x2) / 2)
        else:
            right_x.append((x1 + x2) / 2)

    if not left_x or not right_x:
        return 0

    left_mean = np.mean(left_x)
    right_mean = np.mean(right_x)
    lane_center_x = (left_mean + right_mean) / 2.0
    frame_center_x = w / 2.0
    dx = lane_center_x - frame_center_x
    dy = h * LANE_DY_FACTOR

    angle_rad = math.atan2(dx, dy)  # note order (dx, dy) to get left/right
    angle_deg = math.degrees(angle_rad)
    return float(angle_deg)

def draw_steering_indicator(frame, angle):
    h, w = frame.shape[:2]
    center_x = int(w / 2)
    bottom_y = h - 10
    # scale factor for visual line
    length = 120
    end_x = int(center_x + length * math.sin(math.radians(angle)))
    end_y = int(bottom_y - length * math.cos(math.radians(angle)))
    cv2.line(frame, (center_x, bottom_y), (end_x, end_y), (0, 0, 255), 6)
    cv2.putText(frame, f"Steering: {angle:.1f} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    return frame

# -----------------------------
# YOLO helper
# -----------------------------
def load_yolo_model():
    # If local path exists, load it; otherwise, let ultralytics fetch by name (it downloads)
    if os.path.exists(MODEL_LOCAL_PATH):
        print(f"Loading local model: {MODEL_LOCAL_PATH}")
        return YOLO(MODEL_LOCAL_PATH)
    else:
        print("Local model not found. Loading 'yolov8x.pt' from Ultralytics (will auto-download).")
        return YOLO("yolov8x.pt")

def parse_detections(results, frame_area):
    """
    results: ultralytics result object (one frame)
    returns list of detections: (class_name, conf, bbox, bbox_area_ratio)
    bbox = (x1, y1, x2, y2)
    """
    dets = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0]) if hasattr(box, "conf") else float(box.conf)
            cls_idx = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            # try to get class name from model.names
            name = model.names.get(cls_idx, str(cls_idx)) if hasattr(model, "names") else str(cls_idx)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area_ratio = (w * h) / frame_area
            dets.append((name, conf, (x1, y1, x2, y2), area_ratio))
    return dets

# -----------------------------
# Main runtime pipeline
# -----------------------------
def run(video_source=0):
    global model
    model = load_yolo_model()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("ERROR: Could not open video source:", video_source)
        return

    # set capture width for stable processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # resize for consistent processing speed (keep aspect ratio)
        h0, w0 = frame.shape[:2]
        scale = FRAME_WIDTH / w0 if w0 > FRAME_WIDTH else 1.0
        if scale != 1.0:
            frame_proc = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
        else:
            frame_proc = frame.copy()

        fh, fw = frame_proc.shape[:2]
        frame_area = fw * fh

        # --- YOLO detection (inference)
        # run with stream=True returns generator; passing frame directly returns results list
        results = model(frame_proc, conf=CONF_THRESH, iou=IOU_THRESH, stream=False)

        dets = parse_detections(results, frame_area)

        # Draw detections
        for name, conf, (x1, y1, x2, y2), area_ratio in dets:
            label = f"{name} {conf:.2f}"
            color = (255, 165, 0)  # orange boxes
            cv2.rectangle(frame_proc, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_proc, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Check hazard detection (simple heuristic: large bbox area + class in HAZARD_CLASSES)
        obstacle_detected = False
        for name, conf, bbox, area_ratio in dets:
            if name in HAZARD_CLASSES and conf >= 0.4 and area_ratio >= OBSTACLE_AREA_THRESH:
                obstacle_detected = True
                cv2.putText(frame_proc, "OBSTACLE: STOP", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                break

        # --- Lane detection
        edges = detect_edges(frame_proc)
        cropped = region_of_interest(edges)
        lines = detect_lines(cropped)
        lane_vis = draw_lines(frame_proc, lines)

        # Steering angle
        angle = calculate_steering_angle(frame_proc, lines)
        final_vis = draw_steering_indicator(lane_vis, angle)

        # Overlay some status info
        cv2.putText(final_vis, f"Detections: {len(dets)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        if obstacle_detected:
            cv2.putText(final_vis, ">> EMERGENCY STOP <<", (fw//2 - 170, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 3)

        cv2.imshow("Combined Autopilot", final_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # save current frame snapshot
            cv2.imwrite("snapshot.png", final_vis)
            print("Saved snapshot.png")

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # default video source 0 -> webcam
    run(0)
