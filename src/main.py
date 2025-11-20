#!/usr/bin/env python
"""
src/main.py

Unified multi-object detection runner with lane detection:
- runs face detection, pedestrian detection, YOLO object detection
- draws color-coded boxes on frame
- draws lane lines using draw_lines(frame, lines)
- collects detections per-frame and writes a single JSON log at the end
"""

import cv2
import json
import time
import os
import argparse
from pathlib import Path

# Local imports
from face_detection import detect_faces_in_frame
from multi_object_detection import detect_objects_in_frame, detect_cars_in_frame
from pedestrian_detection import detect_pedestrians_in_frame

# -------------------------------
# Lane detection placeholder
# Replace with your actual implementation
# -------------------------------
def detect_lanes(frame):
    """
    Dummy placeholder for lane detection.
    Replace with real Hough/canny pipeline.
    Should return: list of lines: [ [x1,y1,x2,y2], ... ]
    """
    return []


def draw_lines(frame, lines):
    """
    Draw lane lines on the frame.
    Each line is a list/tuple: (x1, y1, x2, y2)
    """
    lane_frame = frame.copy()
    for line in lines:
        if len(line) == 4:
            x1, y1, x2, y2 = line
            cv2.line(lane_frame, (x1, y1), (x2, y2), (0, 255, 255), 5)
    return lane_frame
# -------------------------------


TYPE_COLORS = {
    "person": (0, 255, 0),
    "car": (0, 200, 0),
    "truck": (0, 180, 0),
    "bicycle": (0, 150, 0),
    "motorcycle": (0, 120, 0),
    "face": (255, 0, 0),
    "pedestrian": (0, 0, 255),
    "default": (255, 255, 0)
}

def get_color_for_label(label: str):
    lbl = label.lower()
    for k in TYPE_COLORS:
        if k != "default" and k in lbl:
            return TYPE_COLORS[k]
    return TYPE_COLORS["default"]


def normalize_bbox(bbox):
    if bbox is None:
        return None
    bbox = list(bbox)
    if len(bbox) == 4:
        x0, y0, x1, y1 = bbox
        if (x1 - x0) < 0 or (y1 - y0) < 0:
            x, y, w, h = bbox
            return [int(x), int(y), int(x + w), int(y + h)]
        else:
            return [int(x0), int(y0), int(x1), int(y1)]
    else:
        return [int(float(v)) for v in bbox[:4]]


def main(args):
    script_dir = Path(__file__).resolve().parent
    default_output_dir = script_dir.parent / "output"
    output_dir = Path(args.log).parent if args.log else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = Path(args.log) if args.log else default_output_dir / "detections_log.json"

    # Video source
    source = args.source
    try:
        source_int = int(source)
        source_val = source_int
    except Exception:
        source_val = source

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {source_val}")
        return

    print(f"Starting continuous object detection on source={source_val}. Press 'q' to quit.")
    frame_count = 0
    all_detections = []
    start_time = time.time()
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or cannot read frame.")
                break

            frame_count += 1
            timestamp = time.time()

            detections = {
                "frame": frame_count,
                "timestamp": timestamp,
                "objects": []
            }

            # 1) Face detection
            try:
                faces = detect_faces_in_frame(frame)
            except Exception as e:
                faces = []
                print(f"Warning: face detection error: {e}")

            for (x, y, w, h) in faces:
                bbox = [int(x), int(y), int(x + w), int(y + h)]
                detections["objects"].append({
                    "type": "face",
                    "bbox": bbox,
                    "confidence": 0.9
                })
                color = TYPE_COLORS["face"]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, "Face", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 2) YOLO / generic object detection
            try:
                objects = detect_objects_in_frame(frame)
            except Exception as e:
                objects = []
                print(f"Warning: object detection error: {e}")

            for obj in objects:
                obj_type = str(obj.get("type", "object"))
                obj_conf = float(obj.get("confidence", 0.0))
                bbox_raw = obj.get("bbox", None)
                bbox = normalize_bbox(bbox_raw)
                if bbox is None:
                    continue

                detections["objects"].append({
                    "type": obj_type,
                    "bbox": bbox,
                    "confidence": obj_conf
                })

                color = get_color_for_label(obj_type)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{obj_type} {obj_conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 3) Cars-specific helper
            try:
                cars = detect_cars_in_frame(frame)
            except Exception:
                cars = []

            for car in cars:
                car_type = str(car.get("type", "car"))
                car_conf = float(car.get("confidence", 0.0))
                bbox = normalize_bbox(car.get("bbox"))
                if bbox is None:
                    continue

                detections["objects"].append({
                    "type": car_type,
                    "bbox": bbox,
                    "confidence": car_conf
                })

                color = get_color_for_label(car_type)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{car_type} {car_conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 4) Pedestrian detection
            try:
                pedestrians = detect_pedestrians_in_frame(frame)
            except Exception as e:
                pedestrians = []
                print(f"Warning: pedestrian detection error: {e}")

            for ped in pedestrians:
                ped_type = ped.get("type", "pedestrian")
                ped_conf = float(ped.get("confidence", 0.9))
                bbox_raw = ped.get("bbox")

                if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
                    x, y, w, h = bbox_raw
                    bbox = [int(x), int(y), int(x + w), int(y + h)]
                else:
                    bbox = normalize_bbox(bbox_raw)

                if bbox is None:
                    continue

                detections["objects"].append({
                    "type": ped_type,
                    "bbox": bbox,
                    "confidence": ped_conf
                })

                color = TYPE_COLORS.get("pedestrian", TYPE_COLORS["default"])
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, "Pedestrian", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Store detection record
            if detections["objects"]:
                print(f"Frame {frame_count}: Detected {len(detections['objects'])} objects")
                all_detections.append(detections)

            # FPS overlay
            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ------------------------
            # LANE DETECTION INTEGRATION
            # ------------------------
            lines = detect_lanes(frame)
            lane_frame = draw_lines(frame, lines)

            # Display lane-augmented frame
            cv2.imshow("Multi-Object Detection", lane_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quitting by user request.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        try:
            with open(log_file_path, "w") as f:
                json.dump(all_detections, f, indent=2)
            print(f"Detections log saved to: {log_file_path}")
        except Exception as e:
            print(f"Error saving log file: {e}")

        cap.release()
        cv2.destroyAllWindows()
        total_time = time.time() - start_time
        print(f"Stopped. Processed {frame_count} frames in "
              f"{total_time:.2f}s ({frame_count/ (total_time+1e-6):.2f} FPS).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-object detection runner")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (0 for webcam or path to video file)")
    parser.add_argument("--log", type=str, default="",
                        help="Optional output JSON log path")

    args = parser.parse_args()

    if not args.log:
        args.log = str(Path(__file__).resolve().parent.parent / "output" / "detections_log.json")

    main(args)
