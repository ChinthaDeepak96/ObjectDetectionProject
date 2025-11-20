#!/usr/bin/env python
"""
src/main.py

Unified multi-object detection runner:
- runs face detection, pedestrian detection, YOLO object detection
- draws color-coded boxes on frame
- collects detections per-frame and writes a single JSON log at the end
- usage: python src/main.py             # runs webcam (0)
       python src/main.py --source video.mp4
       python src/main.py --source 0 --log ../output/detections_log.json
"""
import cv2
import json
import time
import os
import argparse
from pathlib import Path

# Local (same-folder) imports — ensure this file is executed from repo src folder
from face_detection import detect_faces_in_frame
from multi_object_detection import detect_objects_in_frame, detect_cars_in_frame  # keep both if you have separate helpers
from pedestrian_detection import detect_pedestrians_in_frame

# Color map for different types (extend as required)
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
    """
    Pick a color for a label. If label contains known keywords use that color,
    otherwise return default.
    """
    lbl = label.lower()
    for k in TYPE_COLORS:
        if k != "default" and k in lbl:
            return TYPE_COLORS[k]
    return TYPE_COLORS["default"]


def normalize_bbox(bbox):
    """
    Accept bbox in several formats and return integers x1,y1,x2,y2
    - bbox may be [x,y,w,h] or [x1,y1,x2,y2]
    - may be floats - convert to int
    """
    if bbox is None:
        return None
    bbox = list(bbox)
    if len(bbox) == 4:
        # detect if it's x,y,w,h by checking second pair relative to first
        x0, y0, x1, y1 = bbox
        # Heuristic: if x1 and y1 are widths/heights (small) treat as w,h
        if (x1 - x0) < 0 or (y1 - y0) < 0:
            # treat as x,y,w,h
            x, y, w, h = bbox
            return [int(x), int(y), int(x + w), int(y + h)]
        else:
            # treat as x1,y1,x2,y2
            return [int(x0), int(y0), int(x1), int(y1)]
    else:
        # unexpected format
        return [int(float(v)) for v in bbox[:4]]


def main(args):
    # Resolve paths relative to the script's location
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
        source_val = source  # path string

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {source_val}")
        return

    print(f"Starting continuous object detection on source={source_val}. Press 'q' to quit.")
    frame_count = 0
    all_detections = []  # collect dictionaries to save as JSON at the end
    start_time = time.time()

    # For FPS calc
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

            # 1) Face detection (Haar)
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

            # 2) YOLO / generic object detection (cars, etc.)
            try:
                objects = detect_objects_in_frame(frame)  # expects list of dicts: {'type','bbox','confidence'}
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

            # 3) Cars-specific helper (if available)
            try:
                cars = detect_cars_in_frame(frame)
            except Exception:
                cars = []

            for car in cars:
                # car dict expected: {'type','bbox','confidence'}
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

            # 4) Pedestrian detection (Haar or custom)
            try:
                pedestrians = detect_pedestrians_in_frame(frame)
            except Exception as e:
                pedestrians = []
                print(f"Warning: pedestrian detection error: {e}")

            for ped in pedestrians:
                ped_type = ped.get("type", "pedestrian")
                ped_conf = float(ped.get("confidence", 0.9))
                bbox_raw = ped.get("bbox")
                # If ped bbox is (x,y,w,h)
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

            # If any detections, print + store
            if detections["objects"]:
                print(f"Frame {frame_count}: Detected {len(detections['objects'])} objects")
                all_detections.append(detections)

            # FPS overlay
            now = time.time()
            fps = 1.0 / (now - prev_time + 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show
            cv2.imshow("Multi-Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quitting by user request.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        # Save JSON log (write once to avoid trailing commas)
        try:
            with open(log_file_path, "w") as f:
                json.dump(all_detections, f, indent=2)
            print(f"Detections log saved to: {log_file_path}")
        except Exception as e:
            print(f"Error saving log file: {e}")

        cap.release()
        cv2.destroyAllWindows()
        total_time = time.time() - start_time
        print(f"Stopped. Processed {frame_count} frames in {total_time:.2f}s ({frame_count/ (total_time+1e-6):.2f} FPS).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-object detection runner")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam or path to video file)")
    parser.add_argument("--log", type=str, default="", help="Optional output JSON log path (default: ./output/detections_log.json)")
    args = parser.parse_args()

    # Normalize default log path
    if not args.log:
        args.log = str(Path(__file__).resolve().parent.parent / "output" / "detections_log.json")

    main(args)
