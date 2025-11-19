import cv2
import json
import time
import os
from face_detection import detect_faces_in_frame
from src.multi_object_detection import detect_cars_in_frame
from pedestrian_detection import detect_pedestrians_in_frame
from multi_object_detection import detect_objects_in_frame

def main():
    # Resolve paths relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)  # Create output dir if it doesn't exist
    log_file_path = os.path.join(output_dir, 'detections_log.json')

    # Open video source (0 for webcam, or path to file)
    cap = cv2.VideoCapture(0)  # Change to '../data/sample.mp4' for testing
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    # Log file for detections
    log_file = open(log_file_path, 'a')
    log_file.write('[\n')  # Start JSON array

    print("Starting continuous object detection. Press 'q' to quit.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = time.time()

        # Detect all objects in the frame
        detections = {
            'frame': frame_count,
            'timestamp': timestamp,
            'objects': []
        }

        # Face detection (Haar Cascade)
        faces = detect_faces_in_frame(frame)
        for (x, y, w, h) in faces:
            detections['objects'].append({
                'type': 'face',
                'bbox': [int(x), int(y), int(x+w), int(y+h)],  # Convert to int
                'confidence': 0.9
            })
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue for faces
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Car detection (YOLO)
        cars = detect_cars_in_frame(frame)
        for car in cars:
            detections['objects'].append({
                'type': car['type'],
                'bbox': [int(car['bbox'][0]), int(car['bbox'][1]), int(car['bbox'][2]), int(car['bbox'][3])],  # Convert to int
                'confidence': float(car['confidence'])  # Ensure float
            })
            x1, y1, x2, y2 = car['bbox']
            label, conf = car['type'], car['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for cars
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Pedestrian detection (Haar Cascade)
        pedestrians = detect_pedestrians_in_frame(frame)
        for ped in pedestrians:
            detections['objects'].append({
                'type': ped['type'],
                'bbox': [int(ped['bbox'][0]), int(ped['bbox'][1]), int(ped['bbox'][2]), int(ped['bbox'][3])],  # Convert to int
                'confidence': float(ped['confidence'])  # Ensure float
            })
            x, y, w, h = ped['bbox']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for pedestrians
            cv2.putText(frame, 'Pedestrian', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Log detections to console and file
        if detections['objects']:
            print(f"Frame {frame_count}: Detected {len(detections['objects'])} objects - {detections}")
            log_file.write(json.dumps(detections) + ',\n')

        # Display the frame
        cv2.imshow('Multi-Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    log_file.write(']\n')
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Detection stopped. Check '{log_file_path}' for logs.")

if __name__ == "__main__":
    main()


# In the main loop:
objects = detect_objects_in_frame(frame)
for obj in objects:
    detections['objects'].append(obj)
    x1, y1, x2, y2 = obj['bbox']
    label, conf = obj['type'], obj['confidence']
    # Color-code by type (add more as needed)
    color = (0, 255, 0) if 'car' in label else (255, 0, 0) if 'person' in label else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)