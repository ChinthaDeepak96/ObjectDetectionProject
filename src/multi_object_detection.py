import cv2
from ultralytics import YOLO

def detect_cars_in_video(video_path=0):
    # Load pre-trained YOLO model (YOLOv5s for speed; change to 'yolov5m.pt' for accuracy if needed)
    model = YOLO("models/custom_coco_yolov8s.pt")  # Downloads automatically on first run

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)

        # Process detections
        for result in results:
            boxes = result.boxes  # Bounding boxes
            for box in boxes:
                # Get box coordinates and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()  # Confidence
                cls = int(box.cls[0].cpu().numpy())  # Class ID
                label = model.names[cls]  # Class name

                # Filter for vehicles (cars, trucks, buses)
                if label in ['car', 'truck', 'bus'] and conf > 0.5:  # Confidence threshold
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Car Detection with YOLO', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_cars_in_frame(frame):
    model = YOLO('yolov5s.pt')  # Ensure model is loaded

    results = model(frame)
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = model.names[cls]

            if label in ['car', 'truck', 'bus'] and conf > 0.5:
                detections.append({
                    'type': label,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf)
                })

    return detections