from ultralytics import YOLO
import cv2

model = YOLO("/Volumes/WD SABREN/PROJECTS/ObjectDetectionProject/yolov8x.pt")   # Your trained model

cap = cv2.VideoCapture(0)   # Use webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot()

    cv2.imshow("Custom YOLOv8 COCO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
