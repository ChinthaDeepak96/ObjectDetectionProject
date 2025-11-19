import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, '..', 'models')

def detect_pedestrians_in_video(video_path=0, cascade_path=os.path.join(models_dir, 'haarcascade_fullbody.xml')):
    cap = cv2.VideoCapture(video_path)
    pedestrian_cascade = cv2.CascadeClassifier(cascade_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'Pedestrian', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Pedestrian Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_pedestrians_in_frame(frame, cascade_path=os.path.join(models_dir, 'haarcascade_fullbody.xml')):
    pedestrian_cascade = cv2.CascadeClassifier(cascade_path)
    if pedestrian_cascade.empty():
        print("Warning: Pedestrian cascade not loaded.")
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detections = [{'type': 'pedestrian', 'bbox': [x, y, w, h], 'confidence': 0.8} for (x, y, w, h) in pedestrians]
    return detections