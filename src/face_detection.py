import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, '..', 'models')
output_dir = os.path.join(script_dir, '..', 'output')

def detect_faces(image_path, cascade_path=os.path.join(models_dir, 'haarcascade_frontalface_default.xml')):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise ValueError(f"Failed to load cascade classifier: {cascade_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Detected Faces', image)
    output_path = os.path.join(output_dir, 'detected_faces.jpg')
    cv2.imwrite(output_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return len(faces)

def detect_faces_in_video(video_path=0, cascade_path=os.path.join(models_dir, 'haarcascade_frontalface_default.xml')):
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cascade_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Video Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_faces_in_frame(frame, cascade_path=os.path.join(models_dir, 'haarcascade_frontalface_default.xml')):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Warning: Face cascade not loaded.")
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces  # List of (x, y, w, h)