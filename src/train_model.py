import os
from ultralytics import YOLO

# Correct Windows path (raw string)
data_yaml = r"D:\PROJECTS\ObjectDetectionProject\data\yolo_coco\data.yaml"

models_dir = r"D:\PROJECTS\ObjectDetectionProject\models"
os.makedirs(models_dir, exist_ok=True)

output_model = os.path.join(models_dir, "custom_coco_yolov8s.pt")

# Load YOLOv8-s
model = YOLO("yolov8s.pt")

model.train(
    data=data_yaml,
    epochs=10,
    imgsz=640,
    batch=16,
    device=0,
    fraction=0.20,
    workers=0,
    name="coco_custom_train",
    pretrained=True
)

best_model_path = model.ckpt_path
import shutil
shutil.copy(best_model_path, output_model)

print(f"Training complete! Model saved to: {output_model}")
