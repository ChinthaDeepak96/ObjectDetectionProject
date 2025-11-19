import os
import json
import shutil
from tqdm import tqdm
from pathlib import Path

# ----------------------------------------------------
# ✅ EXACT PATHS (YOUR FINAL CONFIRMED PATHS)
# ----------------------------------------------------
project_root = r"D:\PROJECTS\ObjectDetectionProject"
coco_dir = r"D:\PROJECTS\ObjectDetectionProject\data\coco"
output_dir = r"D:\PROJECTS\ObjectDetectionProject\data\yolo_coco"

train_img_src = r"D:\PROJECTS\ObjectDetectionProject\data\coco\train2017"
val_img_src   = r"D:\PROJECTS\ObjectDetectionProject\data\coco\val2017"

train_json = r"D:\PROJECTS\ObjectDetectionProject\data\coco\annotations\instances_train2017.json"
val_json   = r"D:\PROJECTS\ObjectDetectionProject\data\coco\annotations\instances_val2017.json"

# ----------------------------------------------------
# CREATE YOLO OUTPUT FOLDERS
# ----------------------------------------------------
folders = [
    fr"{output_dir}\images\train",
    fr"{output_dir}\images\val",
    fr"{output_dir}\labels\train",
    fr"{output_dir}\labels\val"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("🚀 Starting COCO ➜ YOLO conversion...")


# ----------------------------------------------------
# FUNCTION: Convert COCO JSON → YOLO Label Files
# ----------------------------------------------------
def convert_coco_to_yolo(json_path, img_src_dir, label_dst_dir, img_dst_dir):
    print(f"\n📌 Processing annotation file: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}

    annotations_by_image = {}
    for ann in data["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, img_info in tqdm(images.items(), desc="Converting COCO→YOLO"):
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        # Copy image
        src = os.path.join(img_src_dir, file_name)
        dst = os.path.join(img_dst_dir, file_name)
        if os.path.exists(src):
            shutil.copy(src, dst)

        # Write YOLO label file
        label_path = os.path.join(label_dst_dir, file_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as lf:
            for ann in annotations_by_image.get(image_id, []):
                cls = ann["category_id"] - 1
                x, y, w, h = ann["bbox"]

                xc = (x + w / 2) / width
                yc = (y + h / 2) / height
                wn = w / width
                hn = h / height

                lf.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")


# ----------------------------------------------------
# RUN CONVERSION FOR TRAIN + VAL
# ----------------------------------------------------
convert_coco_to_yolo(train_json, train_img_src, fr"{output_dir}\labels\train", fr"{output_dir}\images\train")
convert_coco_to_yolo(val_json,   val_img_src,   fr"{output_dir}\labels\val",   fr"{output_dir}\images\val")

print("\n✅ COCO ➜ YOLO Conversion Completed!")


# ----------------------------------------------------
# CREATE data.yaml FOR YOLO TRAINING
# ----------------------------------------------------
yaml_path = fr"{output_dir}\data.yaml"

yaml_content = f"""
train: {output_dir}/images/train
val: {output_dir}/images/val

nc: 80

names:
  - person
  - bicycle
  - car
  - motorcycle
  - airplane
  - bus
  - train
  - truck
  - boat
  - traffic light
  - fire hydrant
  - stop sign
  - parking meter
  - bench
  - bird
  - cat
  - dog
  - horse
  - sheep
  - cow
  - elephant
  - bear
  - zebra
  - giraffe
  - backpack
  - umbrella
  - handbag
  - tie
  - suitcase
  - frisbee
  - skis
  - snowboard
  - sports ball
  - kite
  - baseball bat
  - baseball glove
  - skateboard
  - surfboard
  - tennis racket
  - bottle
  - wine glass
  - cup
  - fork
  - knife
  - spoon
  - bowl
  - banana
  - apple
  - sandwich
  - orange
  - broccoli
  - carrot
  - hot dog
  - pizza
  - donut
  - cake
  - chair
  - couch
  - potted plant
  - bed
  - dining table
  - toilet
  - tv
  - laptop
  - mouse
  - remote
  - keyboard
  - cell phone
  - microwave
  - oven
  - toaster
  - sink
  - refrigerator
  - book
  - clock
  - vase
  - scissors
  - teddy bear
  - hair drier
  - toothbrush
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"\n📄 data.yaml saved at: {yaml_path}")
print("🎉 DATASET READY FOR YOLO TRAINING!")
