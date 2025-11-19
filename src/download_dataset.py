import kagglehub

# Download latest version of COCO 2017 dataset
path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")

print("Path to dataset files:", path)
# Optional: Move to your data folder
import shutil
import os
dest = os.path.join(os.path.dirname(__file__), '..', 'data', 'coco')
os.makedirs(dest, exist_ok=True)
shutil.move(path, dest)
print(f"Dataset moved to: {dest}")