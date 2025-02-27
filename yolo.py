from ultralytics import YOLO
import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
images_dir = "/fab3/btech/2022/snehanshu.pal22b/SolarPanel/image_chips_native-20250212T103727Z-001/image_chips_native"  # Update this
labels_dir = "/fab3/btech/2022/snehanshu.pal22b/SolarPanel/labels/labels_native"  # Update this
dataset_dir = "yolo_dataset"  # Directory to store YOLO formatted dataset

# # Create YOLO dataset structure
# for split in ['train', 'val', 'test']:
#     os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
#     os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)

# # Get image and corresponding label files
# image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.tif'))]

# data = []
# for img in image_files:
#     label_file = os.path.splitext(img)[0] + ".txt"
#     if label_file in os.listdir(labels_dir):
#         data.append((img, label_file))

# # Split dataset into train (80%) and test (20%)
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# # Further split train data into train (90%) and val (10%)
# train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# # Move files to YOLO structure
# def move_files(data_split, split_name):
#     for img, label in data_split:
#         shutil.copy(os.path.join(images_dir, img), os.path.join(dataset_dir, 'images', split_name, img))
#         shutil.copy(os.path.join(labels_dir, label), os.path.join(dataset_dir, 'labels', split_name, label))

# move_files(train_data, 'train')
# move_files(val_data, 'val')
# move_files(test_data, 'test')

# Create YOLOv8 data configuration file
data_yaml = f"""
nc: 1
names: ["solar_panel"]  # Update class names
download: ""
train: {os.path.join(dataset_dir, 'images', 'train')}
val: {os.path.join(dataset_dir, 'images', 'val')}
test: {os.path.join(dataset_dir, 'images', 'test')}"""

# with open("yolo_dataset/data.yaml", "w") as f:
#     f.write(data_yaml)

# Train YOLO model
model = YOLO("yolov8s.pt")  # Using YOLOv8 small model
model.train(data="yolo_dataset/data.yaml", epochs=20, imgsz=416, device="5")  # Training on GPU 0
