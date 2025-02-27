import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv
from sklearn.metrics import precision_score, recall_score, f1_score
from iou import yolo_to_xyxy, compute_iou_shapely, compute_iou_supervision

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train6/weights/best.pt")

# Get test images from directory
test_images_dir = "yolo_dataset/images/test"
test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
               if f.endswith(('.jpg', '.jpeg', '.png', '.tif'))]

# Select 5 random test images
selected_images = random.sample(test_images, 5)

def evaluate_predictions(image_path):
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape
    
    # Run YOLOv8 inference
    results = model(image_path)
    predictions = results[0]
    
    pred_boxes = []
    for box in predictions.boxes.xywhn.cpu().numpy():
        x, y, w, h = box
        pred_boxes.append(yolo_to_xyxy(x, y, w, h, img_width, img_height))
    
    # Load ground truth
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    label_path = os.path.join("yolo_dataset/labels/test", f"{base_name}.txt")
    
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                _, x, y, w, h = map(float, line.strip().split())
                gt_boxes.append(yolo_to_xyxy(x, y, w, h, img_width, img_height))
    
    # Compute IoU scores
    iou_scores = []
    for gt in gt_boxes:
        for pred in pred_boxes:
            iou_shapely = compute_iou_shapely(gt, pred)
            iou_supervision = compute_iou_supervision(gt, pred)
            iou_scores.append((iou_shapely, iou_supervision))
    
    return iou_scores, pred_boxes, gt_boxes

def visualize_predictions(image_path, pred_boxes, gt_boxes, save_dir="predictions"):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    image = cv2.imread(image_path)
    image_copy = image.copy()  # Create a copy for saving
    
    # Draw ground truth boxes (Green)
    for box in gt_boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Add "GT" label
        cv2.putText(image_copy, "GT", (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    
    # Draw predicted boxes (Red)
    for box in pred_boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        # Add "Pred" label
        cv2.putText(image_copy, "Pred", (x_min, y_max+15), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 2)
    
    # Save the annotated image
    base_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"pred_{base_name}")
    cv2.imwrite(save_path, image_copy)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.title(f"Green: Ground Truth | Red: Predictions\nSaved as: {save_path}")
    plt.axis("off")
    plt.show()

# Evaluate 5 random test images
results = []
for img in selected_images:
    iou_scores, pred_boxes, gt_boxes = evaluate_predictions(img)
    visualize_predictions(img, pred_boxes, gt_boxes)
    for iou_shapely, iou_supervision in iou_scores:
        results.append([img, iou_shapely, iou_supervision, len(pred_boxes), len(gt_boxes)])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Image", "Shapely IoU", "Supervision IoU", "Predicted Boxes", "GT Boxes"])
print(results_df)

# Compute precision, recall, and F1-score
precisions, recalls, f1_scores = [], [], []

for img in selected_images:
    iou_scores, pred_boxes, gt_boxes = evaluate_predictions(img)
    tp = sum(1 for iou, _ in iou_scores if iou >= 0.5)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall: {np.mean(recalls):.4f}")
print(f"Average F1-score: {np.mean(f1_scores):.4f}")