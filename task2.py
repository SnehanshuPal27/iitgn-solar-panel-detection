import numpy as np
from shapely.geometry import box
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import random

def compute_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    polygon1 = box(*box1)
    polygon2 = box(*box2)
    
    if not polygon1.intersects(polygon2):
        return 0.0
    
    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area
    
    return intersection_area / union_area

def pascal_voc_11_point_ap(recalls, precisions):
    """Compute AP using Pascal VOC 11 point interpolation method."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) != 0 else 0
        ap += p / 11.0
    return ap

def coco_101_point_ap(recalls, precisions):
    """Compute AP using COCO 101-point interpolation method."""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) != 0 else 0
        ap += p / 101.0
    return ap

def auc_pr_ap(recalls, precisions):
    """Compute AP using Area under Precision-Recall Curve method."""
    return auc(recalls, precisions)

def generate_random_boxes(image_size, box_size, num_boxes):
    """Generate random bounding boxes within the given image size."""
    boxes = []
    for _ in range(num_boxes):
        x_min = random.randint(0, image_size - box_size)
        y_min = random.randint(0, image_size - box_size)
        x_max = x_min + box_size
        y_max = y_min + box_size
        boxes.append((x_min, y_min, x_max, y_max))
    return boxes

def compute_precision_recall(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Compute precision and recall for given ground truth and predicted boxes."""
    tp = 0
    fp = 0
    fn = len(gt_boxes)
    
    for pred_box in pred_boxes:
        ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
        if max(ious) >= iou_threshold:
            tp += 1
            fn -= 1
        else:
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

def main():
    image_size = 100
    box_size = 20
    num_images = 10
    num_boxes = 10
    iou_threshold = 0.5

    precisions = []
    recalls = []

    for _ in range(num_images):
        gt_boxes = generate_random_boxes(image_size, box_size, num_boxes)
        pred_boxes = generate_random_boxes(image_size, box_size, num_boxes)
        
        precision, recall = compute_precision_recall(gt_boxes, pred_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    ap_voc = pascal_voc_11_point_ap(recalls, precisions)
    ap_coco = coco_101_point_ap(recalls, precisions)
    ap_auc = auc_pr_ap(recalls, precisions)

    print(f"Pascal VOC 11-point AP: {ap_voc:.4f}")
    print(f"COCO 101-point AP: {ap_coco:.4f}")
    print(f"AUC-PR AP: {ap_auc:.4f}")

if __name__ == "__main__":
    main()