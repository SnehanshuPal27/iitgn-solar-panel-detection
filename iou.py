from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import box
import supervision as sv

def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: float = 1.0,
    img_height: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Convert YOLO format (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max).
    """
    x_min = (x_center - width/2) * img_width
    y_min = (y_center - height/2) * img_height
    x_max = (x_center + width/2) * img_width
    y_max = (y_center + height/2) * img_height
    
    return x_min, y_min, x_max, y_max

def compute_iou_shapely(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """
    Compute IoU between two bounding boxes using Shapely.
    """
    # Create Shapely boxes
    polygon1 = box(*box1)
    polygon2 = box(*box2)
    
    # Check if boxes intersect
    if not polygon1.intersects(polygon2):
        return 0.0
    
    # Compute intersection and union areas
    intersection_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area
    
    return intersection_area / union_area

def compute_iou_supervision(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """
    Compute IoU between two bounding boxes using supervision library.
    """
    box1_arr = np.array([box1], dtype=np.float32)
    box2_arr = np.array([box2], dtype=np.float32)
    
    return sv.box_iou_batch(box1_arr, box2_arr)[0][0]

def main() -> None:
    # Test cases
    # Case 1: Perfectly overlapping boxes
    box1_yolo = (0.5, 0.5, 0.3, 0.3)
    box2_yolo = (0.5, 0.67, 0.3, 0.3)
    
    box1_xyxy = yolo_to_xyxy(*box1_yolo)
    box2_xyxy = yolo_to_xyxy(*box2_yolo)
    
    iou_shapely = compute_iou_shapely(box1_xyxy, box2_xyxy)
    iou_supervision = compute_iou_supervision(box1_xyxy, box2_xyxy)
    
    print(f"Test Case 1: Perfect overlap")
    print(f"Shapely IoU: {iou_shapely:.4f}")
    print(f"Supervision IoU: {iou_supervision:.4f}")

    # Add more test cases as needed

if __name__ == "__main__":
    main()