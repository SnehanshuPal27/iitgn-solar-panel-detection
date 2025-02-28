import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio


LABELS_DIR = "/fab3/btech/2022/snehanshu.pal22b/SolarPanel/labels/labels_native"
IMAGES_DIR = "/fab3/btech/2022/snehanshu.pal22b/SolarPanel/image_chips_native-20250212T103727Z-001/image_chips_native"


DEFAULT_METERS_PER_PIXEL = 0.31  

def get_meters_per_pixel(tiff_path):
    
    try:
        with rasterio.open(tiff_path) as src:
            transform = src.transform
            meters_per_pixel = abs(transform.a)  
            return meters_per_pixel
    except Exception as e:
        print(f"Could not read {tiff_path}: {e}")
        return DEFAULT_METERS_PER_PIXEL  

def match_labels_to_images():
   
    
    label_files = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")])
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".tif")])
    
    
    label_map = {os.path.splitext(f)[0]: f for f in label_files}
    image_map = {os.path.splitext(f)[0]: f for f in image_files}

    
    matched_pairs = []
    for base_name in label_map:
        if base_name in image_map:  
            matched_pairs.append((label_map[base_name], image_map[base_name]))
            print(f"Matched: {base_name}")

    return matched_pairs

def analyze_label_distribution(matched_pairs):
    
    label_counts = []
    
    
    for label_file, _ in matched_pairs:
        with open(os.path.join(LABELS_DIR, label_file), "r") as file:
            num_labels = len(file.readlines())
            label_counts.append(num_labels)
    
    
    min_labels = min(label_counts)
    max_labels = max(label_counts)
    
    
    distribution = {}
    for count in range(min_labels, max_labels + 1):
        num_images = label_counts.count(count)
        if num_images > 0:  
            distribution[count] = num_images
    
    
    df = pd.DataFrame.from_dict(distribution, orient='index', columns=['Number of Images'])
    df.index.name = 'Number of Labels'
    
    return df

def count_solar_panels(matched_pairs):
    """Counts total instances of solar panels and label distribution per image."""
    total_instances = 0
    label_counts = []  

    for label_file, _ in matched_pairs:
        with open(os.path.join(LABELS_DIR, label_file), "r") as file:
            print(f"Reading: {label_file}")
            lines = file.readlines()
            print(f"Labels: {len(lines)}")
            total_instances += len(lines)
            label_counts.append(len(lines))

    return total_instances, label_counts


matched_pairs = match_labels_to_images()

total_instances, label_counts = count_solar_panels(matched_pairs)

label_distribution = analyze_label_distribution(matched_pairs)

with open('Q1_results.txt', 'w') as f:
    
    f.write(f"Total number of solar panel instances: {total_instances}\n\n")
    
    
    f.write("Distribution of labels per image:\n")
    f.write("--------------------------------\n")
    f.write(label_distribution.to_string())
    
print(f"\nResults have been saved to Q1_results.txt")
