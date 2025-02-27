import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio

# Paths to dataset
LABELS_DIR = "/fab3/btech/2022/snehanshu.pal22b/SolarPanel/labels/labels_native"
IMAGES_DIR = "/fab3/btech/2022/snehanshu.pal22b/SolarPanel/image_chips_native-20250212T103727Z-001/image_chips_native"

# Default meters-per-pixel if geospatial data is not available
DEFAULT_METERS_PER_PIXEL = 0.31  # Adjust based on dataset readme

def get_meters_per_pixel(tiff_path):
    """Extract pixel-to-meter conversion from a .tiff file if georeferenced."""
    try:
        with rasterio.open(tiff_path) as src:
            transform = src.transform
            meters_per_pixel = abs(transform.a)  # Scale factor in meters per pixel
            return meters_per_pixel
    except Exception as e:
        print(f"Could not read {tiff_path}: {e}")
        return DEFAULT_METERS_PER_PIXEL  # Use default if metadata isn't available

def match_labels_to_images():
    """Creates a mapping between label files and corresponding image files."""
    # Get all label files and image files
    label_files = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")])
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".tif")])
    
    # Create mapping from base name to full path
    label_map = {os.path.splitext(f)[0]: f for f in label_files}
    image_map = {os.path.splitext(f)[0]: f for f in image_files}

    # Find matching label-image pairs
    matched_pairs = []
    for base_name in label_map:
        if base_name in image_map:  # Check if corresponding image exists
            matched_pairs.append((label_map[base_name], image_map[base_name]))
            print(f"Matched: {base_name}")

    return matched_pairs

def count_solar_panels(matched_pairs):
    """Counts total instances of solar panels and label distribution per image."""
    total_instances = 0
    label_counts = []  # Stores count of labels per image

    for label_file, _ in matched_pairs:
        with open(os.path.join(LABELS_DIR, label_file), "r") as file:
            print(f"Reading: {label_file}")
            lines = file.readlines()
            print(f"Labels: {len(lines)}")
            total_instances += len(lines)
            label_counts.append(len(lines))

    return total_instances, label_counts

def compute_area_statistics(matched_pairs):
    """Computes area statistics for solar panels based on YOLO labels and images."""
    areas = []

    for label_file, image_file in matched_pairs:
        image_path = os.path.join(IMAGES_DIR, image_file)
        meters_per_pixel = get_meters_per_pixel(image_path)

        with open(os.path.join(LABELS_DIR, label_file), "r") as file:
            for line in file:
                parts = line.strip().split()
                _, _, _, width, height = map(float, parts)  # Extract YOLO bbox values
                
                # Convert from normalized YOLO format to meters
                width_m = width * meters_per_pixel
                height_m = height * meters_per_pixel
                
                area = width_m * height_m  # Compute area in square meters
                areas.append(area)

    # Compute statistics
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    return areas, mean_area, std_area

def plot_area_histogram(areas):
    """Plots the histogram of computed areas."""
    plt.figure(figsize=(8, 5))
    plt.hist(areas, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Area (sq. meters)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Solar Panel Areas")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# Match labels to images
matched_pairs = match_labels_to_images()

# Run computations
total_instances, label_counts = count_solar_panels(matched_pairs)
areas, mean_area, std_area = compute_area_statistics(matched_pairs)
print("Label Count Distribution:", label_counts)
# Display results
print(f"Total Solar Panel Instances: {total_instances}")
print("\nLabel Count Distribution:")
label_df = pd.DataFrame(pd.Series(label_counts).value_counts().sort_index(), columns=["Image Count"])
label_df.index.name = "Labels per Image"
print(label_df)

print(f"\nMean Area: {mean_area:.2f} sq. meters")
print(f"Standard Deviation: {std_area:.2f} sq. meters")

# Plot histogram
plot_area_histogram(areas)
