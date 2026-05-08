import os
import numpy as np
from PIL import Image
import csv
import cv2
import json

def dem_to_csv(dem_path, output_csv, min_elevation=0.0, max_elevation=0.0254):
    """
    Convert a DEM grayscale image to a CSV file containing elevation values in meters.
    """
    dem_dir = os.path.dirname(dem_path)
    vertical_dir = os.path.dirname(dem_dir)  # parent of resized_dem is vertical
    scan_dir = os.path.dirname(vertical_dir)  # parent of vertical is scan folder
    scan_name = os.path.basename(scan_dir)
    meta_path = os.path.join(vertical_dir, scan_name + "_meta.json")
    if not os.path.exists(meta_path):
        # fallback: try scan_dir (legacy)
        meta_path_legacy = os.path.join(scan_dir, scan_name + "_meta.json")
        if os.path.exists(meta_path_legacy):
            meta_path = meta_path_legacy
        else:
            raise FileNotFoundError(f"Meta file not found at {meta_path} or {meta_path_legacy}")
    meta = json.load(open(meta_path))
    min_elevation = meta["ele_min"] / 1000
    max_elevation = meta["ele_max"] / 1000
    dem_bits = meta.get("dem_bits", 8)

    # Read DEM as 16-bit if available
    if dem_bits == 16 or dem_path.lower().endswith('.png'):
        dem_img = Image.open(dem_path)
        dem_array = np.array(dem_img, dtype=np.uint16)
        elevation_data = (dem_array / 65535.0) * (max_elevation - min_elevation) + min_elevation
    else:
        dem_img = Image.open(dem_path).convert("L")
        dem_array = np.array(dem_img)
        elevation_data = (dem_array / 255.0) * (max_elevation - min_elevation) + min_elevation

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(elevation_data)

def convert_all_dem_images(dem_folder, output_folder):
    """
    Convert all DEM images in a folder to CSV elevation files.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in sorted(os.listdir(dem_folder)):
        if filename.endswith("DEM.png"):
            dem_path = os.path.join(dem_folder, filename)
            base_name = os.path.splitext(filename)[0]
            output_csv = os.path.join(output_folder, f"{base_name}.csv")

            # Skip if output already exists
            if os.path.exists(output_csv):
                print(f"Skipping {output_csv} (already exists)")
                continue

            dem_to_csv(dem_path, output_csv)

    print("All DEM images have been converted to CSV.")
