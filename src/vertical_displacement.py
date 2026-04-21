import os
import numpy as np
import pandas as pd
import cv2
import json
from PIL import Image
from scipy.ndimage import label

def compute_vertical_displacement(predicted_path, dem_path, csv_path, output_csv, debug=False):
    """
    Calculate vertical displacement for each crack using normalized elevation values.
    - Find left and right edges of the crack.
    - Compute vertical displacement as the difference between max(right) - min(left).
    """
    # Load predicted segmentation (binary mask, where 1 represents a joint)
    predicted_img = Image.open(predicted_path).convert("L")
    predicted_array = np.array(predicted_img)

    dem_dir = os.path.dirname(dem_path)
    parent = os.path.dirname(dem_dir)
    meta_path = os.path.join(parent, os.path.basename(parent) + "_meta.json")
    meta = json.load(open(meta_path))

    # Load DEM elevation image (grayscale, where the pixel values represent heights)
    dem_bits = meta.get("dem_bits", 8)
    if dem_bits == 16 or dem_path.lower().endswith('.png'):
        dem_img = Image.open(dem_path)
        dem_array = np.array(dem_img, dtype=np.uint16)
        min_elevation = meta["ele_min"] / 1000  # Minimum real-world elevation (meters)
        max_elevation = meta["ele_max"] / 1000  # Maximum real-world elevation (meters)
        elevation_data = (dem_array / 65535.0) * (max_elevation - min_elevation) + min_elevation
    else:
        dem_img = Image.open(dem_path).convert("L")
        dem_array = np.array(dem_img)
        min_elevation = meta["ele_min"] / 1000  # Minimum real-world elevation (meters)
        max_elevation = meta["ele_max"] / 1000  # Maximum real-world elevation (meters)
        elevation_data = (dem_array / 255.0) * (max_elevation - min_elevation) + min_elevation

    # Load CSV mask (binary joint mask, 1 = joint, 0 = background)
    csv_data = pd.read_csv(csv_path, header=None).values  # Load as NumPy array

    # Ensure all arrays have the same shape
    if predicted_array.shape != elevation_data.shape or predicted_array.shape != csv_data.shape:
        raise ValueError("Image and CSV dimensions do not match!")

    # Step 1: Use connected component labeling to group adjacent 1s into cracks
    labeled_array, num_features = label(csv_data)  # Find connected regions of 1s (cracks)

    # List to store displacement for each crack
    displacements = []

    # Iterate over each unique labeled crack region
    for crack_label in range(1, num_features + 1):
        # Get the positions of this crack (all pixels with the current label)
        crack_mask = (labeled_array == crack_label)
        crack_positions = np.column_stack(np.where(crack_mask))

        # Find the left edge (closest 0 to the left of the crack)
        left_heights = []
        for y, x in crack_positions:
            left_x = x - 1
            while left_x >= 0 and csv_data[y, left_x] == 1:
                left_x -= 1
            if left_x >= 0:  # Valid left edge
                left_heights.append(elevation_data[y, left_x])

        # Find the right edge (closest 0 to the right of the crack)
        right_heights = []
        for y, x in crack_positions:
            right_x = x + 1
            while right_x < elevation_data.shape[1] and csv_data[y, right_x] == 1:
                right_x += 1
            if right_x < elevation_data.shape[1]:  # Valid right edge
                right_heights.append(elevation_data[y, right_x])

        # Ensure valid left and right edges were found
        if len(left_heights) == 0 or len(right_heights) == 0:
            continue  # Skip if no valid edge found

        # Get the min height for the left side and max height for the right side
        min_left_height = np.min(left_heights)
        max_right_height = np.max(right_heights)

        # Compute vertical displacement: max(right) - min(left)
        vertical_displacement = max_right_height - min_left_height

        # Calculate horizontal displacement (pixel length of crack mask)
        crack_xs = crack_positions[:, 1]
        min_x = np.min(crack_xs)
        max_x = np.max(crack_xs)
        pixel_length = max_x - min_x + 1
        horizontal_displacement = pixel_length * 1.0  # Each pixel is 1mm

        # Store result for this crack
        displacements.append([crack_label, vertical_displacement, horizontal_displacement])

    # Convert to DataFrame and save results
    df_displacements = pd.DataFrame(displacements, columns=["crack_label", "vertical_displacement", "horizontal_displacement"])
    df_displacements.to_csv(output_csv, index=False)

    print(f"Processed {len(displacements)} cracks.")
    print(f"Saved displacement data to {output_csv}")


def vertical_displacement_looping(seg_folder, dem_folder, csv_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all segmentation images
    for seg_filename in sorted(os.listdir(seg_folder)):
        if seg_filename.endswith("SEG.jpg"):
            seg_path = os.path.join(seg_folder, seg_filename)

            # Generate corresponding file paths
            dem_path = os.path.join(dem_folder, seg_filename.replace("SEG.jpg", "DEM.png"))
            csv_path = os.path.join(csv_folder, seg_filename.replace("SEG.jpg", "MASK.csv"))
            output_csv = os.path.join(output_folder, seg_filename.replace("SEG.jpg", "VERT_DISP.csv"))

            # Skip if output already exists
            if os.path.exists(output_csv):
                print(f"Skipping {output_csv} (already exists)")
                continue

            # Compute vertical displacement
            compute_vertical_displacement(seg_path, dem_path, csv_path, output_csv)

            print(f"Processed: {seg_filename} → {output_csv}")