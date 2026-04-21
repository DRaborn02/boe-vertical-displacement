import os
import numpy as np
from PIL import Image
import csv

def image_to_csv(image_path, output_csv):
    """
    Convert a grayscale image to a CSV file where:
    - 0 represents the white background
    - 1 represents the black sidewalk joints
    """
    # Load the image in grayscale mode
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Convert white (255) to 0 and black (0) to 1
    binary_array = np.where(img_array < 128, 1, 0)  # Thresholding (anything darker than 128 -> 1, else 0)
    
    # Save as CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in binary_array:
            writer.writerow(row)
    
    print(f"CSV file saved: {output_csv}")

def convert_all_masks(predicted_output_folder, binary_mask_csv_folder):
    """Convert all segmentation masks (SEG.png) into CSV binary masks (MASK.csv)."""
    os.makedirs(binary_mask_csv_folder, exist_ok=True)

    all_predicted_masks = sorted([f for f in os.listdir(predicted_output_folder) if f.endswith("SEG.jpg")])

    for mask_filename in all_predicted_masks:
        mask_path = os.path.join(predicted_output_folder, mask_filename)

        # Naming: SEG.png → MASK.csv
        csv_filename = mask_filename.replace("SEG.jpg", "MASK.csv")
        csv_output_path = os.path.join(binary_mask_csv_folder, csv_filename)

        # Skip if output already exists
        if os.path.exists(csv_output_path):
            print(f"Skipping {csv_output_path} (already exists)")
            continue

        image_to_csv(mask_path, csv_output_path)