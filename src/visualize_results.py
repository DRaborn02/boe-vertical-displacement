import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
from PIL import Image, ImageOps
import os

def visualize_vertical_displacement(rgb_path, csv_path, displacement_csv_path, results_path, base_name):
    """
    Visualize cracks in the DEM and display the calculated vertical displacement for each crack.
    - Show cracks as outlined regions.
    - Display vertical displacement values for each crack.
    """
    os.makedirs(results_path, exist_ok=True)

    if not os.path.exists(rgb_path):
        print(f"Error: DEM file not found: {rgb_path}")
        return

    rgb_img = Image.open(rgb_path).convert("RGB")
    rgb_array = np.array(rgb_img)

    if not os.path.exists(csv_path):
        print(f"Error: CSV mask file not found: {csv_path}")
        return

    csv_data = pd.read_csv(csv_path, header=None).values

    if not os.path.exists(displacement_csv_path):
        print(f"Error: Displacement CSV file not found: {displacement_csv_path}")
        return

    displacement_df = pd.read_csv(displacement_csv_path)

    labeled_array, num_features = label(csv_data)

    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_array, interpolation="none")

    for crack_label in range(1, num_features + 1):
        crack_mask = (labeled_array == crack_label)
        crack_positions = np.column_stack(np.where(crack_mask))

        color = 'green'

        displacement_row = displacement_df[displacement_df['crack_label'] == crack_label]

        if not displacement_row.empty:
            displacement = displacement_row['vertical_displacement'].values[0]
            if displacement >= 0.002: # check if we should display it
                

                centroid_y = np.mean(crack_positions[:, 0])
                centroid_x = np.mean(crack_positions[:, 1])
                dy = crack_positions[-1, 0] - crack_positions[0, 0]
                dx = crack_positions[-1, 1] - crack_positions[0, 1]
                angle = np.degrees(np.arctan2(dy, dx))

                # Image dimensions and safe margin
                img_w, img_h = 256, 256
                margin = 20

                if centroid_x < margin:
                    centroid_x = margin
                elif centroid_x > img_w - margin:
                    centroid_x = img_w - margin

                if centroid_y < margin:
                    centroid_y = margin
                elif centroid_y > img_h - margin:
                    centroid_y = img_h - margin
                
                # plt.text(centroid_x, centroid_y, f"{displacement*1000:.3f}mm", color='blue', fontsize=25, rotation=angle, rotation_mode='anchor', clip_on=True)

                horizontal_disp = None
                if 'horizontal_displacement' in displacement_row.columns:
                    horizontal_disp = displacement_row['horizontal_displacement'].values[0]

                    #temporary
                    plt.scatter(crack_positions[:, 1], crack_positions[:, 0], c=color, s=5, label=f"Crack {crack_label}")

                # Draw horizontal displacement as a line at the bottom of the crack
                if horizontal_disp is not None:
                    # Find the bottom-most y of the crack
                    bottom_y = np.max(crack_positions[:, 0])
                    # Get all x at this y (bottom row of crack)
                    x_at_bottom = crack_positions[crack_positions[:, 0] == bottom_y, 1]
                    if len(x_at_bottom) == 0:
                        # fallback: use all x
                        x_at_bottom = crack_positions[:, 1]
                    left_x = np.min(x_at_bottom)
                    right_x = np.max(x_at_bottom)
                    # Clamp to image bounds
                    if left_x < margin:
                        left_x = margin
                    if right_x > img_w - margin:
                        right_x = img_w - margin
                    if bottom_y > img_h - margin:
                        bottom_y = img_h - margin
                    # Draw the horizontal line
                    plt.plot([left_x, right_x], [bottom_y, bottom_y], color='blue', linewidth=3)
                    # Place the label at the center x, but at the crack's centroid y
                    label_x = img_w / 2
                    label_y = centroid_y
                    plt.text(label_x, label_y, f"{horizontal_disp:.2f}mm", color='blue', fontsize=72, ha='center', va='center', rotation=0, clip_on=True)

                    #temporary
                    if horizontal_disp >= 13:
                        plt.scatter(crack_positions[:, 1], crack_positions[:, 0], c='red', s=5, label=f"Crack {crack_label}")
                    elif horizontal_disp >= 2:
                        plt.scatter(crack_positions[:, 1], crack_positions[:, 0], c=color, s=5, label=f"Crack {crack_label}")
            #temporary
            # if displacement >= 0.013:
            #     plt.scatter(crack_positions[:, 1], crack_positions[:, 0], c='red', s=5, label=f"Crack {crack_label}")
            # elif displacement >= 0.002:
            #     plt.scatter(crack_positions[:, 1], crack_positions[:, 0], c=color, s=5, label=f"Crack {crack_label}")
            

            

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.legend().set_visible(False)

    # Save temporary image
    temp_path = os.path.join(results_path, f"{base_name}_temp.png")
    plt.savefig(temp_path, dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # Load the saved image and process it to remove borders
    temp_img = Image.open(temp_path)

    # Crop off any potential border \
    temp_img = ImageOps.crop(temp_img, border=1)  # Adjust this if necessary

    # Resize the image to 256x256
    temp_img = temp_img.resize((256, 256), Image.Resampling.LANCZOS)

    # Save the final image
    final_path = os.path.join(results_path, "labeled_rgb", f"{base_name}VERT_DISP.png")
    temp_img.save(final_path)
    os.remove(temp_path)  # Clean up the temporary file

    print(f"Visualization saved: {final_path}")



def visualize_looping(dem_folder, csv_folder, displacement_folder, results_folder):
    """Loops through all displacement CSV files and visualizes vertical displacement."""
    
    # Ensure the results folder exists
    os.makedirs(results_folder, exist_ok=True)

    # Loop through all displacement CSV files
    for disp_filename in sorted(os.listdir(displacement_folder)):
        if disp_filename.endswith("VERT_DISP.csv"):  # Adjust file extension if needed
            displacement_csv_path = os.path.join(displacement_folder, disp_filename)

            # Generate corresponding file paths
            base_name = disp_filename.replace("VERT_DISP.csv", "")  # Extract base name
            dem_path = os.path.join(dem_folder, base_name + "RGB.jpg")
            csv_path = os.path.join(csv_folder, base_name + "MASK.csv")

            # Visualize vertical displacement
            visualize_vertical_displacement(dem_path, csv_path, displacement_csv_path, results_folder, base_name)

            print(f"Processed: {disp_filename} → Visualization saved")