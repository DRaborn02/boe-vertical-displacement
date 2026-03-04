import os
from PIL import Image


def split_dem_image(input_image_path, output_folder, patch_size=256):
    """Splits a single DEM image into 256x256 patches and stores them in the same folder."""
    
    os.makedirs(output_folder, exist_ok=True)

    patch_idx = 0  # Ensure consistent numbering

    # Open the DEM image
    with Image.open(input_image_path) as dem_img:
        width, height = dem_img.size

        # Loop through the image and extract patches
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = dem_img.crop((x, y, x + patch_size, y + patch_size))
                
                # Ensure patches are exactly 256x256 (handle edge cases like in split_testing_images)
                if patch.size != (patch_size, patch_size):
                    patch = patch.resize((patch_size, patch_size), Image.LANCZOS)
                
                # Save patches with continuous numbering
                output_path = os.path.join(output_folder, f"{patch_idx}DEM.png")
                patch.save(output_path)
                print(f"Saved: {output_path}")
                patch_idx += 1





def split_testing_images(input_image_path, output_folder, patch_size=256):  # Changed patch_size to 256
    # Create output folder for patches
    os.makedirs(output_folder, exist_ok=True)

    patch_idx = 0  # Start numbering patches

    with Image.open(input_image_path) as rgb_img:
        width, height = rgb_img.size

        # Loop through the image and extract patches
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = rgb_img.crop((x, y, x + patch_size, y + patch_size))
                
                # Ensure patches are exactly 256x256 (handle edge cases like in split_testing_images)
                if patch.size != (patch_size, patch_size):
                    patch = patch.resize((patch_size, patch_size), Image.LANCZOS)
                
                # Save patches with continuous numbering
                output_path = os.path.join(output_folder, f"{patch_idx}RGB.jpg")
                patch.save(output_path)
                print(f"Saved: {output_path}")
                patch_idx += 1


