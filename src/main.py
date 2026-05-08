import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil

from elevation2csv import convert_all_dem_images
from vertical_displacement import compute_vertical_displacement, vertical_displacement_looping
from visualize_results import visualize_looping, visualize_vertical_displacement
from segmentation2binarymask import convert_all_masks, image_to_csv
from unet import process_segmentation, test_model
from resize_dem_and_ortho import split_dem_image
from resize_dem_and_ortho import split_testing_images
from reassemble_labeledRGB_images import reassemble_image
from pointcloud2orthoimage import p2o_main

from PIL import Image
import os

from PIL import Image
import os

def get_image_dimensions(image_path):
    # Open the image to get its dimensions
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None


# New function to run the pipeline for a given GSDmm2px and result type
def run_pipeline(base_path, sidewalk_name, las_file_path, GSDmm2px, result_type):
    import pointcloud2orthoimage
    # Use a dedicated subfolder for each pipeline
    pipeline_folder = os.path.join(base_path, sidewalk_name, result_type.replace('_results', ''))
    os.makedirs(pipeline_folder, exist_ok=True)

    # Run pointcloud2orthoimage with the actual .las file path, outputting to the pipeline folder
    pointcloud2orthoimage.main2(las_file_path, pointName=sidewalk_name, output_dir=pipeline_folder, GSDmm2px=GSDmm2px, b='win')

    results_path = os.path.join(pipeline_folder, "results")
    os.makedirs(results_path, exist_ok=True)
    labeled_rgb_with_measurements_path = os.path.join(results_path, "labeled_rgb")
    os.makedirs(labeled_rgb_with_measurements_path, exist_ok=True)

    # original_dem_path = os.path.join(pipeline_folder, sidewalk_name + "DEM.png")
    original_dem_path = os.path.join(pipeline_folder, sidewalk_name + "DEM.jpg")
    original_RGB_path = os.path.join(pipeline_folder, sidewalk_name + "RGB.jpg")

    sidewalk_output_folder_rgb = os.path.join(pipeline_folder, "resized_rgb")
    sidewalk_output_folder_dem = os.path.join(pipeline_folder, "resized_dem")
    os.makedirs(sidewalk_output_folder_rgb, exist_ok=True)
    os.makedirs(sidewalk_output_folder_dem, exist_ok=True)

    pretrained_horizontal_model_path = '../horizontal_unet_membrane.hdf5'
    pretrained_vertical_model_path = '../vertical_unet_membrane.hdf5'
    img_size = 256

    split_dem_image(original_dem_path, sidewalk_output_folder_dem)
    split_testing_images(original_RGB_path, sidewalk_output_folder_rgb)


    # Use the correct model for each pass
    if result_type == "horizontal_results":
        process_segmentation(pretrained_horizontal_model_path, sidewalk_output_folder_rgb, img_size, os.path.join(pipeline_folder, "labeled_prediction"))
    else:
        process_segmentation(pretrained_vertical_model_path, sidewalk_output_folder_rgb, img_size, os.path.join(pipeline_folder, "labeled_prediction"))

    predicted_seg_label_path = os.path.join(pipeline_folder, "labeled_prediction")
    binary_mask_csv_path = results_path
    vertical_displacement_csv = results_path

    convert_all_masks(predicted_seg_label_path, binary_mask_csv_path)
    # Pass the resized_dem folder as dem_folder so DEM lookup works correctly
    vertical_displacement_looping(predicted_seg_label_path, sidewalk_output_folder_dem, binary_mask_csv_path, vertical_displacement_csv)

    # Visualize horizontal or vertical displacement
    if result_type == "horizontal_results":
        visualize_looping(sidewalk_output_folder_rgb, binary_mask_csv_path, vertical_displacement_csv, results_path, mode="horizontal")
    else:
        visualize_looping(sidewalk_output_folder_rgb, binary_mask_csv_path, vertical_displacement_csv, results_path, mode="vertical")

    elevation_csv = os.path.join(pipeline_folder, "elevation_data")
    convert_all_dem_images(sidewalk_output_folder_dem, elevation_csv)

    # Look for the correct meta file name (scan_name + '_meta.json')
    meta_path = os.path.join(pipeline_folder, sidewalk_name + '_meta.json')
    if not os.path.exists(meta_path):
        print(f"[Warning] Meta file not found: {meta_path}")

    image_path = original_RGB_path
    width, height = get_image_dimensions(image_path)
    reassemble_image(labeled_rgb_with_measurements_path, results_path, width, height)

    # Move to measured_sidewalks/scanName/result_type if needed (optional, not moving now)
# Run this entire program by running python main.py 

if __name__ == "__main__":
    base_path = "/Users/Lunar/pointcloud_files/" # only change this
    demo_path = os.path.join(base_path, "Demo")
    # Find all .las files in base_path
    all_las = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.las')]
    print(f"Found {len(all_las)} .las files in {base_path}")

    for las_file_path in all_las:
        #check to see if results already exist for this scan, if so, skip
        scan_name = os.path.splitext(os.path.basename(las_file_path))[0]
        final_scan_folder = os.path.join(base_path, scan_name)
        if os.path.exists(final_scan_folder):
            print(f"Results for {scan_name} already exist. Skipping.")
            continue

        print(f"Processing scan: {scan_name}")
        scan_folder = os.path.join(demo_path, scan_name)
        os.makedirs(scan_folder, exist_ok=True)
        # First pass: vertical
        run_pipeline(demo_path, scan_name, las_file_path, GSDmm2px=5, result_type="vertical_results")
        # Second pass: horizontal
        run_pipeline(demo_path, scan_name, las_file_path, GSDmm2px=1, result_type="horizontal_results")

        #move results to base_path/scanName
        shutil.move(scan_folder, final_scan_folder)
        print(f"Finished processing {scan_name}. Results moved to {final_scan_folder}")
        







# I need to first split the images, DEM and RGB. Therefore each DEM will have a dedicated RGB 
# Then we need to loop through each RGB image and predict the segmentation for each RGB cut image, and save somewere easy to retrieve
# then, for each RGB image, we need to make a csv binary mask file with appropiate names.
# then, for each dem image, we will need to loop through each dem&predicted labels and glue and compute a vertical displacement 
# Then we need to overlay the image and save it wwhile looping through each predicted dem
# then we need to glue visualization image together


# Notes, as of now, resized dems are stored in the original folder while resized RGB have their own folder. Consider moving everything to parent folder