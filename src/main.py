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


def main(base_path,sidewalk_name):
   
    # Necessary Paths
    # change this to the desired sidewalk
    sidewalk_path = os.path.join(base_path, sidewalk_name) # make your sidewalk structure similar to this

    # Don't change this
    # Define the results folder path
    results_path = os.path.join(sidewalk_path, "results")
    os.makedirs(results_path, exist_ok=True)
    labeled_rgb_with_measurements_path = os.path.join(results_path, "labeled_rgb") # a folder path containing all the cut RGB pictures with elevation measurements edited
    os.makedirs(labeled_rgb_with_measurements_path, exist_ok=True)

    # Input paths
    # original_dem_path = sidewalk_path + "/" + sidewalk_name +"DEM.JPG"
    original_dem_path = sidewalk_path + "/" + sidewalk_name +"DEM.png"
    original_RGB_path = sidewalk_path + "/" + sidewalk_name +"RGB.jpg"  # Path to the image you want to test

    sidewalk_output_folder_rgb = sidewalk_path + "/resized_rgb/" # for rgb
    sidewalk_output_folder_dem = sidewalk_path + "/resized_dem/" # for dem

    pretrained_model_path = '../unet_membrane.hdf5'

    # loop through rgb and dem images, name them by 12345
    resized_rgb_path = sidewalk_output_folder_rgb
    resized_dem_path = sidewalk_output_folder_dem # Don't enter original DEM path

    # also this, need to save the predicted image by numbers, predicted_{counter} + png
    predicted_seg_label_path = sidewalk_path + "/labeled_prediction/" # Need better naming for this


    # also this, need multiple csv files, naming should follow 12345
    binary_mask_csv_path = results_path  # Binary mask CSV

    # same here, naming follow 12345
    vertical_displacement_csv = results_path # displacement table aka results but in a table
    img_size = 256  # Set the image size to match the model input


    # split the DEM images
    # You input the DEM or Elevation image, and begins the process of splitting the images. Places in original sidewalk path
    split_dem_image(original_dem_path, sidewalk_output_folder_dem)

    # splits the RGB value
    split_testing_images(original_RGB_path, sidewalk_output_folder_rgb)


    # Have the u-net pre-trained model predict the labeled segmentation
    process_segmentation(pretrained_model_path, sidewalk_output_folder_rgb, img_size, predicted_seg_label_path)

    # Convert image to csv binary mask
    convert_all_masks(predicted_seg_label_path, binary_mask_csv_path)

    # Actually calculates the displacement in the areas that have segments
    vertical_displacement_looping(predicted_seg_label_path, resized_dem_path, binary_mask_csv_path, vertical_displacement_csv)

    # Visualize where the displacement occurs and mark displacement height
    # should visualize all images and stick them togther
    visualize_looping(resized_rgb_path, binary_mask_csv_path, vertical_displacement_csv, results_path)

    elevation_csv = sidewalk_path + "/elevation_data"
    convert_all_dem_images(resized_dem_path, elevation_csv)


    image_path = sidewalk_path + "/" + sidewalk_name + "RGB.jpg"  # Replace with your specific image file path
    width, height = get_image_dimensions(image_path)
    print(f"Width: {width}, Height: {height}")
    reassemble_image(labeled_rgb_with_measurements_path, results_path, width, height)
    # Move measured sidewalk into a different folder so we don't recalculate in the future
    
    measured_sidewalks_folder_path = os.path.dirname(base_path) + "/measured_sidewalks" # remove /demo and append /measured_sidewalks
    os.makedirs(measured_sidewalks_folder_path, exist_ok=True)
    shutil.move(sidewalk_path, measured_sidewalks_folder_path)
    
    
# Run this entire program by running python main.py 
if __name__ == "__main__":
    
    base_path = "/Users/Lunar/pointcloud_files/" # only change thiss
    p2o_main(base_path)
    base_path = os.path.join(base_path, "Demo")
    
    # Get all folder names inside base_path (only directories)
    all_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    # Loop through each sidewalk folder and call main with just the folder name
    for folder in all_folders:
        print(f"Found sidewalk: {folder}")  # Debugging output
        main(base_path, folder)







# I need to first split the images, DEM and RGB. Therefore each DEM will have a dedicated RGB 
# Then we need to loop through each RGB image and predict the segmentation for each RGB cut image, and save somewere easy to retrieve
# then, for each RGB image, we need to make a csv binary mask file with appropiate names.
# then, for each dem image, we will need to loop through each dem&predicted labels and glue and compute a vertical displacement 
# Then we need to overlay the image and save it wwhile looping through each predicted dem
# then we need to glue visualization image together


# Notes, as of now, resized dems are stored in the original folder while resized RGB have their own folder. Consider moving everything to parent folder