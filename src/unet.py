import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from PIL import Image
import os

def load_data_testing(img_path, img_size):
    """Load and preprocess a single image for prediction."""
    try:
        img = load_img(img_path, target_size=(img_size, img_size), color_mode='grayscale')  # Load as grayscale
        img_array = img_to_array(img) / 255.0  # Normalize the image (0 to 1)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array, img  # Return both the array and original image
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None, None

def test_model(model, img_path, img_size, output_path):
    """Test on an input image using a pre-loaded model and visualize the result."""
    try:
        print(f"Loading image: {img_path}")  # Debug print
        img_array, original_img = load_data_testing(img_path, img_size)
        if img_array is None:
            print(f"Failed to load image: {img_path}")
            return

        print("Making prediction...")
        prediction = model.predict(img_array)

        # Convert prediction to a binary mask
        pred_mask = np.squeeze(prediction)  # Remove batch dimension
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Threshold to 0 or 255

        print(f"Saving prediction to {output_path}")  # Debug print
        predicted_img = Image.fromarray(pred_mask, mode="L")  # Ensure grayscale format
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        predicted_img.save(output_path)
        print(f"Prediction image saved at: {output_path}")

        # Explicitly close PIL images to free resources
        if hasattr(predicted_img, 'close'):
            predicted_img.close()
        if hasattr(original_img, 'close'):
            original_img.close()

        # Explicitly delete large variables to free memory
        del img_array, original_img, prediction, pred_mask, predicted_img

    except Exception as e:
        print(f"Error in test_model: {e}")

    
def display_predicted_image(resized_rgb_path, predicted_seg_label_path):
    img = Image.open(resized_rgb_path)
    output_img = Image.open(predicted_seg_label_path)
    output_img = output_img.convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray")  # Show input image
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(output_img, cmap="gray")  # Show predicted image
    axes[1].set_title("Predicted Output")
    axes[1].axis('off')

    plt.show()

def process_segmentation(pretrained_model_path, sidewalk_output_folder_rgb, img_size, predicted_output_folder):
    """Loops through all RGB images, runs segmentation model, and saves predicted masks."""
    os.makedirs(predicted_output_folder, exist_ok=True)

    all_rgb_images = sorted([f for f in os.listdir(sidewalk_output_folder_rgb) if f.endswith("RGB.jpg")])

    print(f"Found {len(all_rgb_images)} images in {sidewalk_output_folder_rgb}")  # Debug print

    # Load the model only once
    print(f"Loading model from {pretrained_model_path}")
    model = load_model(pretrained_model_path)

    for img_filename in all_rgb_images:
        img_path = os.path.join(sidewalk_output_folder_rgb, img_filename)

        # Naming: RGB.jpg → SEG.jpg
        output_filename = img_filename.replace("RGB.jpg", "SEG.jpg")
        output_path = os.path.join(predicted_output_folder, output_filename)

        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"Skipping {output_filename} (already exists)")
            continue

        print(f"Processing: {img_filename} -> {output_filename}")  # Debug print
        test_model(model, img_path, img_size, output_path)

        # Explicitly clear Keras session if needed (uncomment if memory issues persist)
        from tensorflow.keras import backend as K
        K.clear_session()

        #display_predicted_image(img_path, output_path)
