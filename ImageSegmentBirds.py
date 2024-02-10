import matplotlib.pyplot as plt
import cv2
from PIL import Image
import tensorflow as tf
import tensorflow_hub as tfhub
import numpy as np
import os

def load_hrnet_model():
    """
    Load the HRNet model for object segmentation using TensorFlow Hub.

    Returns:
        hub_model: Loaded HRNet model.
    """
    hub_model = tfhub.load('https://tfhub.dev/google/HRNet/coco-hrnetv2-w48/1')
    return hub_model

def generate_bird_masks(bird_folder, selected_birds, mask_folder, hub_model):
    """
    Generate mask images for selected bird images using object segmentation.

    Args:
        bird_folder (str): Folder path containing bird images.
        selected_birds (list): List of selected bird image filenames.
        mask_folder (str): Folder path to save the mask images.
        hub_model: Loaded HRNet model for object segmentation.
    """
    os.makedirs(mask_folder, exist_ok=True)

    for image_file in selected_birds:
        # Read and preprocess the image
        image_path = os.path.join(bird_folder, image_file)
        bird_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # Make predictions using the model
        predictions = hub_model.predict([bird_image.astype(np.float32) / 255.])

        # Create a mask image
        mask = np.squeeze(predictions)[:, :, 17] > 0.1
        mask_image = np.zeros_like(bird_image)  # Initialize mask with black pixels

        if np.any(mask):
            mask_image[mask] = [255, 255, 255]  # Set masked pixels to white

        # Save the mask image as JPEG
        mask_file = os.path.join(mask_folder, f'mask_{image_file[:-4]}.jpg')
        mask_image_pil = Image.fromarray(mask_image)
        mask_image_pil.save(mask_file, format='JPEG')

        # Find contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the contours
        bbox_image = bird_image.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the original, masked, and bounding box images
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(bird_image)
        plt.title('Original Image of Birds')
        plt.subplot(1, 3, 2)
        plt.imshow(mask_image)
        plt.title('Masked Image of Birds')
        plt.subplot(1, 3, 3)
        plt.imshow(bbox_image)
        plt.title('Images with Bounding Boxes')
        plt.show()

def main():
    # Load the HRNet model
    hrnet_model = load_hrnet_model()

    # Define the folder and files for birds
    bird_folder = 'Birds_BirdFeeders'
    bird_files = os.listdir(bird_folder)
    selected_birds = np.array(bird_files)[np.random.permutation(len(bird_files))[:5]]

    # Create a folder to save the mask images
    mask_folder = 'BirdMasks'

    # Generate mask images for selected bird images
    generate_bird_masks(bird_folder, selected_birds, mask_folder, hrnet_model)

# Call the main function
if __name__ == "__main__":
    main()
