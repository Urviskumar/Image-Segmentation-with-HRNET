import tensorflow as tf
import tensorflow_hub as tfhub
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image

def load_hrnet_model():
    """
    Load the HRNet model for object segmentation using TensorFlow Hub.

    Returns:
        hub_model: Loaded HRNet model.
    """
    hub_model = tfhub.load('https://tfhub.dev/google/HRNet/coco-hrnetv2-w48/1')
    return hub_model

def generate_squirrel_masks(squirrel_folder, selected_squirrels, mask_folder, hub_model):
    """
    Generate mask images for selected squirrel images using object segmentation.

    Args:
        squirrel_folder (str): Folder path containing squirrel images.
        selected_squirrels (list): List of selected squirrel image filenames.
        mask_folder (str): Folder path to save the mask images.
        hub_model: Loaded HRNet model for object segmentation.
    """
    os.makedirs(mask_folder, exist_ok=True)

    for image_file in selected_squirrels:
        # Read and preprocess the image
        image_path = os.path.join(squirrel_folder, image_file)
        squirrel_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # Make predictions using the model
        predictions = hub_model.predict([squirrel_image.astype(np.float32) / 255.])

        # Create a mask image
        mask = np.squeeze(predictions)[:, :, 18] > 0.1
        mask_image = np.zeros_like(squirrel_image)  # Initialize mask with black pixels

        if np.any(mask):
            mask_image[mask] = [255, 255, 255]  # Set masked pixels to white

        # Save the mask image as JPEG
        mask_file = os.path.join(mask_folder, f'mask_{image_file[:-4]}.jpg')
        mask_image_pil = Image.fromarray(mask_image)
        mask_image_pil.save(mask_file, format='JPEG')

def draw_bounding_boxes(image, mask):
    """
    Draw bounding boxes around the contours of the mask on the image.

    Args:
        image: Original image.
        mask: Mask image.

    Returns:
        bbox_image: Image with bounding boxes drawn.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox_image = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return bbox_image

def main():
    # Load the HRNet model
    hrnet_model = load_hrnet_model()

    # Define the folder and files for squirrels
    squirrel_folder = 'Squirrels_BirdFeeders'
    squirrel_files = os.listdir(squirrel_folder)
    selected_squirrels = np.array(squirrel_files)[np.random.permutation(len(squirrel_files))[:5]]

    # Create a folder to save the mask images
    mask_folder = 'SquirrelMasks'

    # Generate mask images for selected squirrel images
    generate_squirrel_masks(squirrel_folder, selected_squirrels, mask_folder, hrnet_model)

    # Display the original, masked, and bounding box images
    for image_file in selected_squirrels:
        # Read the original, masked, and bounding box images
        image_path = os.path.join(squirrel_folder, image_file)
        mask_path = os.path.join(mask_folder, f'mask_{image_file[:-4]}.jpg')

        squirrel_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask_image = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

        bbox_image = draw_bounding_boxes(squirrel_image, mask_image)

        # Display the images
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(squirrel_image)
        plt.title('Original Image of Squirrels')
        plt.subplot(1, 3, 2)
        plt.imshow(mask_image)
        plt.title('Masked Image of Squirrels')
        plt.subplot(1, 3, 3)
        plt.imshow(bbox_image)
        plt.title('Images with Bounding Boxes')
        plt.show()

# Call the main function
if __name__ == "__main__":
    main()
