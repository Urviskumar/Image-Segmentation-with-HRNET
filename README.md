The objective of this project is to segment images of birds and squirrels using the HRNet model for object segmentation. Two functions, generate_bird_masks and generate_squirrel_masks, process selected images, creating masks for birds and squirrels, respectively. Additionally, the project includes visualization of the segmentation results, showing the original images, and masked images around the detected objects.

IMPLEMENTATION OVERVIEW
The project employs the HRNet model, loaded using TensorFlow Hub, to perform object segmentation on images of birds and squirrels. The model is selected based on its efficiency in capturing detailed object boundaries.

1. Bird Image Segmentation:
The generate_bird_masks function processes selected bird images, generates masks, and saves them along with bounding box visualizations. The process involves preprocessing images, making predictions using the HRNet model, creating masks based on predictions, and saving the resulting masked images.

2. Squirrel Image Segmentation:
The generate_squirrel_masks function similarly processes selected squirrel images, generates masks, and saves them with bounding box visualizations. Image preprocessing, model prediction, mask creation, and saving are repeated for squirrel images.
