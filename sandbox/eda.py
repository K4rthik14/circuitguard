 # eda.py
import os
import cv2
import random
import matplotlib.pyplot as plt

def visualize_random_sample(dataset_path):
    """
    Loads a random test image, its template, and its annotations,
    then displays them for inspection.
    """
    try:
        image_dir = os.path.join(dataset_path, 'images')
        annotation_dir = os.path.join(dataset_path, 'annotations', 'json') # Annotations are in json

        # Get a list of all test images
        all_files = os.listdir(image_dir)
        test_images = [f for f in all_files if f.endswith('_test.jpg')]

        if not test_images:
            print("Error: No test images found. Check the dataset path and structure.")
            return

        # Pick a random image
        random_image_name = random.choice(test_images)
        base_name = random_image_name.replace('_test.jpg', '')

        # Construct file paths
        test_img_path = os.path.join(image_dir, f"{base_name}_test.jpg")
        temp_img_path = os.path.join(image_dir, f"{base_name}_temp.jpg")
        annotation_path = os.path.join(annotation_dir, f"{base_name}.json") # Path to json file

        # --- Load Images ---
        print(f"Loading sample: {base_name}")
        test_image = cv2.imread(test_img_path)
        template_image = cv2.imread(temp_img_path)

        if test_image is None or template_image is None:
            print(f"Error loading images for {base_name}")
            return

        # --- Load Annotations and Draw Bounding Boxes ---
        # The DeepPCB annotations are in the image itself (pre-labeled),
        # but for a real-world scenario, you'd parse an annotation file.
        # Here, we will just display the images to visually inspect alignment and quality.
        # If you have separate annotation files (like .txt or .json), you would parse them here.
        # For this dataset, the provided annotations are for training the classification model later.

        # --- Display the images ---
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Template: {base_name}_temp.jpg")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Test Image: {base_name}_test.jpg")
        plt.axis('off')

        plt.suptitle("Dataset Sample Inspection")
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: Could not find a file. Please check your paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # IMPORTANT: Update this path to where your PCB_DATASET folder is located
    DATASET_PATH = 'PCB_DATASET'
    visualize_random_sample(DATASET_PATH)