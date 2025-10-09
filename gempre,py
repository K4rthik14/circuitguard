import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# Configuration
DATA_DIR = "data"
OUTPUT_DIR = os.path.join("outputs", "rois")

# Main Script
if __name__ == "__main__":
    print("Starting Preprocessing Test on a random sample...")

    # --- Find a random image pair from the dataset ---
    try:
        # Find all group folders
        groups = [g for g in os.listdir(DATA_DIR) if g.startswith("group")]
        if not groups:
            raise FileNotFoundError("No group folders found in 'data' directory.")

        # Pick a random group
        random_group = random.choice(groups)
        group_number = random_group.replace("group", "")
        image_folder_path = os.path.join(DATA_DIR, random_group, group_number)

        # Find all test images in that group
        test_files = [f for f in os.listdir(image_folder_path) if f.endswith("_test.jpg")]
        if not test_files:
            raise FileNotFoundError(f"No test images found in {image_folder_path}")

        # Pick a random test image and create the file paths
        random_test_file = random.choice(test_files)
        sample_base_name = random_test_file.replace("_test.jpg", "")
        
        template_path = os.path.join(image_folder_path, f"{sample_base_name}_temp.jpg")
        test_path = os.path.join(image_folder_path, f"{sample_base_name}_test.jpg")

        print(f"Randomly selected sample: {sample_base_name}")

    except FileNotFoundError as e:
        print(f"Error finding a random sample: {e}")
        exit() # Exit the script if no valid sample could be found
    # --- End of new section ---


    # 1. Load images
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if template_img is not None and test_img is not None:
        # 2. Find the difference between the images
        diff_img = cv2.absdiff(template_img, test_img)

        # 3. Apply a threshold to get a binary mask
        _, thresh_mask = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)

        # 4. Clean the mask to remove noise
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 5. Find contours of the defects
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Found {len(contours)} potential defects.")

        # 6. Crop and save the defects (ROIs)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 20: # Ignore very small contours
                x, y, w, h = cv2.boundingRect(cnt)
                roi = test_img[y:y+h, x:x+w]
                save_path = os.path.join(OUTPUT_DIR, f"{sample_base_name}_roi_{i+1}.png")
                cv2.imwrite(save_path, roi)
        
        print(f"Saved ROIs to '{OUTPUT_DIR}'")

        # 7. Visualizing the steps
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 2, 1); plt.title("Original Test"); plt.imshow(test_img, cmap='gray')
        plt.subplot(2, 2, 2); plt.title("Difference"); plt.imshow(diff_img, cmap='gray')
        plt.subplot(2, 2, 3); plt.title("Thresholded"); plt.imshow(thresh_mask, cmap='gray')
        plt.subplot(2, 2, 4); plt.title("Cleaned Mask"); plt.imshow(cleaned_mask, cmap='gray')
        plt.tight_layout(); plt.show()
    else:
        print(f"Error: Could not load sample images for {sample_base_name}")