import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# --- Config ---
DATA_DIR = "data"
IMG_DISPLAY_SIZE = (640, 640)
IMG_ORIGINAL_SIZE = (640, 640)

# Function to load an image
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image at {path}")
        return None
    return cv2.resize(img, IMG_DISPLAY_SIZE)

# Function to read the text file with defect locations
def read_annotations(txt_path):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes

    # Get ratios to scale the boxes for the smaller display image
    w_ratio = IMG_DISPLAY_SIZE[0] / IMG_ORIGINAL_SIZE[0]
    h_ratio = IMG_DISPLAY_SIZE[1] / IMG_ORIGINAL_SIZE[1]

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 5:
                x1, y1, x2, y2, _ = map(int, parts)
                
                # Convert to (x, y, w, h) and scale
                new_x = int(x1 * w_ratio)
                new_y = int(y1 * h_ratio)
                new_w = int((x2 - x1) * w_ratio)
                new_h = int((y2 - y1) * h_ratio)
                boxes.append((new_x, new_y, new_w, new_h))
    return boxes

# --- Main Script ---
if __name__ == "__main__":

    # Find all the group folders
    groups = [g for g in os.listdir(DATA_DIR) if g.startswith("group")]
    if not groups:
        print("Error: No group folders found in 'data' directory.")
    else:
        # Pick a random group and image to check
        group_name = random.choice(groups)
        group_number = group_name.replace("group", "")
        image_folder_path = os.path.join(DATA_DIR, group_name, group_number)
        
        test_files = [f for f in os.listdir(image_folder_path) if f.endswith("_test.jpg")]
        if test_files:
            base_name = random.choice(test_files).replace("_test.jpg", "")
            print(f"Checking sample: {base_name}")

            # Define the paths for the image pair and annotation
            test_path = os.path.join(image_folder_path, f"{base_name}_test.jpg")
            temp_path = os.path.join(image_folder_path, f"{base_name}_temp.jpg")
            txt_path = os.path.join(image_folder_path, f"{base_name}.txt")

            # Load images and annotations
            template_img = load_image(temp_path)
            test_img = load_image(test_path)
            scaled_boxes = read_annotations(txt_path)

            # Display the images
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.imshow(template_img, cmap='gray'); plt.title('Template')
            plt.subplot(1, 2, 2); plt.imshow(test_img, cmap='gray'); plt.title('Test')
            plt.show()

            # Display the test image with boxes drawn on it
            if scaled_boxes:
                test_with_boxes = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
                for (x, y, w, h) in scaled_boxes:
                    cv2.rectangle(test_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                plt.figure(figsize=(6, 6))
                plt.imshow(test_with_boxes)
                plt.title("Annotated Defects")
                plt.show()
        else:
            print(f"No test images found in {image_folder_path}")

    print("EDA Finished.")