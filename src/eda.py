import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# --- Config ---
# This robustly finds the correct paths, assuming the script is in a 'src' folder
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
MAP_FILE = os.path.join(PROJECT_ROOT, 'data', 'test.txt') # Path to your map file

IMG_DISPLAY_SIZE = (640, 640)
IMG_ORIGINAL_SIZE = (640, 640)

def load_image(path):
    """Loads an image and resizes it for display."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image at {path}")
        return None
    return cv2.resize(img, IMG_DISPLAY_SIZE)

def read_annotations(txt_path):
    """Reads and scales annotations from a text file."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes

    # Since display and original size are the same, ratio is 1
    w_ratio = IMG_DISPLAY_SIZE[0] / IMG_ORIGINAL_SIZE[0]
    h_ratio = IMG_DISPLAY_SIZE[1] / IMG_ORIGINAL_SIZE[1]

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 5:
                x1, y1, x2, y2, _ = map(int, parts)
                new_x = int(x1 * w_ratio)
                new_y = int(y1 * h_ratio)
                new_w = int((x2 - x1) * w_ratio)
                new_h = int((y2 - y1) * h_ratio)
                boxes.append((new_x, new_y, new_w, new_h))
    return boxes

# --- Main Script ---
if __name__ == "__main__":
    print("Starting EDA...")

    if not os.path.exists(MAP_FILE):
        print(f"Error: Map file not found at '{MAP_FILE}'")
    else:
        # Read all pairs from the map file
        with open(MAP_FILE, "r") as f:
            lines = f.readlines()
        
        # Pick a random line (pair) from the map file
        random_line = random.choice(lines).strip().split()
        
        if len(random_line) == 2:
            image_path_from_map, annotation_path_from_map = random_line
            base_name_path = image_path_from_map.replace(".jpg", "")

            # Construct the full, correct paths
            test_path = os.path.join(DATA_DIR, base_name_path + "_test.jpg")
            temp_path = os.path.join(DATA_DIR, base_name_path + "_temp.jpg")
            txt_path = os.path.join(DATA_DIR, annotation_path_from_map)
            
            print(f"Checking sample: {os.path.basename(test_path)}")

            # Load images and annotations
            template_img = load_image(temp_path)
            test_img = load_image(test_path)
            scaled_boxes = read_annotations(txt_path)

            if template_img is not None and test_img is not None:
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
                    
                    plt.figure(figsize=(8, 8))
                    plt.imshow(test_with_boxes)
                    plt.title("Annotated Defects")
                    plt.show()

    print("EDA Finished.")