import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Config ---
MANIFEST_FILE = "manifest.csv"
IMG_DISPLAY_SIZE = (256, 256)
IMG_ORIGINAL_SIZE = (640, 640)

# (The 'load_image' and 'read_annotations' functions are the same as before)

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image at {path}")
        return None
    return cv2.resize(img, IMG_DISPLAY_SIZE)

def read_annotations(txt_path):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
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
    print("Starting EDA using manifest file...")

    # Check if the manifest file exists
    if not os.path.exists(MANIFEST_FILE):
        print(f"Error: Manifest file '{MANIFEST_FILE}' not found.")
        print("Please run 'create_manifest.py' first.")
    else:
        # Read the manifest CSV into a pandas DataFrame
        manifest_df = pd.read_csv(MANIFEST_FILE)
        
        # Select one random row (one random pair) from the DataFrame
        sample = manifest_df.sample(1).iloc[0]
        
        # Get the paths directly from our manifest
        test_path = sample['test_path']
        temp_path = sample['template_path']
        txt_path = sample['annotation_path']
        
        print(f"Checking sample: {os.path.basename(test_path)}")

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

    print("EDA Finished.")