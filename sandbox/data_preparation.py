import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# --- CONFIG ---
# This robustly finds the data directory relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, "..", "PCBData")

IMG_SIZE = (256, 256)   # Uniform resize for display
ORIGINAL_SIZE = (640, 640) # The dataset's standard image size
print("Data directory:", DATA_DIR)

# --- HELPERS ---
def load_image(path, gray=True):
    """Load image with optional grayscale conversion."""
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.resize(img, IMG_SIZE)
    return img

def visualize_pair(temp_path, test_path, boxes=None):
    """Show template, test, and optionally bounding boxes."""
    temp = load_image(temp_path)
    test = load_image(test_path)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(temp, cmap='gray')
    axes[0].set_title('Template')
    axes[1].imshow(test, cmap='gray')
    axes[1].set_title('Test (Defective)')
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    if boxes:
        # We draw on the RESIZED test image
        img_rgb = cv2.cvtColor(test, cv.COLOR_GRAY2BGR)
        for (x, y, w, h) in boxes:
            # The boxes are already scaled, so we can draw them directly
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 0, 255), 1)
        plt.figure(figsize=(4, 4))
        plt.imshow(img_rgb)
        plt.title(f"Defect Regions (on {IMG_SIZE[0]}x{IMG_SIZE[1]} image)")
        plt.axis("off")
        plt.show()


def read_and_scale_annotations(txt_path, original_dims, new_dims):
    """Read annotations and scale them from original to new dimensions."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes

    orig_w, orig_h = original_dims
    new_w, new_h = new_dims

    w_ratio = new_w / orig_w
    h_ratio = new_h / orig_h

    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                _, x, y, w, h = map(float, parts)
                
                
                # Scale the coordinates
                new_x = int(x * w_ratio)
                new_y = int(y * h_ratio)
                new_w = int(w * w_ratio)
                new_h = int(h * h_ratio)
                
                boxes.append((new_x, new_y, new_w, new_h))
    return boxes


# --- DATA EXPLORATION ---
 # --- DATA EXPLORATION (UPDATED) ---
def explore_dataset(base_dir):
    """
    Finds a random sample in the dataset and visualizes it,
    handling the nested folder structure.
    """
    if not os.path.isdir(base_dir):
        print(f"Error: Data directory not found at '{base_dir}'")
        return

    groups = [g for g in os.listdir(base_dir) if g.startswith("group")]
    if not groups:
        print(f"Error: No 'group...' folders found in '{base_dir}'")
        return

    print(f"🔍 Total groups found: {len(groups)}")
    group_name = random.choice(groups)
    sample_group_path = os.path.join(base_dir, group_name)
    print("📦 Exploring Group:", sample_group_path)

    # --- FIX: Handle the nested folder structure ---
    # The images are inside a subfolder named after the group's number.
    group_number = group_name.replace("group", "")
    image_folder_path = os.path.join(sample_group_path, group_number)

    if not os.path.isdir(image_folder_path):
        print(f"Error: Expected image subfolder not found at '{image_folder_path}'")
        return

    print(f"📂 Searching for images in: {image_folder_path}")
    img_files = [f for f in os.listdir(image_folder_path) if f.endswith("_test.jpg")]
    if not img_files:
        print("No test images found in this specific group!")
        return

    test_file = random.choice(img_files)
    base_name = test_file.replace("_test.jpg", "")

    # Construct paths using the correct, deeper image_folder_path
    test_path = os.path.join(image_folder_path, base_name + "_test.jpg")
    temp_path = os.path.join(image_folder_path, base_name + "_temp.jpg")
    txt_path = os.path.join(image_folder_path, base_name + ".txt")

    print(f"Template: {base_name}_temp.jpg")
    print(f"Test: {base_name}_test.jpg")
    print(f"Annotations: {base_name}.txt")

    # Read and scale the annotations from 640x640 to 256x256
    scaled_boxes = read_and_scale_annotations(txt_path, ORIGINAL_SIZE, IMG_SIZE)
    
    # Visualize the pair with the correctly scaled boxes
    visualize_pair(temp_path, test_path, scaled_boxes)

    # Read and scale the annotations from 640x640 to 256x256
    scaled_boxes = read_and_scale_annotations(txt_path, ORIGINAL_SIZE, IMG_SIZE)
    
    # Visualize the pair with the correctly scaled boxes
    visualize_pair(temp_path, test_path, scaled_boxes)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    explore_dataset(DATA_DIR)