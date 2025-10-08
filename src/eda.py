import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# --- CONFIG ---
# This robustly finds the data directory relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
# Note: We point to the "data" folder now
DATA_DIR = os.path.join(script_dir, "..", "data")

IMG_SIZE = (256, 256)   # Uniform resize for display
ORIGINAL_SIZE = (640, 640) # The dataset's standard image size

def load_image(path, gray=True):
    """Loads an image from a given path and resizes it."""
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    img = cv2.resize(img, IMG_SIZE)
    return img

def read_and_scale_annotations(txt_path, original_dims, new_dims):
    """Reads annotations in (x1, y1, x2, y2, type) format and scales them."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    orig_w, orig_h = original_dims
    new_w, new_h = new_dims
    w_ratio = new_w / orig_w
    h_ratio = new_h / orig_h
    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(',') # The format uses commas
            if len(parts) == 5:
                x1, y1, x2, y2, _ = map(int, parts)
                w = x2 - x1
                h = y2 - y1
                new_x = int(x1 * w_ratio)
                new_y = int(y1 * h_ratio)
                new_w = int(w * w_ratio)
                new_h = int(h * h_ratio)
                boxes.append((new_x, new_y, new_w, new_h))
    return boxes

def visualize_pair(temp_path, test_path, boxes=None):
    """Shows the template, test, and optionally draws bounding boxes."""
    temp = load_image(temp_path)
    test = load_image(test_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(temp, cmap='gray')
    axes[0].set_title('Template Image')
    axes[1].imshow(test, cmap='gray')
    axes[1].set_title('Test (Defective) Image')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()
    if boxes:
        img_rgb = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in boxes:
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_rgb)
        plt.title(f"Defect Regions (on {IMG_SIZE[0]}x{IMG_SIZE[1]} image)")
        plt.axis("off")
        plt.show()

def explore_dataset(base_dir):
    """Finds and visualizes a random sample from the dataset."""
    if not os.path.isdir(base_dir):
        print(f"❌ Error: Data directory not found at '{base_dir}'")
        return
    groups = [g for g in os.listdir(base_dir) if g.startswith("group")]
    if not groups:
        print(f"❌ Error: No 'group...' folders found in '{base_dir}'")
        return
    print(f"✅ Found {len(groups)} groups.")
    group_name = random.choice(groups)
    sample_group_path = os.path.join(base_dir, group_name)
    group_number = group_name.replace("group", "")
    image_folder_path = os.path.join(sample_group_path, group_number)
    if not os.path.isdir(image_folder_path):
        print(f"❌ Error: Subfolder not found at '{image_folder_path}'")
        return
    print(f"📂 Exploring: {image_folder_path}")
    img_files = [f for f in os.listdir(image_folder_path) if f.endswith("_test.jpg")]
    if not img_files:
        print("🟡 No test images found in this group!")
        return
    test_file = random.choice(img_files)
    base_name = test_file.replace("_test.jpg", "")
    test_path = os.path.join(image_folder_path, base_name + "_test.jpg")
    temp_path = os.path.join(image_folder_path, base_name + "_temp.jpg")
    txt_path = os.path.join(image_folder_path, base_name + ".txt")
    print(f"📄 Sample files: {base_name}")
    scaled_boxes = read_and_scale_annotations(txt_path, ORIGINAL_SIZE, IMG_SIZE)
    visualize_pair(temp_path, test_path, scaled_boxes)

if __name__ == "__main__":
    print("🚀 Starting Dataset Exploration...")
    explore_dataset(DATA_DIR)
    print("✅ Exploration complete.")