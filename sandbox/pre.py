import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to your dataset
DATA_DIR = "../PCB_DATA"

# Function to list groups
def list_groups(base_dir):
    groups = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Total groups found: {len(groups)}")
    return groups

# Display sample images
def show_samples(group_dir):
    subfolders = [os.path.join(group_dir, sf) for sf in os.listdir(group_dir)]
    for sub in subfolders:
        files = [f for f in os.listdir(sub) if f.endswith(('.png', '.jpg', '.bmp'))]
        if len(files) == 0:
            continue
        img_path = os.path.join(sub, files[0])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(4, 4))
        plt.imshow(img_rgb)
        plt.title(f"Sample from {os.path.basename(sub)}")
        plt.axis('off')
        plt.show()
        break

# Basic image preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))  # Resize for uniformity
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# Perform simple image subtraction (if you want to show conceptually)
def sample_subtraction(template_path, defect_path):
    img1 = preprocess_image(template_path)
    img2 = preprocess_image(defect_path)

    if img1 is None or img2 is None:
        print("Error loading images.")
        return

    diff = cv2.absdiff(img2, img1)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Template')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Defective')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(thresh, cmap='gray')
    plt.title('Defect Mask')
    plt.axis('off')

    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    groups = list_groups(DATA_DIR)

    # Visualize one sample from a random group
    sample_group = os.path.join(DATA_DIR, groups[0])
    print(f"Showing samples from: {sample_group}")
    show_samples(sample_group)

    # Example subtraction demo (you can change these paths)
    template_path = os.path.join(DATA_DIR, "group00041", "00041_not", "00041_00.png")
    defect_path = os.path.join(DATA_DIR, "group00041", "00041", "00041_00.png")

    if os.path.exists(template_path) and os.path.exists(defect_path):
        sample_subtraction(template_path, defect_path)
    else:
        print("Sample image paths not found, please check file names.")
