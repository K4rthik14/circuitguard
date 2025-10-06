# %% [markdown]
# # PCB Defect Detection Notebook
# This notebook reads template and defective images from the dataset and visualizes the defects.

# %%
# Install dependencies (run only once)
pip install opencv-python matplotlib numpy scikit-image

# %%
# Imports
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
# Dataset directory
DATA_DIR = r"/home/karthik/projects/circuitguard/dataset/PCBData"

# %%
# List all groups
groups = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print("Total groups:", len(groups))
print("First few groups:", groups[:5])

# %%
# Pick first group for demonstration
group = os.path.join(DATA_DIR, groups[0])
print("Using group:", group)

# %%
# Step 5: Collect template and test images robustly

# Templates: all images in subfolders NOT ending with '_not'
template_files = []
# Tests/defective: images in subfolders ending with '_not'
test_files = []

for sub in os.listdir(group):
    sub_path = os.path.join(group, sub)
    if os.path.isdir(sub_path):
        if sub.lower().endswith("_not"):
            test_files.extend(glob.glob(os.path.join(sub_path, "*.jpg")))
        else:
            template_files.extend(glob.glob(os.path.join(sub_path, "*.jpg")))

# Debug prints
print(f"Templates found ({len(template_files)}):", template_files)
print(f"Test images found ({len(test_files)}):", test_files)

# Error handling
if not template_files:
    raise FileNotFoundError(f"No template images found in {group}")
if not test_files:
    raise FileNotFoundError(f"No test images found in {group}")

# Pick first template and test
template_path = template_files[0]
test_path     = test_files[0]
print("Template:", template_path)
print("Test:", test_path)

# %%
# Read images as grayscale
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
test     = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

# %%
# Resize test to match template and compute difference
test_resized = cv2.resize(test, (template.shape[1], template.shape[0]))
diff = cv2.absdiff(template, test_resized)

# %%
# Threshold difference to get defect mask
_, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# %%
# Visualize template, test, difference, and defect mask
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(template, cmap='gray'); axs[0].set_title("Template"); axs[0].axis("off")
axs[1].imshow(test, cmap='gray'); axs[1].set_title("Test"); axs[1].axis("off")
axs[2].imshow(diff, cmap='gray'); axs[2].set_title("Difference"); axs[2].axis("off")
axs[3].imshow(mask, cmap='gray'); axs[3].set_title("Defect Mask"); axs[3].axis("off")
plt.show()
