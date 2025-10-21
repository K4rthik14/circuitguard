import os
import shutil
import random


project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# This is the directory with your labeled class folders (copper, open, etc.)
base_dir = os.path.join(project_root, "outputs", "labeled_rois_jpeg")

# create train and eval directories
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Create the train and val directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print("Splitting dataset into 'train' and 'val' folders...")

# Find all the class folders (like 'copper', 'open')
# and ignore any 'train' or 'val' folders if they already exist
class_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d not in ["train", "val"]]

# Loop through each class folder
for cls in class_folders:
    src_class_dir = os.path.join(base_dir, cls)
    
    # Get a list of all images in the class folder
    images = [f for f in os.listdir(src_class_dir) if f.lower().endswith((".jpg", ".jpeg"))]
    random.shuffle(images) 
    
    # Split 80% for training, 20% for validation
    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    val_images = images[split_index:]

    #corresponding class folders inside 'train' and 'val'
    dest_train_dir = os.path.join(train_dir, cls)
    dest_val_dir = os.path.join(val_dir, cls)
    os.makedirs(dest_train_dir, exist_ok=True)
    os.makedirs(dest_val_dir, exist_ok=True)

    # Move the images to their new homes
    for img in train_images:
        shutil.move(os.path.join(src_class_dir, img), os.path.join(dest_train_dir, img))
    for img in val_images:
        shutil.move(os.path.join(src_class_dir, img), os.path.join(dest_val_dir, img))
        
    # After moving, the original class folder will be empty and can be removed
    os.rmdir(src_class_dir)

print("\n Dataset split complet")
print(f"Data is now organized in:\n- {train_dir}\n- {val_dir}")