import os
import cv2

# --- File and Folder Paths ---
# Assumes the script is in the 'src' folder
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
data_dir = os.path.join(project_root, 'data', 'raw')
map_file = os.path.join(project_root, 'data', 'test.txt')
output_dir = os.path.join(project_root, 'outputs', 'labeled_rois_jpeg')

# --- Defect Type Mapping ---
DEFECT_MAP = {
    1: 'open',
    2: 'short',
    3: 'mousebite',
    4: 'spur',
    5: 'copper',
    6: 'pin-hole'
}

if __name__ == "__main__":
    print("--- Starting Labeled ROI Generation ---")

    if not os.path.exists(map_file):
        print(f"Error: Map file '{map_file}' not found.")
        exit()

    # Create output folders for each defect type
    for defect_name in DEFECT_MAP.values():
        os.makedirs(os.path.join(output_dir, defect_name), exist_ok=True)

    with open(map_file, "r") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} entries in map file. Processing all valid pairs...")

    total_rois_saved = 0
    # Loop through each line in the map file, which is our source of truth
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        # Get the relative paths from the map file
        image_rel_path, annotation_rel_path = parts
        
        # Create the base name (e.g., "group00041/00041/00041000")
        base_name_path = image_rel_path.replace(".jpg", "")

        # Construct the full, absolute paths for all three required files
        test_path = os.path.join(data_dir, base_name_path + "_test.jpg")
        temp_path = os.path.join(data_dir, base_name_path + "_temp.jpg")
        txt_path = os.path.join(data_dir, annotation_rel_path)

        # Validate that all three files actually exist before processing
        if not (os.path.exists(test_path) and os.path.exists(temp_path) and os.path.exists(txt_path)):
            continue # If any file is missing, skip to the next entry

        # Load the test image
        test_img = cv2.imread(test_path)
        if test_img is None:
            continue

        # Read the annotation file to get defect locations and types
        with open(txt_path, "r") as f_ann:
            for ann_line in f_ann:
                ann_parts = ann_line.strip().split(',')
                if len(ann_parts) == 5:
                    x1, y1, x2, y2, defect_id = map(int, ann_parts)
                    defect_name = DEFECT_MAP.get(defect_id)
                    if defect_name:
                        # Crop the defect from the image
                        roi = test_img[y1:y2, x1:x2]
                        if roi.size > 0: # Ensure the cropped image is not empty
                            img_name = os.path.basename(base_name_path)
                            # Save as .jpg
                            roi_filename = f"{img_name}_roi_{total_rois_saved}.jpg"
                            save_path = os.path.join(output_dir, defect_name, roi_filename)
                            cv2.imwrite(save_path, roi)
                            total_rois_saved += 1

    print(f"\n--- Processing Complete ---")
    print(f"Successfully generated and saved {total_rois_saved} labeled ROIs.")
    print(f"Your dataset for Module 3 is ready in: '{output_dir}'")
    