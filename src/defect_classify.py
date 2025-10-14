import os
import cv2

# --- File and Folder Paths ---
# This setup assumes the script is in the 'src' folder
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
data_dir = os.path.join(project_root, 'data', 'raw')
map_file = os.path.join(project_root, 'data', 'test.txt')
clean_file_list = os.path.join(project_root, 'clean_file_list.txt')
output_dir = os.path.join(project_root, 'outputs', 'labeled_rois')

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
    print("--- Starting Labeled ROI Generation (Module 2) ---")

    if not os.path.exists(clean_file_list):
        print(f"Error: '{clean_file_list}' not found. Please run the validation script first.")
        exit()

    # Create a fast lookup map from test.txt, ensuring all keys use forward slashes
    annotation_map = {}
    with open(map_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                # Normalize the key by replacing any backslashes
                key = parts[0].replace('\\', '/')
                value = parts[1].replace('\\', '/')
                annotation_map[key] = value

    # Create output folders
    for defect_name in DEFECT_MAP.values():
        os.makedirs(os.path.join(output_dir, defect_name), exist_ok=True)

    with open(clean_file_list, "r") as f:
        valid_base_paths = [line.strip() for line in f]

    print(f"Found {len(valid_base_paths)} valid image pairs to process.")

    total_rois_saved = 0
    # Loop through each valid base path from the clean list
    for base_path in valid_base_paths:
        
        # --- THIS IS THE ROBUST PATH CORRECTION ---
        # 1. Normalize the path from the clean list to use only forward slashes
        normalized_path = base_path.replace('\\', '/')
        # 2. Get the part of the path relative to the 'data/raw' directory
        try:
            relative_key_part = os.path.relpath(normalized_path, os.path.join(project_root, 'data', 'raw')).replace('\\', '/')
        except ValueError:
            print(f"Warning: Unexpected path format in clean_file_list: {base_path}. Skipping.")
            continue
        # 3. Create the final key for the map
        image_map_key = relative_key_part + '.jpg'
        # --- END OF CORRECTION ---

        # Instantly find the annotation path from the map
        annotation_rel_path = annotation_map.get(image_map_key)

        if not annotation_rel_path:
            # This warning will tell us if a match is still failing
            print(f"Warning: Could not find annotation for key '{image_map_key}' in map. Skipping.")
            continue

        test_path = base_path + "_test.jpg"
        txt_path = os.path.join(data_dir, annotation_rel_path)

        # Load the image and process annotations
        test_img = cv2.imread(test_path)
        if test_img is None:
            continue

        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 5:
                    x1, y1, x2, y2, defect_id = map(int, parts)
                    defect_name = DEFECT_MAP.get(defect_id)
                    if defect_name:
                        roi = test_img[y1:y2, x1:x2]
                        img_name = os.path.basename(base_path)
                        roi_filename = f"{img_name}_roi_{total_rois_saved}.jpg" # Save as JPG
                        save_path = os.path.join(output_dir, defect_name, roi_filename)
                        cv2.imwrite(save_path, roi)
                        total_rois_saved += 1

    print(f"\n--- Processing Complete ---")
    print(f"Successfully generated and saved {total_rois_saved} labeled ROIs.")
    print(f"Your dataset for Module 3 is ready in: '{output_dir}'")