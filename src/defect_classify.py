import os
import cv2

# --- File and Folder Paths ---
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
        print(f"Error: '{clean_file_list}' not found. Please run validate_dataset.py first.")
        exit()

    # Create a fast lookup map from test.txt
    annotation_map = {}
    with open(map_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                # Normalize keys to use only forward slashes
                key = parts[0].replace('\\', '/')
                value = parts[1].replace('\\', '/')
                annotation_map[key] = value

    # Create output folders for each defect type
    for defect_name in DEFECT_MAP.values():
        os.makedirs(os.path.join(output_dir, defect_name), exist_ok=True)

    with open(clean_file_list, "r") as f:
        valid_base_paths = [line.strip() for line in f]

    print(f"Found {len(valid_base_paths)} valid image pairs to process.")

    total_rois_saved = 0
    # Loop through each valid base path
    for base_path in valid_base_paths:
        
        # --- ROBUST PATH CORRECTION ---
        # Normalize the path to use only forward slashes
        normalized_path = base_path.replace('\\', '/')
        # Find the start of the 'group...' part of the path
        try:
            key_start_index = normalized_path.find('group')
            if key_start_index == -1:
                continue # Skip if "group" isn't in the path
            
            # Create the key for the map (e.g., "group00041/00041/00041000.jpg")
            key_part = normalized_path[key_start_index:]
            image_map_key = key_part + '.jpg'
        except Exception:
            continue
        # --- END OF CORRECTION ---

        # Find the annotation path from the map
        annotation_rel_path = annotation_map.get(image_map_key)

        if not annotation_rel_path:
            continue

        test_path = base_path + "_test.jpg"
        txt_path = os.path.join(data_dir, annotation_rel_path)

        # Load image and process annotations
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
                        # Crop the defect and save it
                        roi = test_img[y1:y2, x1:x2]
                        if roi.size > 0:
                            img_name = os.path.basename(base_path)
                            roi_filename = f"{img_name}_roi_{total_rois_saved}.jpg"
                            save_path = os.path.join(output_dir, defect_name, roi_filename)
                            cv2.imwrite(save_path, roi)
                            total_rois_saved += 1

    print(f"\n--- Processing Complete ---")
    print(f"Successfully generated and saved {total_rois_saved} labeled ROIs.")
    print(f"Your dataset for Module 3 is ready in: '{output_dir}'")