import os
import cv2

# --- File and Folder Paths ---
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
        print(f"❌ Error: Map file '{map_file}' not found.")
        exit(1)

    # Create output folders for each defect type
    for defect_name in DEFECT_MAP.values():
        os.makedirs(os.path.join(output_dir, defect_name), exist_ok=True)

    with open(map_file, "r") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} entries in map file. Processing all valid pairs...\n")

    total_rois_saved = 0
    skipped_lines = 0

    for idx, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) != 2:
            print(f"⚠️ Line {idx}: Invalid format → {line.strip()}")
            skipped_lines += 1
            continue

        image_rel_path, annotation_rel_path = parts

        # --- Path Handling (auto _test fix) ---
        base_name, ext = os.path.splitext(image_rel_path)
        if not base_name.endswith("_test"):
            test_path = os.path.join(data_dir, base_name + "_test" + ext)
        else:
            test_path = os.path.join(data_dir, image_rel_path)

        temp_path = test_path.replace("_test.jpg", "_temp.jpg")
        txt_path = os.path.join(data_dir, annotation_rel_path)

        # --- Validate File Existence ---
        missing_files = []
        for p in [test_path, txt_path]:  # temp_path is optional now
            if not os.path.exists(p):
                missing_files.append(p)

        if missing_files:
            print(f"⚠️ Line {idx}: Missing files:")
            for mf in missing_files:
                print("   →", mf)
            skipped_lines += 1
            continue

        # --- Read Test Image ---
        test_img = cv2.imread(test_path)
        if test_img is None:
            print(f"⚠️ Line {idx}: Failed to load image: {test_path}")
            skipped_lines += 1
            continue

        # --- Read Annotation File ---
        with open(txt_path, "r") as f_ann:
            ann_lines = f_ann.readlines()

        if not ann_lines:
            print(f"⚠️ Line {idx}: Empty annotation file: {txt_path}")
            skipped_lines += 1
            continue

        for ann_line in ann_lines:
            ann_parts = ann_line.strip().split(',')
            if len(ann_parts) != 5:
                print(f"⚠️ Invalid annotation format in {txt_path}: {ann_line.strip()}")
                continue

            try:
                x1, y1, x2, y2, defect_id = map(int, ann_parts)
            except ValueError:
                print(f"⚠️ Non-integer values in annotation: {ann_line.strip()}")
                continue

            defect_name = DEFECT_MAP.get(defect_id)
            if not defect_name:
                print(f"⚠️ Unknown defect ID {defect_id} in {txt_path}")
                continue

            # Crop and save ROI
            roi = test_img[y1:y2, x1:x2]
            if roi.size == 0:
                print(f"⚠️ Empty ROI for coords ({x1},{y1})–({x2},{y2}) in {txt_path}")
                continue

            img_name = os.path.basename(test_path).replace("_test.jpg", "")
            roi_filename = f"{img_name}_roi_{total_rois_saved}.jpg"
            save_path = os.path.join(output_dir, defect_name, roi_filename)
            cv2.imwrite(save_path, roi)
            total_rois_saved += 1

        if idx % 100 == 0:
            print(f"✅ Processed {idx}/{len(lines)} entries... ROIs so far: {total_rois_saved}")

    # --- Summary ---
    print("\n--- Processing Complete ---")
    print(f"✅ Successfully saved {total_rois_saved} labeled ROIs.")
    print(f"⚠️ Skipped {skipped_lines} entries due to errors or missing files.")
    print(f"📂 Output directory: '{output_dir}'")
