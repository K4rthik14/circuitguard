import os
import cv2
import numpy as np

# --- File and Folder Paths ---
# This setup assumes the script is in the 'src' folder
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
data_dir = os.path.join(project_root, 'data', 'raw')
map_file = os.path.join(project_root, 'data', 'test.txt')
# Save to a new, clean folder
output_dir = os.path.join(project_root, 'outputs', 'all_unlabeled_rois_jpeg')

if __name__ == "__main__":
    print("--- Starting Generation of All Unlabeled ROIs ---")

    if not os.path.exists(map_file):
        print(f"Error: Map file '{map_file}' not found.")
        exit()

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    with open(map_file, "r") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} total entries in map file. Processing...")

    total_rois_saved = 0
    # Loop through each line in the map file, which is our single source of truth
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        image_rel_path, _ = parts
        base_name_path = image_rel_path.replace(".jpg", "")

        # Construct the full paths for the image pair
        test_path = os.path.join(data_dir, base_name_path + "_test.jpg")
        temp_path = os.path.join(data_dir, base_name_path + "_temp.jpg")

        # Load the images
        template_img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

        # Validate that both images were loaded correctly
        if template_img is None or test_img is None:
            continue

        # --- This is the proven pipeline from your preprocess.py ---

        # 1. Align images to prevent false differences
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        try:
            (_, warp_matrix) = cv2.findTransformECC(template_img, test_img, warp_matrix, warp_mode, criteria)
            aligned_test_img = cv2.warpAffine(test_img, warp_matrix, (template_img.shape[1], template_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            aligned_test_img = test_img # Use original if alignment fails

        # 2. Subtract, Threshold, and Clean
        diff_img = cv2.absdiff(template_img, aligned_test_img)
        _, thresh_mask = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3. Find defect contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Crop and save each ROI
        rois_found_in_image = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = aligned_test_img[y:y+h, x:x+w]
                if roi.size > 0:
                    img_name = os.path.basename(base_name_path)
                    # Save as .jpg
                    roi_filename = f"{img_name}_roi_{rois_found_in_image}.jpg"
                    save_path = os.path.join(output_dir, roi_filename)
                    cv2.imwrite(save_path, roi)
                    rois_found_in_image += 1
        
        total_rois_saved += rois_found_in_image
        print(f"Processed [{i+1}/{len(lines)}]: Found {rois_found_in_image} ROIs in {os.path.basename(test_path)}")

    print(f"\n--- Processing Complete ---")
    print(f"Successfully generated and saved {total_rois_saved} unlabeled ROIs.")
    print(f"Your ROIs are ready in: '{output_dir}'")