import cv2
import numpy as np
import os

# --- Configuration ---
CLEAN_LIST_FILE = "data/clean_file_list.txt"
OUTPUT_DIR = os.path.join("outputs", "rois_all") # Saving to a new folder

# --- Processing Functions ---

def process_image_pair(template_path, test_path, min_area=20):
    """
    Takes a template and test image path, finds defects, and returns
    a list of cropped defect images (ROIs).
    """
    # 1. Load images
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if template_img is None or test_img is None:
        # Silently skip if an image can't be read
        return [], None

    # 2. Find the difference
    diff_img = cv2.absdiff(template_img, test_img)

    # 3. Apply threshold
    _, thresh_mask = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)

    # 4. Clean the mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5. Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Crop ROIs
    rois = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = test_img[y:y+h, x:x+w]
            rois.append(roi)
            
    return rois, os.path.basename(test_path).replace("_test.jpg", "")

# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting to Process Entire Dataset ---")

    # Check if the clean list exists
    if not os.path.exists(CLEAN_LIST_FILE):
        print(f"Error: '{CLEAN_LIST_FILE}' not found. Please run 'validate_dataset.py' first.")
        exit()

    # Read the list of valid file paths
    with open(CLEAN_LIST_FILE, "r") as f:
        valid_base_paths = [line.strip() for line in f]
    
    total_files = len(valid_base_paths)
    print(f"Found {total_files} valid pairs to process.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_rois_found = 0

    # Loop through all the valid files
    for i, base_path in enumerate(valid_base_paths):
        template_path = base_path + "_temp.jpg"
        test_path = base_path + "_test.jpg"

        # Use the function to process the pair
        extracted_rois, sample_base_name = process_image_pair(template_path, test_path)

        if extracted_rois:
            # Save each found ROI with a unique name
            for j, roi in enumerate(extracted_rois):
                save_path = os.path.join(OUTPUT_DIR, f"{sample_base_name}_roi_{j+1}.png")
                cv2.imwrite(save_path, roi)
            total_rois_found += len(extracted_rois)
        
        # Print progress
        print(f"Processing: [{i+1}/{total_files}] - Found {len(extracted_rois)} ROIs in {sample_base_name}")

    print("\n--- Processing Complete ---")
    print(f"Processed {total_files} image pairs.")
    print(f"Extracted a total of {total_rois_found} ROIs.")
    print(f"All ROIs saved in: '{OUTPUT_DIR}'")