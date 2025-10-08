import os
import cv2

# --- Config ---
DATA_DIR = "data"
# The script will save the list of good files here
CLEAN_LIST_FILE = "clean_file_list.txt"

def validate_and_clean_dataset():
    """
    Scans the dataset for issues and generates a clean list of valid file paths.
    """
    print(f"--- Starting Validation of Dataset in '{DATA_DIR}' ---")
    
    all_test_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith("_test.jpg"):
                all_test_files.append(os.path.join(root, file))

    if not all_test_files:
        print("Validation Error: No test images found.")
        return

    # Lists for problems and for the good files
    error_log = []
    clean_file_basenames = []
    total_pairs_count = len(all_test_files)

    print(f"Found {total_pairs_count} test images. Checking each pair...")

    for test_path in all_test_files:
        base_path = test_path.replace("_test.jpg", "")
        temp_path = base_path + "_temp.jpg"
        txt_path = base_path + ".txt"

        # Check for missing files
        if not os.path.exists(temp_path):
            error_log.append(f"[Missing Template] {test_path}")
            continue
        if not os.path.exists(txt_path):
            error_log.append(f"[Missing Annotation] {test_path}")
            continue

        # Check for corrupted images
        test_img = cv2.imread(test_path)
        if test_img is None:
            error_log.append(f"[Corrupted Image] {test_path}")
            continue

        # If all checks pass, add the base path to our clean list
        clean_file_basenames.append(base_path)

    # --- Print Final Report ---
    print("\n--- Validation Report ---")
    print(f"Total pairs checked: {total_pairs_count}")
    print(f"✅ Valid pairs found: {len(clean_file_basenames)}")
    print(f"❌ Problematic pairs found: {len(error_log)}")

    if error_log:
        print("\n-- Error Details --")
        for error in error_log[:20]: # Print first 20 errors to avoid flooding the screen
            print