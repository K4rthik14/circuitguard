import os
import cv2

# --- Config ---
DATA_DIR = "data"
CLEAN_LIST_FILE = "clean_file_list.txt"
# --- NEW: Set to True to print detailed checks for the first 5 samples ---
DEBUG_MODE = True

def validate_dataset():
    print(f"--- Starting Validation of Dataset in '{DATA_DIR}' ---")
    
    all_test_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith("_test.jpg"):
                all_test_files.append(os.path.join(root, file))

    if not all_test_files:
        print("Validation Error: No test images ('_test.jpg') found.")
        return

    clean_file_basenames = []
    errors_found = 0
    total_pairs_count = len(all_test_files)

    print(f"Found {total_pairs_count} test images. Checking for complete pairs...")

    for i, test_path in enumerate(all_test_files):
        base_path = test_path.replace("_test.jpg", "")
        temp_path = base_path + "_temp.jpg"
        txt_path = base_path + ".txt"

        # Check for companions
        template_exists = os.path.exists(temp_path)
        annotation_exists = os.path.exists(txt_path)

        # If debug mode is on, print details for the first few files
        if DEBUG_MODE and i < 5:
            print(f"\n--- Debugging Sample {i+1} ---")
            print(f"Checking Test Image: {test_path}")
            print(f"  -> Looking for Template: {temp_path} ... Found: {template_exists}")
            print(f"  -> Looking for Annotation: {txt_path} ... Found: {annotation_exists}")

        if template_exists and annotation_exists:
            clean_file_basenames.append(base_path)
        else:
            errors_found += 1

    # --- Print Final Report ---
    print("\n--- Validation Report ---")
    print(f"Total pairs checked: {total_pairs_count}")
    print(f"✅ Valid pairs found: {len(clean_file_basenames)}")
    print(f"❌ Incomplete pairs found: {errors_found}")
    
    if clean_file_basenames:
        with open(CLEAN_LIST_FILE, "w") as f:
            for base_path in clean_file_basenames:
                f.write(base_path + "\n")
        print(f"\n🎉 Successfully created a clean list at '{CLEAN_LIST_FILE}'")

    print("\n--- End of Report ---")

if __name__ == "__main__":
    validate_dataset()
    