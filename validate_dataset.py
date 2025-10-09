import os
import cv2

# --- Config ---
DATA_DIR = "data/raw"
MAP_FILE = "data/test.txt" # The map file you provided
CLEAN_LIST_FILE = "clean_file_list.txt"

def validate_dataset_with_map():
    """
    Uses the provided map file (test.txt) to find and validate
    all complete image pairs in the dataset.
    """
    print(f"--- Starting Validation using '{MAP_FILE}' ---")

    if not os.path.exists(MAP_FILE):
        print(f"Error: Map file '{MAP_FILE}' not found. Please make sure it's in the project folder.")
        return

    # Lists to keep track of files
    error_log = []
    clean_file_basenames = []
    
    # Read the map file to get the paths
    with open(MAP_FILE, "r") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} entries in the map file. Now validating each one...")

    # Loop through each entry in the map file
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue # Skip malformed lines

        image_path_from_map, annotation_path_from_map = parts
        
        # The map gives us the annotation path directly.
        # We assume the image path in the map corresponds to the _test.jpg
        # and we derive the _temp.jpg from it.
        base_name_path = image_path_from_map.replace(".jpg", "")
        
        # Construct the full, real paths to the three files
        test_path = os.path.join(DATA_DIR, base_name_path + "_test.jpg")
        temp_path = os.path.join(DATA_DIR, base_name_path + "_temp.jpg")
        annotation_path = os.path.join(DATA_DIR, annotation_path_from_map)

        # Check if all three files actually exist at those locations
        if os.path.exists(test_path) and os.path.exists(temp_path) and os.path.exists(annotation_path):
            # If they all exist, it's a valid pair. We save the base path.
            clean_file_basenames.append(os.path.join(DATA_DIR, base_name_path))
        else:
            error_log.append(f"Incomplete pair for entry: {line.strip()}")
    
    # --- Print Final Report ---
    print("\n--- Validation Report ---")
    print(f"Total entries checked in map file: {len(lines)}")
    print(f"✅ Valid pairs found: {len(clean_file_basenames)}")
    print(f"❌ Incomplete pairs found: {len(error_log)}")

    # --- Save the Clean List ---
    if clean_file_basenames:
        with open(CLEAN_LIST_FILE, "w") as f:
            for base_path in clean_file_basenames:
                f.write(base_path + "\n")
        print(f"\n🎉 Successfully created a clean list at '{CLEAN_LIST_FILE}'")

    print("\n--- End of Report ---")


if __name__ == "__main__":
    validate_dataset_with_map()