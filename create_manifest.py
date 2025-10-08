import os
import pandas as pd

# --- Config ---
# The script will read this file to find the valid pairs.
CLEAN_LIST_FILE = "clean_file_list.txt"
OUTPUT_CSV = "manifest.csv"

def generate_manifest_from_clean_list():
    """
    Reads the list of clean file paths and generates a manifest.csv.
    """
    print(f"Reading clean file list from '{CLEAN_LIST_FILE}'...")

    # First, check if the clean list exists.
    if not os.path.exists(CLEAN_LIST_FILE):
        print(f"Error: '{CLEAN_LIST_FILE}' not found.")
        print("Please run 'validate_dataset.py' first to generate it.")
        return

    pairs = []
    # Read the base path of each valid pair from the file
    with open(CLEAN_LIST_FILE, "r") as f:
        for base_path in f:
            base_path = base_path.strip() # Remove any extra whitespace/newlines
            if base_path:
                # Reconstruct the full paths for all three files
                pairs.append({
                    "template_path": base_path + "_temp.jpg",
                    "test_path": base_path + "_test.jpg",
                    "annotation_path": base_path + ".txt"
                })

    if not pairs:
        print("Error: No paths were found in the clean file list.")
        return

    # Convert the list of pairs to a pandas DataFrame
    manifest_df = pd.DataFrame(pairs)
    
    # Save the DataFrame to a CSV file
    manifest_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✅ Successfully created '{OUTPUT_CSV}' with {len(manifest_df)} pairs.")


if __name__ == "__main":
    # You may need to install pandas: pip install pandas
    generate_manifest_from_clean_list()