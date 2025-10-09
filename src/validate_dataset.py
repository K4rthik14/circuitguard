import os
import cv2

# folders and files
data_dir = "data/raw"
map_file = "data/test.txt"
clean_file_list = "clean_file_list.txt"

if __name__ == "__main__":
    print("Checking map file:", map_file)

    if not os.path.exists(map_file):
        print("Map file not found!")
        exit()

    with open(map_file, "r") as f:
        lines = f.readlines()

    print("Total entries in map file:", len(lines))

    clean_files = []
    errors = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 2:
            img_path, txt_path = parts
            base = img_path.replace(".jpg", "")

            test_path = os.path.join(data_dir, base + "_test.jpg")
            temp_path = os.path.join(data_dir, base + "_temp.jpg")
            ann_path = os.path.join(data_dir, txt_path)

            if os.path.exists(test_path) and os.path.exists(temp_path) and os.path.exists(ann_path):
                clean_files.append(os.path.join(data_dir, base))
            else:
                print("Missing file for:", base)
                errors += 1

    print("\n--- Report ---")
    print("Total checked:", len(lines))
    print("Valid pairs:", len(clean_files))
    print("Missing ones:", errors)

    if len(clean_files) > 0:
        with open(clean_file_list, "w") as f:
            for item in clean_files:
                f.write(item + "\n")
        print("\nClean file list saved to:", clean_file_list)
    else:
        print("\nNo valid pairs found.")

    print("\nDone!")
