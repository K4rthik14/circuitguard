import os
import cv2
import traceback

# --- CONFIGURATION ---
RAW_DATA_DIR = os.path.join("..", "data", "raw")
OUTPUT_DIR = os.path.join("..", "outputs", "labeled_rois_jpeg")

LABEL_MAP = {
    1: "copper",
    2: "mousebite",
    3: "open",
    4: "pin-hole",
    5: "short",
    6: "spur",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
for label in LABEL_MAP.values():
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# --- HELPER FUNCTION ---
def process_annotation_file(txt_path, img_path):
    """Reads an annotation file and extracts valid bounding boxes."""
    rois = []
    try:
        with open(txt_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"⚠️ Cannot read file: {txt_path} ({e})")
        return rois

    for line in lines:
        parts = line.strip().split()
        if len(parts) not in [5, 6]:
            print(f"⚠️ Invalid annotation format in {txt_path}: {line.strip()}")
            continue

        try:
            if len(parts) == 5:
                x1, y1, x2, y2, label_id = map(float, parts)
            else:
                _, x1, y1, x2, y2, label_id = map(float, parts)

            label_id = int(label_id)
            if label_id not in LABEL_MAP:
                print(f"⚠️ Unknown label ID {label_id} in {txt_path}")
                continue

            rois.append((int(x1), int(y1), int(x2), int(y2), LABEL_MAP[label_id]))

        except Exception:
            print(f"⚠️ Skipping invalid line in {txt_path}: {line.strip()}")
            continue

    return rois


# --- MAIN LOOP ---
total_processed = 0
total_skipped = 0
total_saved = 0

for group in os.listdir(RAW_DATA_DIR):
    group_path = os.path.join(RAW_DATA_DIR, group)
    if not os.path.isdir(group_path):
        continue

    for subdir in os.listdir(group_path):
        subdir_path = os.path.join(group_path, subdir)

        for file in os.listdir(subdir_path):
            if not file.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(subdir_path, file)
            txt_path = img_path.replace(".jpg", ".txt")

            total_processed += 1
            if not os.path.exists(txt_path):
                print(f"⚠️ Missing annotation: {txt_path}")
                total_skipped += 1
                continue

            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"⚠️ Cannot load image: {img_path}")
                    total_skipped += 1
                    continue

                rois = process_annotation_file(txt_path, img_path)
                if not rois:
                    total_skipped += 1
                    continue

                for i, (x1, y1, x2, y2, label) in enumerate(rois):
                    roi = image[y1:y2, x1:x2]
                    if roi.size == 0:
                        print(f"⚠️ Empty ROI in {txt_path} (coords: {x1},{y1},{x2},{y2})")
                        continue

                    save_path = os.path.join(OUTPUT_DIR, label, f"{os.path.splitext(file)[0]}_{i}.jpg")
                    cv2.imwrite(save_path, roi)
                    total_saved += 1

            except Exception:
                print(f"⚠️ Error processing {img_path}")
                traceback.print_exc()
                total_skipped += 1
                continue

print("\n--- Processing Complete ---")
print(f"✅ Successfully saved {total_saved} labeled ROIs.")
print(f"⚠️ Skipped {total_skipped} entries due to errors or missing files.")
print(f"📂 Output directory: '{os.path.abspath(OUTPUT_DIR)}'")
print(f"📊 Total processed entries: {total_processed}")