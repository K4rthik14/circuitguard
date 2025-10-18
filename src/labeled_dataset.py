import os
import cv2
from tqdm import tqdm

# --- CONFIGURATION ---
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RAW_DATA_DIR = os.path.join(project_root, "data", "raw")
MAP_FILE = os.path.join(project_root, "data", "test.txt")
OUTPUT_DIR = os.path.join(project_root, "outputs", "labeled_rois_jpeg")

LABEL_MAP = {
    1: "copper",
    2: "mousebite",
    3: "open",
    4: "pin-hole",
    5: "short",
    6: "spur",
}

# --- SETUP OUTPUT DIRECTORIES ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
for label_name in LABEL_MAP.values():
    os.makedirs(os.path.join(OUTPUT_DIR, label_name), exist_ok=True)


# --- FUNCTION: Process Annotation File ---
def process_annotation_file(txt_path):
    """Reads bounding boxes and labels from an annotation text file."""
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
            print(f"⚠️ Invalid format in {txt_path}: {line.strip()}")
            continue
        try:
            # Format can be: x1 y1 x2 y2 label_id OR img_id x1 y1 x2 y2 label_id
            coords = list(map(float, parts[-5:]))  # last 5 values are coords + label
            x1, y1, x2, y2, label_id = coords
            label_id = int(label_id)

            if label_id not in LABEL_MAP:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if x2 <= x1 or y2 <= y1:
                continue

            rois.append((x1, y1, x2, y2, LABEL_MAP[label_id]))
        except Exception:
            continue

    return rois


# --- MAIN EXECUTION ---
if not os.path.exists(MAP_FILE):
    print(f"❌ Error: test.txt not found at {MAP_FILE}")
    exit(1)

with open(MAP_FILE, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

total_processed = 0
total_saved = 0
total_skipped = 0

print("🔍 Processing images...\n")

for line in tqdm(lines, desc="Extracting ROIs", unit="file"):
    parts = line.split()
    if len(parts) != 2:
        total_skipped += 1
        continue

    img_rel, txt_rel = parts
    img_path = os.path.join(RAW_DATA_DIR, img_rel)
    txt_path = os.path.join(RAW_DATA_DIR, txt_rel)

    # Try to find image if missing
    if not os.path.exists(img_path):
        img_base, img_ext = os.path.splitext(img_path)
        found = False
        for suffix in ["_temp", "_test"]:
            alt_path = f"{img_base}{suffix}{img_ext}"
            if os.path.exists(alt_path):
                img_path = alt_path
                found = True
                break
        if not found:
            print(f"⚠️ Missing image: {img_path}")
            total_skipped += 1
            continue

    if not os.path.exists(txt_path):
        print(f"⚠️ Missing annotation: {txt_path}")
        total_skipped += 1
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Cannot load image: {img_path}")
        total_skipped += 1
        continue

    rois = process_annotation_file(txt_path)
    if not rois:
        total_skipped += 1
        continue

    h, w = image.shape[:2]
    base_name, ext = os.path.splitext(os.path.basename(img_path))

    for i, (x1, y1, x2, y2, label) in enumerate(rois):
        # Clip coordinates to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        save_path = os.path.join(OUTPUT_DIR, label, f"{base_name}_{i}{ext}")
        cv2.imwrite(save_path, roi)
        total_saved += 1

    total_processed += 1

# --- SUMMARY ---
print("\n--- ✅ Processing Complete ---")
print(f"📸 Total images processed : {total_processed}")
print(f"🧩 Total ROIs saved       : {total_saved}")
print(f"⚠️ Total skipped          : {total_skipped}")
print(f"📂 Output directory       : {OUTPUT_DIR}")
