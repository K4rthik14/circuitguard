import os

import cv2

import traceback



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



# Create output dirs

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in LABEL_MAP.values():

    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)



# --- HELPER FUNCTION ---

def process_annotation_file(txt_path, image):

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

                continue



            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if x2 <= x1 or y2 <= y1:

                continue

            rois.append((x1, y1, x2, y2, LABEL_MAP[label_id]))

        except Exception:

            continue

    return rois



# --- MAIN LOOP ---

if not os.path.exists(MAP_FILE):

    print(f"❌ Error: test.txt not found at {MAP_FILE}")

    exit()



with open(MAP_FILE, "r") as f:

    lines = f.readlines()



total_processed = 0

total_saved = 0

total_skipped = 0



for line in lines:

    parts = line.strip().split()

    if len(parts) != 2:

        continue



    img_rel, txt_rel = parts

    img_path = os.path.join(RAW_DATA_DIR, img_rel)

    txt_path = os.path.join(RAW_DATA_DIR, txt_rel)



    if not os.path.exists(img_path):



    # Try _temp or _test variants

        img_base, img_ext = os.path.splitext(img_path)

        for suffix in ["_temp", "_test"]:

            alt_path = f"{img_base}{suffix}{img_ext}"

            if os.path.exists(alt_path):

                img_path = alt_path

            break

    else:

        print(f"⚠️ Missing image: {img_path}")

        total_skipped += 1

        continue



    if not os.path.exists(txt_path):

        print(f"⚠️ Missing annotation: {txt_path}")

        total_skipped += 1

        continue



    image = cv2.imread(img_path)

    if image is None:

        total_skipped += 1

        continue



    rois = process_annotation_file(txt_path, image)

    if not rois:

        total_skipped += 1

        continue



    for i, (x1, y1, x2, y2, label) in enumerate(rois):

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:

            continue

        save_path = os.path.join(OUTPUT_DIR, label, f"{os.path.basename(img_path).replace('.jpg', f'_{i}.jpg')}")

        cv2.imwrite(save_path, roi)

        total_saved += 1



    total_processed += 1



print("\n--- Processing Complete ---")

print(f"✅ Total ROIs saved: {total_saved}")

print(f"⚠️ Total skipped: {total_skipped}")

print(f"📂 Output directory: {OUTPUT_DIR}")