import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "data"
OUTPUT_DIR = os.path.join("outputs", "rois")

if __name__ == "__main__":

    # Sample details (change as needed)
    sample_base_name = "00041002"
    sample_group = "group00041"
    sample_sub_group = "00041"

    base_path = os.path.join(DATA_DIR, sample_group, sample_sub_group)
    template_path = os.path.join(base_path, f"{sample_base_name}_temp.jpg")
    test_path = os.path.join(base_path, f"{sample_base_name}_test.jpg")

    # 1️⃣ Load images
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if template is None or test is None:
        print("❌ Error: Could not load images.")
        exit()

    # 2️⃣ Resize (optional, ensures same dimensions)
    test = cv2.resize(test, (template.shape[1], template.shape[0]))

    # 3️⃣ Preprocess: normalize lighting & reduce noise
    template = cv2.GaussianBlur(template, (5, 5), 0)
    test = cv2.GaussianBlur(test, (5, 5), 0)

    template = cv2.equalizeHist(template)
    test = cv2.equalizeHist(test)

    # 4️⃣ Image subtraction
    diff = cv2.absdiff(template, test)

    # 5️⃣ Threshold using Otsu (auto finds best cutoff)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6️⃣ Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 7️⃣ Contour detection
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    roi_count = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if 50 < area < 2000:  # ignore too small or too large areas
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi = test[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{sample_base_name}_roi_{i+1}.png"), roi)
            roi_count += 1

    print(f"✅ Found {roi_count} valid defects. Saved ROIs to '{OUTPUT_DIR}'")

    # 8️⃣ Visualization
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1); plt.title("Template"); plt.imshow(template, cmap='gray')
    plt.subplot(2, 2, 2); plt.title("Test"); plt.imshow(test, cmap='gray')
    plt.subplot(2, 2, 3); plt.title("Difference"); plt.imshow(diff, cmap='gray')
    plt.subplot(2, 2, 4); plt.title("Detected Defects"); plt.imshow(overlay)
    plt.tight_layout(); plt.show()
