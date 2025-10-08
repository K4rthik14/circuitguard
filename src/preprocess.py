import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
# This robustly finds the project root relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(script_dir, "..")
# Note: This points to your "data" folder
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "rois")

def subtract_images(template_path, test_path):
    """Loads a template and test image and returns their absolute difference."""
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if template_img is None or test_img is None:
        raise FileNotFoundError("Could not read one or both images.")

    # Calculate the absolute difference between the two images
    diff_img = cv2.absdiff(template_img, test_img)
    return diff_img, test_img # Return diff and the original test image for later use

def apply_threshold(diff_img):
    """Applies Otsu's thresholding to a difference image to create a binary mask."""
    # Otsu's method automatically determines the best threshold value
    _, threshold_mask = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold_mask

def clean_mask(mask):
    """Cleans a binary mask using morphological opening to remove noise."""
    # A 3x3 kernel is standard for this kind of cleaning
    kernel = np.ones((3, 3), np.uint8)
    
    # MORPH_OPEN is an erosion followed by a dilation, which removes small specks
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return cleaned_mask

def extract_rois(cleaned_mask, original_test_img, min_area=20):
    """Finds contours in the mask and extracts them as ROIs from the original image."""
    rois = []
    # Find the contours of all distinct defect areas
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out very small contours that are likely still noise
        if cv2.contourArea(contour) > min_area:
            # Get the bounding box (x, y, width, height) of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Use the bounding box coordinates to crop the defect from the test image
            roi = original_test_img[y:y+h, x:x+w]
            rois.append(roi)
            
    return rois

if __name__ == "__main__":
    # --- SETUP FOR A SINGLE TEST RUN ---
    # We will test on a known sample from group00041
    sample_base_name = "00041002"
    sample_group = "group00041"
    sample_sub_group = "00041"

    # Construct the full paths
    base_path = os.path.join(DATA_DIR, sample_group, sample_sub_group)
    template_path = os.path.join(base_path, sample_base_name + "_temp.jpg")
    test_path = os.path.join(base_path, sample_base_name + "_test.jpg")

    print(f"🚀 Processing sample: {sample_base_name}")

    try:
        # --- RUN THE PIPELINE ---
        # 1. Subtract images
        difference_image, test_image = subtract_images(template_path, test_path)
        # 2. Apply threshold
        thresholded_mask = apply_threshold(difference_image)
        # 3. Clean the mask
        final_mask = clean_mask(thresholded_mask)
        # 4. Extract ROIs
        extracted_rois = extract_rois(final_mask, test_image)

        print(f"✅ Pipeline complete. Found {len(extracted_rois)} ROIs.")

        # --- SAVE THE RESULTS ---
        if extracted_rois:
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            
            for i, roi in enumerate(extracted_rois):
                save_path = os.path.join(OUTPUT_DIR, f"{sample_base_name}_roi_{i+1}.png")
                cv2.imwrite(save_path, roi)
            print(f"💾 Saved ROIs to: {OUTPUT_DIR}")

        # --- VISUALIZE THE STEPS ---
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 2, 1); plt.title("1. Difference"); plt.imshow(difference_image, cmap='gray')
        plt.subplot(2, 2, 2); plt.title("2. Thresholded"); plt.imshow(thresholded_mask, cmap='gray')
        plt.subplot(2, 2, 3); plt.title("3. Cleaned Mask"); plt.imshow(final_mask, cmap='gray')
        plt.subplot(2, 2, 4); plt.title("4. Original Test"); plt.imshow(test_image, cmap='gray')
        plt.tight_layout(); plt.show()
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find sample files. Please check the path.")
        print(e)