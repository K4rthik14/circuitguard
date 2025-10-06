# preprocessing.py
import cv2
import numpy as np

def process_pcb_pair(template_path, test_path):
    """
    Performs the full image processing pipeline for one pair of PCB images.
    """
    # 1. Load images in grayscale
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if template is None or test is None:
        raise FileNotFoundError("Could not read one or both image files.")

    # The dataset is pre-aligned, so we can proceed with subtraction.
    # In a real-world case, you would add an alignment step here.

    # 2. Image Subtraction [cite: 39]
    diff_image = cv2.absdiff(template, test)

    # 3. Otsu's Thresholding to create a binary mask of defects [cite: 40]
    _, threshold_image = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Morphological Operations to clean up noise [cite: 29]
    # Use a kernel for erosion and dilation
    kernel = np.ones((3, 3), np.uint8)
    # Opening: Erosion followed by Dilation (removes small noise)
    cleaned_mask = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # 5. Contour Detection to find defect boundaries [cite: 30, 50]
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "template": template,
        "test": test,
        "difference": diff_image,
        "threshold": threshold_image,
        "cleaned_mask": cleaned_mask,
        "contours": contours
    }

def extract_rois(original_test_image, contours, min_area=20):
    """
    Extracts Regions of Interest (ROIs) from the test image based on contours.
    Returns the image with bounding boxes drawn and a list of cropped defect images.
    [cite: 51]
    """
    output_image = cv2.cvtColor(original_test_image, cv2.COLOR_GRAY2BGR) # Convert to color to draw boxes
    cropped_defects = []

    for contour in contours:
        # Only process contours larger than a minimum area to filter out noise
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw a green bounding box on the output image
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Crop the defect from the original test image [cite: 51]
            roi = original_test_image[y:y+h, x:x+w]
            cropped_defects.append(roi)

    return output_image, cropped_defects