import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#Configuration
DATA_DIR = "data"
OUTPUT_DIR = os.path.join("outputs", "rois")

#Main Script
if __name__ == "__main__":

    # Define a sample to test the pipeline on
    sample_base_name = "00041002"
    sample_group = "group00041"
    sample_sub_group = "00041"
    
    # Construct the paths to the sample files
    base_path = os.path.join(DATA_DIR, sample_group, sample_sub_group)
    template_path = os.path.join(base_path, f"{sample_base_name}_temp.jpg")
    test_path = os.path.join(base_path, f"{sample_base_name}_test.jpg")
    
    # 1. Load images
    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if template_img is not None and test_img is not None:
        # 2. Find the difference between the images
        diff_img = cv2.absdiff(template_img, test_img)

        # 3. Apply a threshold to get a binary mask
        _, thresh_mask = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)

        # 4. Clean the mask to remove noise
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 5. Find contours of the defects
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Found {len(contours)} potential defects.")

        # 6. Crop and save the defects (ROIs)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 20: # Ignore very small contours
                x, y, w, h = cv2.boundingRect(cnt)
                roi = test_img[y:y+h, x:x+w]
                save_path = os.path.join(OUTPUT_DIR, f"{sample_base_name}_roi_{i+1}.png")
                cv2.imwrite(save_path, roi)
        
        print(f"Saved ROIs to '{OUTPUT_DIR}'")
        
        #overlay visulazation
        

        # Visualizing the steps
    
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 2, 4); plt.title("Original Test"); plt.imshow(test_img, cmap='gray')
        plt.subplot(2, 2, 1); plt.title("Difference"); plt.imshow(diff_img, cmap='gray')
        plt.subplot(2, 2, 2); plt.title("Thresholded"); plt.imshow(thresh_mask, cmap='gray')
        plt.subplot(2, 2, 3); plt.title("Cleaned Mask"); plt.imshow(cleaned_mask, cmap='gray')
        
        plt.tight_layout(); plt.show()
    else:
        print(f"Error: Could not load sample images from {base_path}")

    #end
    
    
    overlay = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 20:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,0,255), 2)
    plt.imshow(overlay)
    plt.title("Defects Highlighted")
    plt.show()
