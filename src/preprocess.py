import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# paths setup
project_root = os.path.join(os.path.dirname(__file__), '..')
data_dir = os.path.join(project_root, 'data', 'raw')
output_dir = os.path.join(project_root, 'outputs', 'rois')

if __name__ == "__main__":
    # picking a random image pair
    try:
        groups = [g for g in os.listdir(data_dir) if g.startswith("group")]
        if len(groups) == 0:
            print("No group folders found!")
            exit()

        random_group = random.choice(groups)
        group_num = random_group.replace("group", "")
        folder = os.path.join(data_dir, random_group, group_num)

        test_imgs = [f for f in os.listdir(folder) if f.endswith("_test.jpg")]
        if len(test_imgs) == 0:
            print("No test images found in", folder)
            exit()

        chosen_test = random.choice(test_imgs)
        name = chosen_test.replace("_test.jpg", "")

        # temp and test file paths
        temp_path = os.path.join(folder, name + "_temp.jpg")
        test_path = os.path.join(folder, name + "_test.jpg")

        print("Selected")

    except Exception as e:
        print("Error finding sample:", e)
        exit()

    # reading the images
    temp = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    if temp is None or test is None:
        print("Error loading images")
        exit()
        
    
    
    #### PIPE LINE
    
    
    # difference
    diff = cv2.absdiff(temp, test)

    # threshold (Otsu)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cleaning mask
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # find contours
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Found contours:", len(contours))

    os.makedirs(output_dir, exist_ok=True)
    overlay = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)

    i = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = test[y:y+h, x:x+w]
            save_path = os.path.join(output_dir, f"{name}_roi_{i}.jpeg")
            cv2.imwrite(save_path, roi)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
            i += 1

    print("Saved", i-1, "ROIs")

    # show results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Template")
    plt.imshow(temp, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title("Test")
    plt.imshow(test, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("Difference")
    plt.imshow(diff, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title("Thresholded")
    plt.imshow(mask, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title("Cleaned Mask")
    plt.imshow(mask_clean, cmap='gray')

    plt.subplot(2, 3, 6)
    plt.title("Defects")
    plt.imshow(overlay)

    plt.tight_layout()
    plt.show()
