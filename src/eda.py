import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# setting up folder paths
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
data_dir = os.path.join(project_root, 'data', 'raw')
map_file = os.path.join(project_root, 'data', 'test.txt')

img_size = (640, 640)

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Couldn't load image")
        return None
    img = cv2.resize(img, img_size)
    return img
#box read
def read_boxes(txt_path):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            if len(parts) == 5:
                x1, y1, x2, y2, _ = map(int, parts)
                w = x2 - x1
                h = y2 - y1
                boxes.append((x1, y1, w, h))
    return boxes


if __name__ == "__main__":


    if not os.path.exists(map_file):
        print("Map file not found:")
        exit()

    # read all image pairs
    with open(map_file, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        print("Map file is empty.")
        exit()

    # pick one random pair
    random_line = random.choice(lines).strip().split()
    if len(random_line) != 2:
        exit()

    img_path, txt_rel_path = random_line
    base = img_path.replace(".jpg", "")

    test_path = os.path.join(data_dir, base + "_test.jpg")
    temp_path = os.path.join(data_dir, base + "_temp.jpg")
    txt_path = os.path.join(data_dir, txt_rel_path)

    print("Sample selected:", os.path.basename(test_path))

    # load images
    temp = load_img(temp_path)
    test = load_img(test_path)
    boxes = read_boxes(txt_path)

    if temp is None or test is None:
        print("Image loading failed.")
        exit()

    # template img
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(temp, cmap='gray')
    plt.title("Template")
      # test img
    plt.subplot(1, 2, 2)
    plt.imshow(test, cmap='gray')
    plt.title("Test")
    plt.show()

    # draw boxes
    if len(boxes) > 0:
        test_col = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in boxes:
            cv2.rectangle(test_col, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.imshow(test_col)
        plt.title("Detected Regions")
        plt.show()
    else:
        print("No boxes found in txt file.")

#END
