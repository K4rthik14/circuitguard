import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# setting up folder paths
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
data_dir = os.path.join(project_root, 'data', 'raw')
map_file = os.path.join(project_root, 'data', 'test.txt')
clean_file_list = os.path.join(project_root, 'clean_file_list.txt') # Path to your clean list

img_size = (640, 640)

# Map defect IDs to readable names
DEFECT_MAP = {
    1: 'open', 
    2: 'short',
    3: 'mousebite',
    4: 'spur',
    5: 'copper',
    6: 'pin-hole'
}


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Could not load image:", path)
        return None
    img = cv2.resize(img, img_size)
    return img

def read_boxes(txt_path):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    
    with open(txt_path, "r") as f:
        for line in f:
            # This handles both comma and space separators
            parts = line.strip().replace(',', ' ').split()
            if len(parts) == 5:
                x1, y1, x2, y2, _ = map(int, parts)
                w = x2 - x1
                h = y2 - y1
                boxes.append((x1, y1, w, h))
    return boxes



    # Generate Defect Distribution Graph
    if not os.path.exists(clean_file_list):
        print(f"Error: '{clean_file_list}' not found. Please run the validation script first.")
    else:
        with open(clean_file_list, "r") as f:
            valid_base_paths = [line.strip() for line in f]
        
        defect_counts = {name: 0 for name in DEFECT_MAP.values()}

        # Loop through each valid file to find its annotation and count defects
        for base_path in valid_base_paths:
            img_name_no_dir = os.path.basename(base_path)
            
            with open(map_file, "r") as map_f:
                for line in map_f:
                    if img_name_no_dir in line:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            annotation_rel_path = parts[1]
                            txt_path = os.path.join(data_dir, annotation_rel_path)
                            
                            with open(txt_path, "r") as ann_f:
                                for ann_line in ann_f:
                                    ann_parts = ann_line.strip().split(',')
                                    if len(ann_parts) == 5:
                                        defect_id = int(ann_parts[4])
                                        defect_name = DEFECT_MAP.get(defect_id)
                                        if defect_name:
                                            defect_counts[defect_name] += 1
                            break

        # Create and show the bar chart
        names = list(defect_counts.keys())
        counts = list(defect_counts.values())

        plt.figure(figsize=(10, 6))
        plt.bar(names, counts, color='skyblue')
        plt.title('Distribution of Defect Types in Dataset')
        plt.xlabel('Defect Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    print("\nEDA Done.")