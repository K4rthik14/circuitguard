import cv2
import os

group_path = "../PCBData/group00041"

# Subfolders
template_folder = os.path.join(group_path, "00041")
test_folder = os.path.join(group_path, "00041_not")

print("Template folder:", os.listdir(template_folder)[:5])
print("Test folder:", os.listdir(test_folder)[:5])

# Example: load first template + test image
template = cv2.imread(os.path.join(template_folder, "1.jpg"), 0)
test = cv2.imread(os.path.join(test_folder, "1.jpg"), 0)

print("Template loaded:", template is not None)
print("Test loaded:", test is not None)
