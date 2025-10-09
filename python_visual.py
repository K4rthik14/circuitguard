import os
import cv2

# --- Config ---
CLEAN_LIST_FILE = "clean_file_list.txt"
WINDOW_NAME = "Dataset Visual Validator"

def read_annotations_for_validation(txt_path):
    """Reads the (x1, y1, x2, y2, type) annotations from a text file."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
        
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 5:
                x1, y1, x2, y2, _ = map(int, parts)
                boxes.append((x1, y1, x2, y2))
    return boxes

if __name__ == "__main__":
    # Check if the clean list exists
    if not os.path.exists(CLEAN_LIST_FILE):
        print(f"Error: '{CLEAN_LIST_FILE}' not found. Please run 'validate_dataset.py' first.")
        exit()

    # Read the list of all valid file paths
    with open(CLEAN_LIST_FILE, "r") as f:
        valid_base_paths = [line.strip() for line in f]
    
    total_images = len(valid_base_paths)
    if total_images == 0:
        print("The clean file list is empty. Nothing to validate.")
        exit()

    print("--- Starting Interactive Visual Validator ---")
    print("Controls:")
    print("  -> Right Arrow or Spacebar = Next Image")
    print("  <- Left Arrow = Previous Image")
    print("  'q' or ESC = Quit")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    current_index = 0

    while True:
        # Get the paths for the current image
        base_path = valid_base_paths[current_index]
        test_path = base_path + "_test.jpg"
        txt_path = base_path.replace("_test.jpg", ".txt") # Reconstruct txt path

        # Load the original, full-size image
        image = cv2.imread(test_path)
        if image is None:
            print(f"Warning: Could not load image {test_path}")
            image = np.zeros((640, 640, 3), dtype=np.uint8) # Show a black screen on error

        # Read annotations and draw them
        boxes = read_annotations_for_validation(txt_path)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw a green box

        # Add text to the image to show progress
        info_text = f"Image {current_index + 1} / {total_images}"
        file_text = os.path.basename(test_path)
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, file_text, (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the image
        cv2.imshow(WINDOW_NAME, image)
        
        # Wait for a key press
        key = cv2.waitKey(0)

        # --- Handle Controls ---
        # Next image (right arrow or spacebar)
        if key == 83 or key == 32: 
            current_index = min(current_index + 1, total_images - 1)
        # Previous image (left arrow)
        elif key == 81:
            current_index = max(current_index - 1, 0)
        # Quit ('q' or ESC)
        elif key == ord('q') or key == 27:
            break
            
    cv2.destroyAllWindows()
    print("--- Validator Closed ---")