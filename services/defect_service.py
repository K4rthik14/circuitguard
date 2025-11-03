# services/defect_service.py
import torch
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import os # Added for path joining
from typing import List, Tuple, Dict

# --- Configuration ---
# Get project root assuming this file is in circuitguard/services/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_model.pth")
CLASSES = ['copper', 'mousebite', 'open', 'pin-hole', 'short', 'spur']
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_CONTOUR_AREA_DEFAULT = 5

# --- Model Loading (Cached using a simple dictionary) ---
_model_cache = {}
def load_classification_model(model_path: str, num_classes: int):
    """Loads the EfficientNet model."""
    if model_path in _model_cache:
        return _model_cache[model_path]

    print(f"Loading model from: {model_path}") # Log loading
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Model file not found at {model_path}")

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
    # Load state dict robustly
    try:
        # Added weights_only=True for security, might need adjustment if your .pth needs pickle
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
        # Handle potential keys mismatch (e.g., if saved with DataParallel or older torch versions)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Remove `module.` prefix if present (from DataParallel saving)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
    except Exception as e:
        # Fallback if weights_only=True fails (less secure)
        try:
             print("Warning: Loading model with weights_only=True failed. Attempting weights_only=False (less secure).")
             state = torch.load(model_path, map_location=DEVICE, weights_only=False)
             if isinstance(state, dict) and "state_dict" in state:
                 state = state["state_dict"]
             state = {k.replace("module.", ""): v for k, v in state.items()}
             model.load_state_dict(state)
        except Exception as inner_e:
             raise RuntimeError(f"Error loading model state_dict from {model_path}: {inner_e}")

    model.to(DEVICE)
    model.eval()
    _model_cache[model_path] = model # Cache the loaded model
    return model

# --- Transforms ---
roi_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Classification ---
def classify_roi(roi_pil: Image.Image, model) -> str:
    """Runs inference on a single ROI (PIL image)."""
    roi_rgb = roi_pil.convert("RGB") # Ensure 3 channels
    transformed = roi_transform(roi_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(transformed)
        pred_index = int(out.argmax(1).item())
    # Basic check for valid index
    if 0 <= pred_index < len(CLASSES):
        return CLASSES[pred_index]
    else:
        print(f"Warning: Predicted index {pred_index} out of range for classes.")
        return "unknown" # Fallback label

# --- Image Processing ---
def find_defects(
    template_img_pil: Image.Image,
    test_img_pil: Image.Image,
    diff_threshold: int = 0,
    morph_iterations: int = 2,
    min_area: int = MIN_CONTOUR_AREA_DEFAULT
):

    """
    Performs subtraction, thresholding, and contour extraction.
    Returns ROIs, boxes, processed BGR image, and contour areas.
    """
    template_cv = np.array(template_img_pil.convert('L'))
    test_cv = np.array(test_img_pil.convert('L'))
    h, w = template_cv.shape
    test_cv = cv2.resize(test_cv, (w, h))

    # --- Image Subtraction ---
    diff = cv2.absdiff(template_cv, test_cv)

    # --- Thresholding ---
    if diff_threshold > 0:
        _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois, boxes, areas = [], [], []
    test_img_rgb_pil = test_img_pil.convert('RGB')

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            roi_pil = test_img_rgb_pil.crop((x, y, x + w_box, y + h_box))
            rois.append(roi_pil)
            boxes.append((x, y, w_box, h_box))
            areas.append(area)

    output_image_cv_bgr = cv2.cvtColor(np.array(test_img_rgb_pil), cv2.COLOR_RGB2BGR)
    return rois, boxes, output_image_cv_bgr, areas

# services/defect_service.py

# --- Drawing ---
def draw_annotations(image_cv_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: List[str]) -> np.ndarray:
    """Draws bounding boxes and labels on the image (expects BGR)."""
    out = image_cv_bgr.copy()
    img_height, img_width = out.shape[:2] # Get image dimensions for boundary checks
    padding = 5 # Pixels to add around the original box - ADJUST AS NEEDED

    for (x,y,w,h), label in zip(boxes, labels):

        # --- Calculate padded coordinates ---
        x1_pad = max(0, x - padding) # New top-left x (don't go below 0)
        y1_pad = max(0, y - padding) # New top-left y (don't go below 0)
        x2_pad = min(img_width, x + w + padding) # New bottom-right x (don't exceed width)
        y2_pad = min(img_height, y + h + padding) # New bottom-right y (don't exceed height)
        # --- End calculation ---

        # Draw the LARGER rectangle using padded coordinates
        cv2.rectangle(out, (x1_pad, y1_pad), (x2_pad, y2_pad), (255,0,0), 2) # Blue box, thickness 2

        # --- Label drawing logic (adjust position relative to padded box) ---
        text_y = max(12, y1_pad - 6) # Position label above the padded box
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Ensure background box doesn't go off-screen top
        bg_y1 = max(0, text_y - text_height - baseline)
        bg_y2 = text_y + baseline
        # Draw background and text relative to original x or padded x1_pad
        cv2.rectangle(out, (x1_pad, bg_y1), (x1_pad + text_width, bg_y2), (255, 255, 255), cv2.FILLED)
        cv2.putText(out, label, (x1_pad, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
        # --- End label drawing ---

    return out

# --- Main Service Function (Module 6 Logic) ---
def process_and_classify_defects(
    template_pil: Image.Image,
    test_pil: Image.Image,
    diff_threshold: int = 0,
    morph_iterations: int = 2,
    min_area: int = MIN_CONTOUR_AREA_DEFAULT
):

    """Main service: finds, classifies, annotates, and measures defect areas."""
    model = load_classification_model(MODEL_PATH, len(CLASSES))

    rois, boxes, output_image_cv_bgr, areas = find_defects(
        template_pil, test_pil,
        diff_threshold=diff_threshold,
        morph_iterations=morph_iterations,
        min_area=min_area
    )

    defect_details = []
    if not rois:
        return cv2.cvtColor(np.array(test_pil.convert('RGB')), cv2.COLOR_RGB2BGR), defect_details

    labels = [classify_roi(roi, model) for roi in rois]
    annotated_cv_bgr = draw_annotations(output_image_cv_bgr, boxes, labels)

    for idx, (label, (x, y, w, h), area) in enumerate(zip(labels, boxes, areas)):
        defect_details.append({
            "id": idx + 1,
            "label": label,
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": round(float(area), 2)
        })

    return annotated_cv_bgr, defect_details
