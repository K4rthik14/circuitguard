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
MIN_CONTOUR_AREA_DEFAULT = 20

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
        state = torch.load(model_path, map_location=DEVICE)
        # Handle potential keys mismatch (e.g., if saved with DataParallel or older torch versions)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Remove `module.` prefix if present (from DataParallel saving)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(f"Error loading model state_dict from {model_path}: {e}")

    model.to(DEVICE)
    model.eval()
    _model_cache[model_path] = model # Cache the loaded model
    return model

# --- Transforms ---
roi_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #
])

# --- Classification ---
def classify_roi(roi_pil: Image.Image, model) -> str:
    """Runs inference on a single ROI (PIL image)."""
    roi_rgb = roi_pil.convert("RGB") # Ensure 3 channels
    transformed = roi_transform(roi_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(transformed)
        pred_index = int(out.argmax(1).item())
    return CLASSES[pred_index]

# --- Image Processing ---
def find_defects(template_img_pil: Image.Image, test_img_pil: Image.Image, min_area:int=MIN_CONTOUR_AREA_DEFAULT) -> Tuple[List[Image.Image], List[Tuple], np.ndarray]:
    """Performs subtraction, thresholding, and contour extraction."""
    template_cv = np.array(template_img_pil.convert('L')) # Grayscale
    test_cv = np.array(test_img_pil.convert('L'))     # Grayscale
    h, w = template_cv.shape
    test_cv = cv2.resize(test_cv, (w, h))             # Ensure same size

    # Image Subtraction & Thresholding
    diff = cv2.absdiff(template_cv, test_cv)          #
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #
    kernel = np.ones((3,3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2) # Erosion -> Dilation
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1) # Dilation -> Erosion

    # Contour Extraction
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #

    rois, boxes = [], []
    # Use original color test image for cropping ROIs
    test_img_rgb_pil = test_img_pil.convert('RGB')

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x,y,w_box,h_box = cv2.boundingRect(cnt)
            # Crop ROI from the original PIL test image
            # Add padding check to avoid errors on edges
            if x + w_box <= test_img_rgb_pil.width and y + h_box <= test_img_rgb_pil.height:
                roi_pil = test_img_rgb_pil.crop((x, y, x + w_box, y + h_box))
                # Ensure ROI is not empty
                if roi_pil.width > 0 and roi_pil.height > 0:
                    rois.append(roi_pil)
                    boxes.append((x,y,w_box,h_box))
            else:
                print(f"Warning: Bounding box ({x},{y},{w_box},{h_box}) exceeds image dimensions ({test_img_rgb_pil.width}x{test_img_rgb_pil.height}). Skipping ROI.")


    # Return RGB version of test image for drawing annotations later
    output_image_cv_bgr = cv2.cvtColor(np.array(test_img_rgb_pil), cv2.COLOR_RGB2BGR)
    return rois, boxes, output_image_cv_bgr # Return BGR format for OpenCV drawing

# --- Drawing ---
def draw_annotations(image_cv_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: List[str]) -> np.ndarray:
    """Draws bounding boxes and labels on the image (expects BGR)."""
    out = image_cv_bgr.copy()
    for (x,y,w,h), label in zip(boxes, labels):
        # Blue box (BGR color)
        cv2.rectangle(out, (x,y), (x+w, y+h), (255,0,0), 2)
        text_y = max(12, y - 6) # Position label above box
        # Add white background for label text
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, text_y - text_height - baseline), (x + text_width, text_y + baseline), (255, 255, 255), cv2.FILLED)
        # Add blue label text
        cv2.putText(out, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
    return out

# --- Main Service Function (Module 6 Logic) ---
def process_and_classify_defects(template_pil: Image.Image, test_pil: Image.Image, min_area: int = MIN_CONTOUR_AREA_DEFAULT) -> Tuple[np.ndarray, List[Dict]]:
    """
    Main service function: Finds defects, classifies them, draws annotations.
    Returns the annotated OpenCV image (BGR) and a list of defect details.
    """
    # Load model (uses cache)
    model = load_classification_model(MODEL_PATH, len(CLASSES))

    # Find defect ROIs and bounding boxes using image processing pipeline
    rois, boxes, output_image_cv_bgr = find_defects(template_pil, test_pil, min_area=min_area)

    defect_details = []
    if not rois:
        # Return the original test image (converted to BGR) if no defects found
        return cv2.cvtColor(np.array(test_pil.convert('RGB')), cv2.COLOR_RGB2BGR), defect_details

    # Classify each ROI using the loaded model
    labels = [classify_roi(roi, model) for roi in rois]

    # Draw annotations on the output image
    annotated_cv_bgr = draw_annotations(output_image_cv_bgr, boxes, labels)

    # Prepare defect details list (for potential future use, e.g., JSON response)
    for idx, (label, (x,y,w,h)) in enumerate(zip(labels, boxes)):
        defect_details.append({"id": idx+1, "label": label, "x": x, "y": y, "w": w, "h": h})

    # Return annotated image (BGR) and defect list
    return annotated_cv_bgr, defect_details