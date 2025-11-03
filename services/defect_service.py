# services/defect_service.py
import torch
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import os
from typing import List, Tuple, Dict

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_model.pth")
CLASSES = ['copper', 'mousebite', 'open', 'pin-hole', 'short', 'spur']
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_CONTOUR_AREA_DEFAULT = 5 # Default Min Area

# --- Model Loading (Cached) ---
_model_cache = {}
def load_classification_model(model_path: str, num_classes: int):
    """Loads the EfficientNet model."""
    if model_path in _model_cache:
        return _model_cache[model_path]

    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Model file not found at {model_path}")

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
    try:
        # Try loading with weights_only=True for security
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except Exception:
        # Fallback for older .pth files
        print("Warning: Loading model with weights_only=True failed. Attempting weights_only=False (less secure).")
        state = torch.load(model_path, map_location=DEVICE, weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    _model_cache[model_path] = model
    return model

# --- Transforms ---
roi_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Classification (Returns label and confidence) ---
def classify_roi(roi_pil: Image.Image, model) -> Tuple[str, float]:
    """Runs inference on a single ROI and returns (label, confidence)."""
    roi_rgb = roi_pil.convert("RGB")
    transformed = roi_transform(roi_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(transformed)
        probabilities = torch.softmax(out, dim=1)
        confidence = probabilities.max().item()
        pred_index = int(probabilities.argmax(1).item())
    
    label = CLASSES[pred_index] if 0 <= pred_index < len(CLASSES) else "unknown"
    return label, float(confidence)

# --- Image Processing (Accepts parameters) ---
def find_defects(
    template_img_pil: Image.Image,
    test_img_pil: Image.Image,
    diff_threshold: int = 0,
    morph_iterations: int = 2,
    min_area: int = MIN_CONTOUR_AREA_DEFAULT
) -> Tuple[List[Image.Image], List[Tuple], np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    Performs subtraction, thresholding, and contour extraction.
    Returns: ROIs, boxes, processed BGR image, diff_img, mask_clean, areas
    """
    template_cv = np.array(template_img_pil.convert('L'))
    test_cv = np.array(test_img_pil.convert('L'))
    h, w = template_cv.shape
    test_cv = cv2.resize(test_cv, (w, h))

    diff = cv2.absdiff(template_cv, test_cv)

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
            if x + w_box <= test_img_rgb_pil.width and y + h_box <= test_img_rgb_pil.height:
                if w_box > 0 and h_box > 0:
                    roi_pil = test_img_rgb_pil.crop((x, y, x + w_box, y + h_box))
                    rois.append(roi_pil)
                    boxes.append((x, y, w_box, h_box))
                    areas.append(area)
    
    diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    mask_clean_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
    output_image_cv_bgr = cv2.cvtColor(np.array(test_img_rgb_pil), cv2.COLOR_RGB2BGR)
    
    return rois, boxes, output_image_cv_bgr, diff_bgr, mask_clean_bgr, areas

# --- Drawing ---
def draw_annotations(image_cv_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: List[str]) -> np.ndarray:
    out = image_cv_bgr.copy()
    img_height, img_width = out.shape[:2]
    padding = 5 

    for (x,y,w,h), label in zip(boxes, labels):
        x1_pad = max(0, x - padding)
        y1_pad = max(0, y - padding)
        x2_pad = min(img_width, x + w + padding)
        y2_pad = min(img_height, y + h + padding)
        
        cv2.rectangle(out, (x1_pad, y1_pad), (x2_pad, y2_pad), (255,0,0), 2) # Blue box

        text_y = max(12, y1_pad - 6)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bg_y1 = max(0, text_y - text_height - baseline)
        bg_y2 = text_y + baseline
        cv2.rectangle(out, (x1_pad, bg_y1), (x1_pad + text_width, bg_y2), (255, 255, 255), cv2.FILLED)
        cv2.putText(out, label, (x1_pad, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
        
    return out

# --- Main Service Function (Returns all data) ---
def process_and_classify_defects(
    template_pil: Image.Image,
    test_pil: Image.Image,
    diff_threshold: int = 0,
    morph_iterations: int = 2,
    min_area: int = MIN_CONTOUR_AREA_DEFAULT
) -> Dict:
    """Main service: finds, classifies, annotates, and returns all results."""
    model = load_classification_model(MODEL_PATH, len(CLASSES))

    rois, boxes, output_image_cv_bgr, diff_img_bgr, mask_img_bgr, areas = find_defects(
        template_pil, test_pil,
        diff_threshold=diff_threshold,
        morph_iterations=morph_iterations,
        min_area=min_area
    )

    defect_details = []
    annotated_cv_bgr = cv2.cvtColor(np.array(test_pil.convert('RGB')), cv2.COLOR_RGB2BGR) # Start with original image

    if not rois:
        print("No ROIs found meeting minimum area criteria.")
    else:
        print(f"Classifying {len(rois)} ROIs...")
        labels_and_confidences = [classify_roi(roi, model) for roi in rois]
        labels = [item[0] for item in labels_and_confidences]
        annotated_cv_bgr = draw_annotations(output_image_cv_bgr, boxes, labels)

        for idx, ((label, confidence), (x, y, w, h), area) in enumerate(zip(labels_and_confidences, boxes, areas)):
            defect_details.append({
                "id": idx + 1,
                "label": label,
                "confidence": round(float(confidence), 4),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": round(float(area), 2)
            })
        print(f"Prepared details for {len(defect_details)} defects.")
        
    return {
        "annotated_image_bgr": annotated_cv_bgr,
        "diff_image_bgr": diff_img_bgr,
        "mask_image_bgr": mask_img_bgr,
        "defects": defect_details
    }