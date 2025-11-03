import os
from typing import List, Tuple, Dict
import numpy as np
import cv2
from PIL import Image

# Optional torch/timm import guarded so the API still works without GPU
try:
    import torch
    import timm
    from torchvision import transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'final_model.pth')
CLASSES = ['copper', 'mousebite', 'open', 'pin-hole', 'short', 'spur']
IMG_SIZE = 128
MIN_CONTOUR_AREA_DEFAULT = 5

_model_cache = {}

def _load_model():
    if not TORCH_AVAILABLE:
        return None
    if MODEL_PATH in _model_cache:
        return _model_cache[MODEL_PATH]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=len(CLASSES))
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    try:
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    _model_cache[MODEL_PATH] = (model, device)
    return _model_cache[MODEL_PATH]

def _classify_roi(roi: Image.Image) -> Tuple[str, float]:
    if not TORCH_AVAILABLE:
        return 'unknown', 0.0
    model_device = _load_model()
    if model_device is None:
        return 'unknown', 0.0
    model, device = model_device
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        tensor = transform(roi.convert('RGB')).unsqueeze(0).to(device)
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = probs.max(dim=1)
        label = CLASSES[int(idx.item())] if 0 <= int(idx.item()) < len(CLASSES) else 'unknown'
        return label, float(conf.item())

def _find_defects(template_pil: Image.Image, test_pil: Image.Image, diff_threshold: int, morph_iterations: int, min_area: int):
    template_gray = np.array(template_pil.convert('L'))
    test_gray = np.array(test_pil.convert('L'))
    h, w = template_gray.shape
    test_gray = cv2.resize(test_gray, (w, h))

    diff = cv2.absdiff(template_gray, test_gray)
    if diff_threshold > 0:
        _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois, boxes, areas = [], [], []
    test_rgb = test_pil.convert('RGB')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= min_area:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww <= 0 or hh <= 0:
            continue
        x2, y2 = x + ww, y + hh
        if x2 > test_rgb.width or y2 > test_rgb.height:
            continue
        rois.append(test_rgb.crop((x, y, x2, y2)))
        boxes.append((x, y, ww, hh))
        areas.append(area)

    diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    output_bgr = cv2.cvtColor(np.array(test_rgb), cv2.COLOR_RGB2BGR)
    return rois, boxes, output_bgr, diff_bgr, mask_bgr, areas

def _draw_boxes(image_bgr: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: List[str]) -> np.ndarray:
    out = image_bgr.copy()
    for (x, y, w, h), label in zip(boxes, labels):
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(out, label, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
    return out

def process_and_classify_defects(template_pil: Image.Image, test_pil: Image.Image, diff_threshold: int = 0, morph_iterations: int = 2, min_area: int = MIN_CONTOUR_AREA_DEFAULT) -> Dict:
    rois, boxes, out_bgr, diff_bgr, mask_bgr, areas = _find_defects(template_pil, test_pil, diff_threshold, morph_iterations, min_area)

    details = []
    labels = []
    for idx, (roi, box, area) in enumerate(zip(rois, boxes, areas)):
        label, conf = _classify_roi(roi)
        labels.append(label)
        x, y, w, h = box
        details.append({
            'id': idx + 1,
            'label': label,
            'confidence': round(float(conf), 4),
            'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
            'area': round(float(area), 2)
        })

    annotated = _draw_boxes(out_bgr, boxes, labels) if boxes else out_bgr
    return {
        'annotated_image_bgr': annotated,
        'diff_image_bgr': diff_bgr,
        'mask_image_bgr': mask_bgr,
        'defects': details
    }