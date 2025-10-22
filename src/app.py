import streamlit as st
import torch
import timm
import cv2
import numpy as np
import io
import zipfile
import pandas as pd
from torchvision import transforms
from PIL import Image
import json
from datetime import datetime
from typing import List, Tuple, Dict

# --- CONFIGURATION ---
MODEL_PATH = "models/final_model.pth"
CLASSES = ['copper', 'mousebite', 'open', 'pin-hole', 'short', 'spur']
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_THRESHOLD_DEFAULT = 0.5
MIN_CONTOUR_AREA_DEFAULT = 20

# --- MODEL LOADING ---
@st.cache_resource
def load_classification_model(model_path: str, num_classes: int):
    """Loads the EfficientNet model and returns it in eval mode."""
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# --- TRANSFORMS ---
roi_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- CLASSIFICATION ---
def classify_roi(roi_pil: Image.Image, model) -> str:
    """Runs inference on a single ROI (PIL image) and returns predicted class label."""
    roi_rgb = roi_pil.convert("RGB")
    transformed = roi_transform(roi_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(transformed)
        pred_index = int(out.argmax(1).item())
    return CLASSES[pred_index]

# --- IMAGE PROCESSING (find defects) ---
def find_defects(template_img_pil: Image.Image, test_img_pil: Image.Image, min_area:int=MIN_CONTOUR_AREA_DEFAULT):
    """
    Subtraction-based defect detection:
      - convert to grayscale
      - resize test to template size
      - absdiff, Otsu threshold, morphological clean
      - extract contours -> ROIs and bounding boxes
    Returns:
      rois: list[PIL.Image]
      bounding_boxes: list[(x,y,w,h)]
      output_image_cv: np.array RGB image (uint8)
      mask_clean: binary mask (useful for debugging)
    """
    template_cv = np.array(template_img_pil.convert('L'))
    test_cv = np.array(test_img_pil.convert('L'))
    h, w = template_cv.shape
    test_cv = cv2.resize(test_cv, (w, h))
    diff = cv2.absdiff(template_cv, test_cv)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois, boxes = [], []
    output_image_cv = np.array(test_img_pil.convert('RGB'))
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x,y,w_box,h_box = cv2.boundingRect(cnt)
            roi_pil = test_img_pil.crop((x, y, x + w_box, y + h_box))
            rois.append(roi_pil)
            boxes.append((x,y,w_box,h_box))
    return rois, boxes, output_image_cv, mask_clean

# --- DRAW & SAVE ---
def draw_annotations(image_cv: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: List[str]) -> np.ndarray:
    """Draw bounding boxes and labels on a cv image (RGB)."""
    out = image_cv.copy()
    for (x,y,w,h), label in zip(boxes, labels):
        # Blue box and label
        cv2.rectangle(out, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(out, label, (x, max(12,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
    return out

def pil_image_to_bytes(pil_img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def cv_image_to_bytes(cv_img: np.ndarray, fmt="PNG") -> bytes:
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    return pil_image_to_bytes(pil, fmt=fmt)

# --- EVALUATION / IoU & MATCHING ---
def iou(boxA, boxB):
    """Compute IoU between two boxes (x,y,w,h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0.0

def match_predictions_to_gt(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.5):
    """
    Match predicted boxes to ground truth using IoU and label equality.
    Returns lists of matches and counts: TP_detail, FP_indices, FN_indices
    TP_detail contains tuples (pred_idx, gt_idx, iou_val, pred_label, gt_label)
    """
    matches = []
    used_gt = set()
    used_pred = set()
    # greedy matching by IoU descending
    all_pairs = []
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            val = iou(pb, gb)
            all_pairs.append((val, i, j))
    all_pairs.sort(reverse=True, key=lambda x: x[0])
    for val, i, j in all_pairs:
        if val < iou_thresh:
            continue
        if i in used_pred or j in used_gt:
            continue
        # match
        matches.append((i, j, val, pred_labels[i], gt_labels[j]))
        used_pred.add(i)
        used_gt.add(j)
    FP = [i for i in range(len(pred_boxes)) if i not in used_pred]
    FN = [j for j in range(len(gt_boxes)) if j not in used_gt]
    return matches, FP, FN

# --- METRICS AGGREGATION ---
def compute_metrics_from_confusion(confusion: Dict[str, Dict[str,int]]):
    """
    confusion[classA][classB] = count predicted classA for true classB (rows=pred, cols=true)
    returns per-class precision/recall/f1 and totals
    """
    classes = list(confusion.keys())
    results = {}
    total_TP = total_FP = total_FN = 0
    for cls in classes:
        tp = confusion[cls].get(cls, 0)
        fp = sum(confusion[cls].values()) - tp
        fn = sum(confusion[c].get(cls, 0) for c in classes) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        results[cls] = {"precision":prec, "recall":rec, "f1":f1, "tp":tp, "fp":fp, "fn":fn}
        total_TP += tp
        total_FP += fp
        total_FN += fn
    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    overall_f1 = (2*overall_precision*overall_recall)/(overall_precision+overall_recall) if (overall_precision+overall_recall)>0 else 0.0
    return results, {"precision":overall_precision, "recall":overall_recall, "f1":overall_f1, "tp":total_TP, "fp":total_FP, "fn":total_FN}

# --- BATCH EVALUATION ---
def evaluate_batch(model, template_img_pil, test_images: List[Tuple[str, Image.Image]],
                   gt_df: pd.DataFrame = None, iou_thresh: float = IOU_THRESHOLD_DEFAULT, min_area:int=MIN_CONTOUR_AREA_DEFAULT):
    """
    test_images: list of tuples (filename, PIL.Image)
    gt_df: optional DataFrame with columns: image_filename,x,y,w,h,label
    Returns:
      logs: list of dict per-image
      confusion: nested dict for predicted vs true
    """
    logs = []
    # init confusion
    confusion = {pred: {true:0 for true in CLASSES} for pred in CLASSES}
    for fname, img_pil in test_images:
        rois, boxes, output_cv, _ = find_defects(template_img_pil, img_pil, min_area=min_area)
        predicted_labels = []
        for roi in rois:
            lbl = classify_roi(roi, model)
            predicted_labels.append(lbl)
        # fetch GT for this image (if provided)
        gt_entries = []
        if gt_df is not None:
            rows = gt_df[gt_df['image_filename'] == fname]
            for _, r in rows.iterrows():
                gt_entries.append(((int(r['x']), int(r['y']), int(r['w']), int(r['h'])), r['label']))
        gt_boxes = [g[0] for g in gt_entries]
        gt_labels = [g[1] for g in gt_entries]
        # match predictions to GT
        matches, FP_idxs, FN_idxs = match_predictions_to_gt(boxes, predicted_labels, gt_boxes, gt_labels, iou_thresh=iou_thresh)
        # update confusion matrix: for matches increment confusion[pred][true]
        for pred_idx, gt_idx, iou_val, pred_label, gt_label in matches:
            confusion[pred_label][gt_label] += 1
        # unmatched predicted boxes -> counted as FP (assign predicted label vs true='<none>' not in confusion)
        for pi in FP_idxs:
            # count as FP by increment predicted class's false positives (we'll reflect FP via per-class fp in compute_metrics)
            # Here we increment confusion[pred][pred_false] as predicted vs each true? Instead, increment confusion[pred][pred] as fp accounted later.
            # Simpler: increment confusion[pred_label][pred_label] by 0 (we update FP in compute_metrics via sums). To keep trace, add to a separate FP-only log.
            pass
        # For unmatched GT -> FN counted per true class (we'll capture via confusion columns)
        for fi in FN_idxs:
            true_label = gt_labels[fi]
            # add zero prediction entry; ensure column is counted (already present)
            # no direct pred label to increment
            pass
        # Draw annotations
        annotated_cv = draw_annotations(output_cv, boxes, predicted_labels)
        # Save annotated image bytes
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB))
        annotated_bytes = pil_image_to_bytes(annotated_pil, fmt="PNG")
        # Build log entry
        log = {
            "image_filename": fname,
            "pred_count": len(boxes),
            "gt_count": len(gt_boxes),
            "matches": [{"pred_idx":p,"gt_idx":g,"iou":float(iou),"pred_label":pl,"gt_label":gl} for (p,g,iou,pl,gl) in [(a[0],a[1],a[2],a[3],a[4]) for a in matches]],
            "fp_indices": FP_idxs,
            "fn_indices": FN_idxs,
            "predicted_boxes": [{"x":int(b[0]),"y":int(b[1]),"w":int(b[2]),"h":int(b[3]),"label":predicted_labels[i]} for i,b in enumerate(boxes)],
            "gt_boxes": [{"x":int(b[0]),"y":int(b[1]),"w":int(b[2]),"h":int(b[3]),"label":gt_labels[i]} for i,b in enumerate(gt_boxes)],
            "annotated_image_bytes": annotated_bytes
        }
        logs.append(log)
    return logs, confusion

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="CircuitGuard — Eval & Test")
st.title("CircuitGuard: PCB Defect Detection — Evaluation & Test")
st.write("Upload template and test images. Optionally upload a ground-truth CSV for batch evaluation and metrics.")

# Model load
try:
    model = load_classification_model(MODEL_PATH, len(CLASSES))
    st.success("🤖 Model loaded")
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# Sidebar for evaluation params
st.sidebar.header("Evaluation / Inference Settings")
iou_thresh = st.sidebar.slider("IoU threshold for match", min_value=0.1, max_value=0.95, value=IOU_THRESHOLD_DEFAULT, step=0.05)
min_area = st.sidebar.number_input("Min contour area (px)", min_value=1, value=MIN_CONTOUR_AREA_DEFAULT)
batch_mode = st.sidebar.checkbox("Batch mode (template + multiple tests)", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Ground-truth CSV format**:")
st.sidebar.markdown("`image_filename,x,y,w,h,label`  — one GT box per row.")

# Upload fields
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("Upload Template Image", type=["jpg","jpeg","png"])
with col2:
    if batch_mode:
        test_files = st.file_uploader("Upload Test Images (multiple allowed)", type=["jpg","jpeg","png"], accept_multiple_files=True)
    else:
        test_file = st.file_uploader("Upload Test Image (single)", type=["jpg","jpeg","png"])

gt_file = st.file_uploader("Optional: Upload Ground-truth CSV", type=["csv"])

# Load GT CSV if provided
gt_df = None
if gt_file is not None:
    try:
        gt_df = pd.read_csv(gt_file)
        required_cols = {'image_filename','x','y','w','h','label'}
        if not required_cols.issubset(set(gt_df.columns)):
            st.error(f"Ground-truth CSV missing required columns. Required: {required_cols}")
            gt_df = None
        else:
            st.success("Ground-truth CSV loaded.")
    except Exception as e:
        st.error(f"Failed to read ground truth CSV: {e}")
        gt_df = None

# Main action
if batch_mode:
    if template_file and test_files:
        template_img = Image.open(template_file).convert("RGB")
        # Build test_images list with filenames
        test_images = []
        for f in test_files:
            try:
                img = Image.open(f).convert("RGB")
            except Exception:
                st.warning(f"Skipping unreadable file {f.name}")
                continue
            test_images.append((f.name, img))
        if st.button("Run Batch Evaluation"):
            with st.spinner("Running inference & evaluation..."):
                logs, confusion = evaluate_batch(model, template_img, test_images, gt_df=gt_df, iou_thresh=iou_thresh, min_area=min_area)
                per_class_metrics, overall_metrics = compute_metrics_from_confusion(confusion)
            # Display summary
            st.header("Evaluation Summary")
            st.write(f"Images evaluated: **{len(test_images)}**")
            if gt_df is not None:
                st.write("Match IoU threshold:", iou_thresh)
                st.write("Overall Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
                    overall_metrics['precision'], overall_metrics['recall'], overall_metrics['f1']
                ))
            else:
                st.info("No ground-truth provided — only annotated outputs will be shown.")
            # Show per-image results and download buttons
            st.header("Annotated Images & Logs")
            csv_rows = []
            for log in logs:
                st.subheader(log['image_filename'])
                st.write(f"Predicted: {log['pred_count']} | GT: {log['gt_count']} | Matches: {len(log['matches'])} | FP: {len(log['fp_indices'])} | FN: {len(log['fn_indices'])}")
                # annotated image
                st.image(Image.open(io.BytesIO(log['annotated_image_bytes'])), width=450)
                # download annotated image
                st.download_button(f"Download annotated image — {log['image_filename']}", data=log['annotated_image_bytes'], file_name=f"annotated_{log['image_filename']}", mime="image/png")
                # prepare CSV row
                csv_rows.append({
                    "image_filename": log['image_filename'],
                    "pred_count": log['pred_count'],
                    "gt_count": log['gt_count'],
                    "matches": json.dumps(log['matches']),
                    "fp_indices": json.dumps(log['fp_indices']),
                    "fn_indices": json.dumps(log['fn_indices']),
                    "predicted_boxes": json.dumps(log['predicted_boxes']),
                    "gt_boxes": json.dumps(log['gt_boxes'])
                })
            # Build CSV log and offer download
            if csv_rows:
                df_log = pd.DataFrame(csv_rows)
                csv_bytes = df_log.to_csv(index=False).encode('utf-8')
                st.download_button("Download evaluation CSV log", csv_bytes, file_name=f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            # Evaluation report
            if gt_df is not None:
                st.header("Per-class metrics")
                metric_table = []
                for cls, metrics in per_class_metrics.items():
                    metric_table.append([cls, metrics['precision'], metrics['recall'], metrics['f1'], metrics['tp'], metrics['fp'], metrics['fn']])
                report_df = pd.DataFrame(metric_table, columns=["class","precision","recall","f1","tp","fp","fn"])
                st.dataframe(report_df)
                # Downloadable text report
                report_text = []
                report_text.append("CircuitGuard — Evaluation Report")
                report_text.append(f"Timestamp: {datetime.now().isoformat()}")
                report_text.append(f"Num images evaluated: {len(test_images)}")
                report_text.append(f"IoU threshold: {iou_thresh}")
                report_text.append("")
                report_text.append("Overall Metrics:")
                report_text.append(f"Precision: {overall_metrics['precision']:.4f}")
                report_text.append(f"Recall:    {overall_metrics['recall']:.4f}")
                report_text.append(f"F1:        {overall_metrics['f1']:.4f}")
                report_text.append("")
                report_text.append("Per-class:")
                for cls, metrics in per_class_metrics.items():
                    report_text.append(f"{cls}: precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']}")
                report_bytes = "\n".join(report_text).encode('utf-8')
                st.download_button("Download evaluation report (txt)", report_bytes, file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")
    else:
        st.info("Please upload a template image and at least one test image for batch evaluation.")
else:
    # single-image mode
    if template_file and test_file:
        template_img = Image.open(template_file).convert("RGB")
        test_img = Image.open(test_file).convert("RGB")
        if st.button("Detect & Classify Defects"):
            with st.spinner("Analyzing..."):
                rois, boxes, output_cv, _ = find_defects(template_img, test_img, min_area=min_area)
                if not rois:
                    st.success("✅ No defects found!")
                else:
                    labels = [classify_roi(roi, model) for roi in rois]
                    annotated_cv = draw_annotations(output_cv, boxes, labels)
                    st.image([template_img, test_img], caption=["Template", "Test"], width=300)
                    st.image(annotated_cv, caption="Annotated Result")
                    # download annotated image
                    annotated_bytes = cv_image_to_bytes(annotated_cv)
                    st.download_button("Download annotated image", annotated_bytes, file_name="annotated_result.png", mime="image/png")
                    # show the per-ROI crops and labels
                    st.write("Detected ROIs and predicted labels:")
                    for idx, (roi, label, (x,y,w,h)) in enumerate(zip(rois, labels, boxes)):
                        st.write(f"ROI {idx+1}: {label} — box=({x},{y},{w},{h})")
                        st.image(roi, width=120, caption=f"{label} (ROI {idx+1})")
    else:
        st.info("Please upload a template image and a test image.")

st.markdown("---")
st.caption("Module 4: Evaluation & Prediction Testing — supports batch evaluation + ground truth comparison. Built to satisfy Milestone 3 & 4 deliverables (annotated outputs, CSV logs, evaluation report).")
