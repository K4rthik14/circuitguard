import streamlit as st
import torch
import timm
import cv2
import numpy as np
import io
from torchvision import transforms
from PIL import Image
from typing import List, Tuple
import os # Added for os.path.basename

# --- CONFIGURATION ---
MODEL_PATH = "models/final_model.pth"
CLASSES = ['copper', 'mousebite', 'open', 'pin-hole', 'short', 'spur']
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_CONTOUR_AREA_DEFAULT = 20

# --- MODEL LOADING ---
@st.cache_resource # Caches the model so it loads only once
def load_classification_model(model_path: str, num_classes: int):
    """Loads the trained EfficientNet model."""
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
    Performs subtraction, thresholding, and contour extraction to find ROIs.
    Returns:
      rois: list[PIL.Image] - Cropped defect images.
      bounding_boxes: list[(x,y,w,h)] - Coordinates for drawing.
      output_image_cv: np.array RGB image to draw on.
    """
    template_cv = np.array(template_img_pil.convert('L'))
    test_cv = np.array(test_img_pil.convert('L'))
    h, w = template_cv.shape
    test_cv = cv2.resize(test_cv, (w, h))

    # Image Subtraction & Thresholding
    diff = cv2.absdiff(template_cv, test_cv)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Contour Extraction
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois, boxes = [], []
    output_image_cv = np.array(test_img_pil.convert('RGB')) # Use test image for output

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x,y,w_box,h_box = cv2.boundingRect(cnt)
            # Crop ROI from the original PIL test image
            roi_pil = test_img_pil.crop((x, y, x + w_box, y + h_box))
            rois.append(roi_pil)
            boxes.append((x,y,w_box,h_box))

    return rois, boxes, output_image_cv

# --- DRAW & SAVE UTILITIES ---
def draw_annotations(image_cv: np.ndarray, boxes: List[Tuple[int,int,int,int]], labels: List[str]) -> np.ndarray:
    """Draw bounding boxes and labels on the output image."""
    out = image_cv.copy()
    for (x,y,w,h), label in zip(boxes, labels):
        # Blue box and label
        cv2.rectangle(out, (x,y), (x+w, y+h), (255,0,0), 2)
        # Ensure text is visible
        text_y = max(12, y - 6)
        # Add background rectangle for better text visibility (optional)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, text_y - text_height - baseline), (x + text_width, text_y + baseline), (255, 255, 255), cv2.FILLED)
        cv2.putText(out, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
    return out

def cv_image_to_bytes(cv_img: np.ndarray, fmt="PNG") -> bytes:
    """Converts OpenCV image (BGR) to bytes for download."""
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="CircuitGuard: PCB Defect Detection")
st.title("CircuitGuard: PCB Defect Detection & Classification")
st.write("Upload a defect-free **Template** image and a **Test** image to automatically identify and classify manufacturing defects.")

# --- SIDEBAR ---
st.sidebar.header("Settings")
min_area = st.sidebar.number_input(
    "Minimum Defect Area (pixels)",
    min_value=1,
    value=MIN_CONTOUR_AREA_DEFAULT,
    help="Filters out very small contours detected during image subtraction (adjust if noise is detected)."
)
st.sidebar.markdown("---")
st.sidebar.info(f"Model: `{os.path.basename(MODEL_PATH)}`")
st.sidebar.info(f"Defect Classes: `{', '.join(CLASSES)}`")

# --- MODEL LOADING ---
try:
    model = load_classification_model(MODEL_PATH, len(CLASSES))
    st.success(f"✅ Trained Model ({os.path.basename(MODEL_PATH)}) loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Model file not found at '{MODEL_PATH}'. Please ensure the model exists.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- UI FOR IMAGE UPLOAD ---
st.header("1. Upload Images")
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("Upload Template Image (Defect-Free)", type=["jpg","jpeg","png"], key="template")
with col2:
    test_file = st.file_uploader("Upload Test Image (With Potential Defects)", type=["jpg","jpeg","png"], key="test")

# --- PROCESSING AND DISPLAY ---
if template_file and test_file:
    template_img = Image.open(template_file).convert("RGB")
    test_img = Image.open(test_file).convert("RGB")

    st.header("2. Input Images Preview")
    st.image([template_img, test_img], caption=["Uploaded Template", "Uploaded Test"], width=350)

    if st.button("🚀 Run Defect Detection & Classification", type="primary"):
        with st.spinner("⏳ Processing... Analyzing images using the trained model..."):
            # Run the core pipeline
            rois, boxes, output_cv = find_defects(template_img, test_img, min_area=min_area)

            if not rois:
                st.success("✅ No defects found in the test image compared to the template!")
            else:
                st.info(f"🔎 Found {len(rois)} potential defect regions. Classifying each...")

                # Classify each detected ROI
                labels = [classify_roi(roi, model) for roi in rois]

                # Draw annotations
                annotated_cv = draw_annotations(output_cv, boxes, labels)

                st.header("3. Results")
                st.subheader("Annotated Test Image")
                st.image(annotated_cv, caption="Test Image with Detected Defects and Predicted Labels", use_column_width=True)

                # Provide download for the annotated image
                annotated_bytes = cv_image_to_bytes(annotated_cv)
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=annotated_bytes,
                    file_name=f"annotated_{test_file.name}",
                    mime="image/png"
                )

                # Display details about each detected defect
                st.subheader("Detected Defect Details")
                defect_data = []
                cols = st.columns(min(len(rois), 5)) # Display up to 5 defects side-by-side
                for idx, (roi, label, (x,y,w,h)) in enumerate(zip(rois, labels, boxes)):
                    defect_data.append({"Defect ID": idx+1, "Label": label, "X": x, "Y": y, "Width": w, "Height": h})
                    with cols[idx % 5]:
                        st.image(roi, width=120, caption=f"#{idx+1}: {label}")
                        st.write(f"`({x},{y},{w},{h})`")

                # Optionally display defect data in a table
                if defect_data:
                     st.dataframe(defect_data)


st.markdown("---")
st.caption("Powered by CircuitGuard - Automated PCB Defect Detection.")