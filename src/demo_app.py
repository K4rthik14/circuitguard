import streamlit as st
import torch
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "models/final_model.pth"
CLASSES = ['copper', 'mousebite', 'open', 'pin-hole', 'short', 'spur']
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODULE 6: Backend Pipeline ---
# This function is the "backend logic"

@st.cache_resource  # Caches the model so it loads only once
def load_classification_model(model_path, num_classes):
    """Loads your trained EfficientNet model."""
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Define the same transformations used for your test set
roi_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def classify_roi(roi_pil, model):
    """Runs inference on a single cropped defect (ROI)."""
    
    # --- ADD THIS LINE ---
    # Convert the PIL image to RGB to ensure it has 3 channels
    roi_rgb = roi_pil.convert("RGB")
    
    # Now apply the transform to the RGB image
    transformed_roi = roi_transform(roi_rgb).unsqueeze(0)  # Add batch dimension
    transformed_roi = transformed_roi.to(DEVICE)
    
    with torch.no_grad():
        output = model(transformed_roi)
        pred_index = output.argmax(1).item()
        
    return CLASSES[pred_index]

def find_defects(template_img_pil, test_img_pil):
    """
    This function contains your full Module 1 & 2 pipeline:
    1. Image Subtraction
    2. Thresholding
    3. Contour Extraction
    4. ROI Cropping
    """
    # Convert PIL Images to OpenCV format (Grayscale)
    template_cv = np.array(template_img_pil.convert('L'))
    test_cv = np.array(test_img_pil.convert('L'))
    
    # Ensure images are the same size
    h, w = template_cv.shape
    test_cv = cv2.resize(test_cv, (w, h))

    # 1. Image Subtraction
    diff = cv2.absdiff(template_cv, test_cv)
    
    # 2. Thresholding
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Clean the mask (Erosion/Dilation)
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4. Contour Extraction
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    bounding_boxes = []
    
    # Convert original test image to RGB for drawing
    output_image_cv = np.array(test_img_pil.convert('RGB'))

    for cnt in contours:
        if cv2.contourArea(cnt) > 20:  # Filter out very small noise
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Crop the ROI from the original PIL image
            roi_pil = test_img_pil.crop((x, y, x + w, y + h))
            
            rois.append(roi_pil)
            bounding_boxes.append((x, y, w, h))
            
    return rois, bounding_boxes, output_image_cv

# --- MODULE 5: Web UI ---
#

st.title("CircuitGuard: PCB Defect Detection")
st.write("Upload a template image and a test image to find and classify defects.")

# 1. Load the model
try:
    model = load_classification_model(MODEL_PATH, len(CLASSES))
    st.success("🤖 Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. Add upload fields
col1, col2 = st.columns(2)
with col1:
    template_file = st.file_uploader("Upload Template Image", type=["jpg", "jpeg", "png"])
with col2:
    test_file = st.file_uploader("Upload Test Image", type=["jpg", "jpeg", "png"])

if template_file and test_file:
    # Load images
    template_image = Image.open(template_file)
    test_image = Image.open(test_file)
    
    # Display uploaded images
    st.image([template_image, test_image], caption=["Template Image", "Test Image"], width=300)
    
    if st.button("Detect Defects", type="primary"):
        with st.spinner("Analyzing images..."):
            
            # 3. Run the backend pipeline
            rois, boxes, output_image = find_defects(template_image, test_image)
            
            if not rois:
                st.success("✅ No defects found!")
            else:
                st.info(f"Found {len(rois)} potential defects. Classifying...")
                
                # 4. Classify each defect and draw the result
                for roi, (x, y, w, h) in zip(rois, boxes):
                    
                    # Run inference
                    label = classify_roi(roi, model)
                    
                    # Draw bounding box and label
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue box
                    cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # 5. Display the final annotated image
                st.image(output_image, caption="Final Annotated Image")
                st.success("Processing complete!")