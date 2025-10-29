# controllers/detection_routes.py
from flask import Blueprint, request, jsonify, Response, current_app
from PIL import Image
import io
import cv2
import numpy as np
from services.defect_service import process_and_classify_defects
import logging
import base64 # Import base64 encoding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
detection_bp = Blueprint('detection', __name__, url_prefix='/api')

@detection_bp.route('/detect', methods=['POST'])
def detect_defects_api():
    logging.info("Received request for /api/detect")
    # --- Input Validation (keep as is) ---
    if 'template_image' not in request.files or 'test_image' not in request.files:
        # ... (error handling)
        return jsonify({"error": "Missing template_image or test_image file"}), 400
    template_file = request.files['template_image']
    test_file = request.files['test_image']
    if not template_file or template_file.filename == '' or not test_file or test_file.filename == '':
        # ... (error handling)
         return jsonify({"error": "No selected file provided for template or test image"}), 400

    try:
        template_img_pil = Image.open(template_file.stream).convert("RGB")
        test_img_pil = Image.open(test_file.stream).convert("RGB")
        logging.info("Template and Test images loaded.")

        # --- Call Service ---
        logging.info("Calling defect service...")
        # Service now returns annotated image (BGR) AND defect details list
        annotated_cv_bgr, defect_details = process_and_classify_defects(
            template_img_pil, test_img_pil
        )
        logging.info(f"Service complete. Found {len(defect_details)} defects.")

        # --- Prepare Response ---
        # 1. Encode annotated image to PNG bytes
        is_success, img_encoded = cv2.imencode('.png', annotated_cv_bgr)
        if not is_success:
            raise RuntimeError("Failed to encode annotated image")
        image_bytes = img_encoded.tobytes()

        # 2. Encode image bytes to Base64 string
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_data_url = f"data:image/png;base64,{image_base64}"

        # 3. Return JSON response containing image data URL and defect details
        logging.info("Sending JSON response with image and defect details.")
        return jsonify({
            "annotated_image_url": image_data_url,
            "defects": defect_details # The list from defect_service
        })

    # --- Error Handling (keep as is) ---
    except FileNotFoundError as e:
        # ... (error handling)
        return jsonify({"error": f"Configuration error: Model file not found. {str(e)}"}), 500
    except Exception as e:
        # ... (error handling)
        return jsonify({"error": "An internal server error occurred."}), 500