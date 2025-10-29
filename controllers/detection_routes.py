# controllers/detection_routes.py
from flask import Blueprint, request, jsonify, Response, current_app
from PIL import Image
import io
import cv2
import numpy as np
# Import the service logic function correctly
from services.defect_service import process_and_classify_defects # Correct import
import logging
import base64 # Import base64 encoding

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a Blueprint for detection-related routes, define URL prefix here
detection_bp = Blueprint('detection', __name__, url_prefix='/api')

@detection_bp.route('/detect', methods=['POST'])
def detect_defects_api():
    """API endpoint to detect and classify defects."""
    logging.info("Received request for /api/detect")

    # --- Input Validation ---
    if 'template_image' not in request.files or 'test_image' not in request.files:
        logging.warning("Missing template or test image file in request.")
        return jsonify({"error": "Missing template_image or test_image file"}), 400

    template_file = request.files['template_image']
    test_file = request.files['test_image']

    # Check if filenames are empty
    if not template_file or template_file.filename == '' or \
       not test_file or test_file.filename == '':
        logging.warning("No file selected for template or test image.")
        return jsonify({"error": "No selected file provided for template or test image"}), 400

    # --- Image Loading and Processing ---
    try:
        # Read image streams directly into PIL Images
        template_img_pil = Image.open(template_file.stream).convert("RGB")
        test_img_pil = Image.open(test_file.stream).convert("RGB")
        logging.info("Template and Test images loaded successfully from request.")

        # --- Call the Service Layer ---
        logging.info("Calling defect service to process images...")
        # Service returns annotated image (BGR) AND defect details list
        annotated_cv_bgr, defect_details = process_and_classify_defects(
            template_img_pil, test_img_pil
            # Add min_area passing if needed: min_area=request.form.get('min_area', MIN_CONTOUR_AREA_DEFAULT, type=int)
        )
        logging.info(f"Service processing complete. Found {len(defect_details)} defects.")

        # --- Prepare Response ---
        # 1. Encode annotated image to PNG bytes
        is_success, img_encoded = cv2.imencode('.png', annotated_cv_bgr)
        if not is_success:
            logging.error("Failed to encode annotated image to PNG format.")
            raise RuntimeError("Failed to encode annotated image to PNG")

        image_bytes = img_encoded.tobytes()

        # 2. Encode image bytes to Base64 string for embedding in JSON
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_data_url = f"data:image/png;base64,{image_base64}" # Create data URL

        # 3. Return JSON response containing the image data URL and defect details list
        logging.info("Sending JSON response with image data URL and defect details.")
        return jsonify({
            "annotated_image_url": image_data_url,
            "defects": defect_details # The list returned by the service
        })

    # --- Error Handling ---
    except FileNotFoundError as e:
        logging.error(f"Model file not found error during processing: {e}", exc_info=True)
        return jsonify({"error": f"Configuration error: Could not find required model file. {str(e)}"}), 500
    except RuntimeError as e: # Catch encoding errors or model load errors
        logging.error(f"Runtime error during processing: {e}", exc_info=True)
        return jsonify({"error": f"Failed during processing: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"Unexpected error processing images: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during image processing."}), 500

# Ensure __init__.py exists in 'services' and 'controllers' folders.