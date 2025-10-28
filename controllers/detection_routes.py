# controllers/detection_routes.py
from flask import Blueprint, request, jsonify, Response, current_app, send_file
from PIL import Image
import io
import cv2
import numpy as np
# Import the service logic function directly
from services.defect_service import process_and_classify_defects
import logging # Added for better logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a Blueprint for detection-related routes
detection_bp = Blueprint('detection', __name__, url_prefix='/api') # Define URL prefix here

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

    # Check if filenames are empty (user might not have selected a file)
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
        # This performs the core logic: find defects, classify, annotate
        logging.info("Calling defect service to process images...")
        annotated_cv_bgr, defect_details = process_and_classify_defects(
            template_img_pil, test_img_pil
            # Consider adding min_area from request.form if you want it configurable via API
        )
        logging.info(f"Service processing complete. Found {len(defect_details)} defects.")

        # --- Prepare and Send Response ---
        # Convert the final annotated OpenCV image (BGR format) to PNG bytes
        is_success, img_encoded = cv2.imencode('.png', annotated_cv_bgr)
        if not is_success:
            # Log the error if encoding fails
            logging.error("Failed to encode annotated image to PNG format.")
            raise RuntimeError("Failed to encode annotated image to PNG")

        image_bytes = img_encoded.tobytes()

        # Return the annotated image directly as a file response
        logging.info("Sending annotated image as PNG response.")
        return Response(image_bytes, mimetype='image/png')

    # --- Error Handling ---
    except FileNotFoundError as e: # Specific error for model file issues
        logging.error(f"Model file not found error during processing: {e}", exc_info=True)
        # Return a more specific error for configuration issues
        return jsonify({"error": f"Configuration error: Could not find required model file. {str(e)}"}), 500
    except RuntimeError as e: # Specific error for image encoding issues
        logging.error(f"Image encoding error during processing: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate the output image."}), 500
    except Exception as e:
        # Log the full exception traceback for any other unexpected errors
        logging.error(f"Unexpected error processing images: {e}", exc_info=True)
        # Return a generic server error message to the client
        return jsonify({"error": "An internal server error occurred during image processing."}), 500

# Remember to create __init__.py files in 'services' and 'controllers' folders
# to make them Python packages.