# controllers/detection_routes.py
from flask import Blueprint, request, jsonify, Response, current_app
from PIL import Image
import io
import cv2
import numpy as np
from services.defect_service import process_and_classify_defects, MIN_CONTOUR_AREA_DEFAULT # Import service and default
import logging
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
detection_bp = Blueprint('detection', __name__, url_prefix='/api')

def encode_image_to_data_url(img_bgr: np.ndarray) -> str:
    """Encodes an OpenCV BGR image to a Base64 Data URL."""
    is_success, img_encoded = cv2.imencode('.png', img_bgr)
    if not is_success:
        raise RuntimeError("Failed to encode image to PNG")
    image_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

@detection_bp.route('/detect', methods=['POST'])
def detect_defects_api():
    """API endpoint to detect and classify defects."""
    logging.info("Received request for /api/detect")

    if 'template_image' not in request.files or 'test_image' not in request.files:
        logging.warning("Missing template or test image file.")
        return jsonify({"error": "Missing template_image or test_image file"}), 400

    template_file = request.files['template_image']
    test_file = request.files['test_image']

    if not template_file or template_file.filename == '' or not test_file or test_file.filename == '':
        logging.warning("No file selected for template or test image.")
        return jsonify({"error": "No selected file provided for template or test image"}), 400

    try:
        template_img_pil = Image.open(template_file.stream).convert("RGB")
        test_img_pil = Image.open(test_file.stream).convert("RGB")
        logging.info("Template and Test images loaded.")

        # --- Get parameters from the frontend (with defaults) ---
        diff_threshold = int(request.form.get('diffThreshold', 0)) # Match JS name
        min_area = int(request.form.get('minArea', MIN_CONTOUR_AREA_DEFAULT)) # Match JS name
        morph_iterations = int(request.form.get('morphIter', 2)) # Match JS name
        
        logging.info(f"Parameters received -> diff_threshold={diff_threshold}, morph_iterations={morph_iterations}, min_area={min_area}")

        # --- Call the Service Layer ---
        logging.info("Calling defect service...")
        # Service returns a dictionary with all data
        results = process_and_classify_defects(
            template_img_pil,
            test_img_pil,
            diff_threshold=diff_threshold,
            morph_iterations=morph_iterations,
            min_area=min_area
        )
        logging.info(f"Service complete. Found {len(results['defects'])} defects.")

        # --- Prepare Full JSON Response ---
        logging.info("Encoding images for JSON response...")
        response_data = {
            "annotated_image_url": encode_image_to_data_url(results["annotated_image_bgr"]),
            "diff_image_url": encode_image_to_data_url(results["diff_image_bgr"]),
            "mask_image_url": encode_image_to_data_url(results["mask_image_bgr"]),
            "defects": results["defects"] # The list from the service
        }
        
        logging.info("Sending JSON response with all images and defect details.")
        return jsonify(response_data)

    except FileNotFoundError as e:
        logging.error(f"Model file not found: {e}", exc_info=True)
        return jsonify({"error": f"Configuration error: Model file not found. {str(e)}"}), 500
    except RuntimeError as e:
        logging.error(f"Runtime error: {e}", exc_info=True)
        return jsonify({"error": f"Failed during processing: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500