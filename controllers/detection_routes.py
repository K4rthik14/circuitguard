from flask import Blueprint, request, jsonify
from PIL import Image
import cv2
import numpy as np
import base64
import logging
from services.defect_service import process_and_classify_defects, MIN_CONTOUR_AREA_DEFAULT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

detection_bp = Blueprint('detection', __name__, url_prefix='/api')

def _to_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode('.png', img_bgr)
    if not ok:
        raise RuntimeError('encode failed')
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('utf-8')

@detection_bp.route('/detect', methods=['POST'])
def detect_defects_api():
    if 'template_image' not in request.files or 'test_image' not in request.files:
        return jsonify({"error": "Missing template_image or test_image"}), 400

    template_file = request.files.get('template_image')
    test_file = request.files.get('test_image')
    if not template_file or not template_file.filename or not test_file or not test_file.filename:
        return jsonify({"error": "No files selected"}), 400

    try:
        template_pil = Image.open(template_file.stream).convert('RGB')
        test_pil = Image.open(test_file.stream).convert('RGB')

        diff_threshold = int(request.form.get('diffThreshold', 0))
        min_area = int(request.form.get('minArea', MIN_CONTOUR_AREA_DEFAULT))
        morph_iterations = int(request.form.get('morphIter', 2))

        result = process_and_classify_defects(
            template_pil, test_pil,
            diff_threshold=diff_threshold,
            morph_iterations=morph_iterations,
            min_area=min_area
        )

        payload = {
            "annotated_image_url": _to_data_url(result["annotated_image_bgr"]),
            "diff_image_url": _to_data_url(result["diff_image_bgr"]),
            "mask_image_url": _to_data_url(result["mask_image_bgr"]),
            "defects": result["defects"]
        }
        return jsonify(payload)
    except Exception as e:
        logging.exception("/api/detect failed")
        return jsonify({"error": str(e)}), 500