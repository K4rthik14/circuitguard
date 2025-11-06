from flask import Blueprint, request, jsonify
from PIL import Image
import cv2
import numpy as np
import base64
import logging
import matplotlib
import matplotlib.pyplot as plt
import io
matplotlib.use('Agg')
from services.defect_service import process_and_classify_defects, MIN_CONTOUR_AREA_DEFAULT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

detection_bp = Blueprint('detection', __name__, url_prefix='/api')

def _to_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode('.png', img_bgr)
    if not ok:
        raise RuntimeError('encode failed')
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('utf-8')

# detection_routes.py

# detection_routes.py

def _create_bar_chart_base64(summary_data: dict) -> str:
    if not summary_data: return ""

    labels = list(summary_data.keys())
    counts = list(summary_data.values())

    fig, ax = plt.subplots(figsize=(5, 4))

    # --- FIX ---
    # Use hex code for color and a separate alpha
    ax.bar(labels, counts,
           color='#36A2EB',  # Hex for 'rgba(54,162,235,0.6)'
           edgecolor='#36A2EB',
           linewidth=1,
           alpha=0.6) # Separate alpha
    # --- END FIX ---

    ax.set_ylabel('Defect Count')
    ax.set_title('Defect Count per Class')
    plt.xticks(rotation=45, ha='right')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return 'data:image/png;base64,' + img_base64

def _create_pie_chart_base64(summary_data: dict) -> str:
    if not summary_data: return ""

    labels = list(summary_data.keys())
    counts = list(summary_data.values())

    # --- FIX ---
    # Use matplotlib-friendly hex codes
    color_map = {
        'copper': '#FF9F40',
        'mousebite': '#4BC0C0',
        'open': '#36A2EB',
        'pin-hole': '#FFCE56',
        'short': '#FF6384',
        'spur': '#9966FF',
        'unknown': '#C9CBCF'
    }
    # --- END FIX ---

    colors = [color_map.get(l, color_map['unknown']) for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    ax.set_title('Defect Class Distribution')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return 'data:image/png;base64,' + img_base64

def _create_scatter_chart_base64(defects: list) -> str:
    if not defects: return ""

    # 1. Group defects by label
    grouped_defects = {}
    for d in defects:
        label = d['label']
        if label not in grouped_defects: grouped_defects[label] = {'x': [], 'y': []}
        grouped_defects[label]['x'].append(d['x'])
        grouped_defects[label]['y'].append(d['y'])

    # 2. Define colors using HEX CODES (matplotlib-friendly)
    colors = {
        'copper': '#FF9F40',
        'mousebite': '#4BC0C0',
        'open': '#36A2EB',
        'pin-hole': '#FFCE56',
        'short': '#FF6384',
        'spur': '#9966FF',
        'unknown': '#C9CBCF'
    }

    fig, ax = plt.subplots(figsize=(6, 5))

    for label, coords in grouped_defects.items():
        # --- THE FIX ---
        # 'c' is now a single hex string.
        # 'alpha' is a separate argument.
        ax.scatter(coords['x'], coords['y'],
                   label=label,
                   c=colors.get(label, colors['unknown']),  # <-- Just the hex string
                   s=30,
                   alpha=0.7) # <-- Transparency is now here
        # --- END FIX ---

    ax.set_title('Defect Scatter Plot')
    ax.set_xlabel('X Position (px)')
    ax.set_ylabel('Y Position (px)')
    ax.legend(loc='best')

    # Invert Y-axis (image coordinates have 0,0 at top-left)
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return 'data:image/png;base64,' + img_base64


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

        summary = result.get('summary', {})
        defects_list = result.get('defects', [])

        bar_chart_url = _create_bar_chart_base64(summary)
        pie_chart_url = _create_pie_chart_base64(summary)
        scatter_chart_url = _create_scatter_chart_base64(defects_list)

        payload = {
            "annotated_image_url": _to_data_url(result["annotated_image_bgr"]),
            "diff_image_url": _to_data_url(result["diff_image_bgr"]),
            "mask_image_url": _to_data_url(result["mask_image_bgr"]),
            "defects": result["defects"],
            "bar_chart_url": bar_chart_url,
            "pie_chart_url": pie_chart_url,
            "scatter_chart_url": scatter_chart_url
        }
        return jsonify(payload)
    except Exception as e:
        logging.exception("/api/detect failed")
        return jsonify({"error": str(e)}), 500