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

def _create_bar_chart_base64(summary_data: dict) -> str:
    if not summary_data: return ""

    labels = list(summary_data.keys())
    counts = list(summary_data.values())

    fig, ax = plt.subplots(figsize=(5, 4)) # Create a figure and an axes
    ax.bar(labels, counts, color='rgba(54,162,235,0.6)', edgecolor='rgba(54,162,235,1)', linewidth=1)

    ax.set_ylabel('Defect Count')
    ax.set_title('Defect Count per Class')
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels

    # Save to a memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) # IMPORTANT: Close the figure to free up memory
    return 'data:image/png;base64,' + img_base64

def _create_pie_chart_base64(summary_data: dict) -> str:
    if not summary_data: return ""

    labels = list(summary_data.keys())
    counts = list(summary_data.values())

    # Define colors to match your original JS
    color_map = { 'copper':'rgba(255,159,64,0.7)','mousebite':'rgba(75,192,192,0.7)','open':'rgba(54,162,235,0.7)','pin-hole':'rgba(255,206,86,0.7)','short':'rgba(255,99,132,0.7)','spur':'rgba(153,102,255,0.7)','unknown':'rgba(201,203,207,0.7)' }
    colors = [color_map.get(l, color_map['unknown']) for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Defect Class Distribution')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return 'data:image/png;base64,' + img_base64

def _create_scatter_chart_base64(defects: list) -> str:
    if not defects: return ""

    # 1. Group defects by label for colored plotting
    grouped_defects = {}
    for d in defects:
        label = d['label']
        if label not in grouped_defects: grouped_defects[label] = {'x': [], 'y': []}
        grouped_defects[label]['x'].append(d['x'])
        grouped_defects[label]['y'].append(d['y'])

    # 2. Define colors
    colors = { 'copper': 'rgba(255,159,64,0.7)','mousebite': 'rgba(75,192,192,0.7)','open': 'rgba(54,162,235,0.7)','pin-hole': 'rgba(255,206,86,0.7)','short': 'rgba(255,99,132,0.7)','spur': 'rgba(153,102,255,0.7)','unknown': 'rgba(201,203,207,0.7)' }

    fig, ax = plt.subplots(figsize=(6, 5))

    for label, coords in grouped_defects.items():
        ax.scatter(coords['x'], coords['y'],
                   label=label,
                   color=colors.get(label, colors['unknown']),
                   s=30, alpha=0.7)

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
            "defects": result["defects"]
        }
        return jsonify(payload)
    except Exception as e:
        logging.exception("/api/detect failed")
        return jsonify({"error": str(e)}), 500