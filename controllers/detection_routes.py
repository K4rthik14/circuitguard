# controllers/detection_routes.py
from flask import Blueprint, request, jsonify, send_file
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
from services.report_service import create_pdf_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

detection_bp = Blueprint('detection', __name__, url_prefix='/api')

def _to_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode('.png', img_bgr)
    if not ok:
        raise RuntimeError('encode failed')
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('utf-8')

# Chart Functions

def _create_bar_chart_fig(summary_data: dict):
    if not summary_data: return None
    labels = list(summary_data.keys()); counts = list(summary_data.values())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts, color='#36A2EB', edgecolor='#36A2EB', linewidth=1, alpha=0.6)
    ax.set_ylabel('Defect Count'); ax.set_title('Defect Count per Class')
    plt.xticks(rotation=45, ha='right')
    return fig

def _create_pie_chart_fig(summary_data: dict):
    if not summary_data: return None
    labels = list(summary_data.keys()); counts = list(summary_data.values())
    color_map = { 'copper': '#FF9F40', 'mousebite': '#4BC0C0', 'open': '#36A2EB', 'pin-hole': '#FFCE56', 'short': '#FF6384', 'spur': '#9966FF', 'unknown': '#C9CBCF' }
    colors = [color_map.get(l, color_map['unknown']) for l in labels]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal'); ax.set_title('Defect Class Distribution')
    return fig # <-- RETURN FIG

def _create_scatter_chart_fig(defects: list):
    if not defects: return None
    grouped_defects = {}
    for d in defects:
        label = d['label']
        if label not in grouped_defects: grouped_defects[label] = {'x': [], 'y': []}
        grouped_defects[label]['x'].append(d['x']); grouped_defects[label]['y'].append(d['y'])
    colors = { 'copper': '#FF9F40', 'mousebite': '#4BC0C0', 'open': '#36A2EB', 'pin-hole': '#FFCE56', 'short': '#FF6384', 'spur': '#9966FF', 'unknown': '#C9CBCF' }
    fig, ax = plt.subplots(figsize=(6, 4)) # Made scatter plot shorter
    for label, coords in grouped_defects.items():
        ax.scatter(coords['x'], coords['y'], label=label, c=colors.get(label, colors['unknown']), s=30, alpha=0.7)
    ax.set_title('Defect Scatter Plot'); ax.set_xlabel('X Position (px)'); ax.set_ylabel('Y Position (px)')
    ax.legend(loc='best'); ax.invert_yaxis(); ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def _fig_to_base64(fig):
    """Helper to convert fig to base64 for JSON and close it."""
    if fig is None: return ""
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) # Close fig to save memory
    return 'data:image/png;base64,' + img_base64

# --- detect_defects_api

@detection_bp.route('/detect', methods=['POST'])
def detect_defects_api():
    if 'template_image' not in request.files or 'test_image' not in request.files:
        return jsonify({"error": "Missing template_image or test_image"}), 400

    template_file = request.files.get('template_image')
    test_file = request.files.get('test_image')
    if not template_file or not template_file.filename or not test_file or not test_file.filename:
        return jsonify({"error": "No files selected"}), 400

    try:
        template_stream = io.BytesIO(template_file.stream.read())
        test_stream = io.BytesIO(test_file.stream.read())

        template_pil = Image.open(template_stream).convert('RGB')
        test_pil = Image.open(test_stream).convert('RGB')

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

        bar_fig = _create_bar_chart_fig(summary)
        pie_fig = _create_pie_chart_fig(summary)
        scatter_fig = _create_scatter_chart_fig(defects_list)

        bar_chart_url = _fig_to_base64(bar_fig)
        pie_chart_url = _fig_to_base64(pie_fig)
        scatter_chart_url = _fig_to_base64(scatter_fig)

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

# --- download_report_api

@detection_bp.route('/download_report', methods=['POST'])
def download_report_api():
    if 'template_image' not in request.files or 'test_image' not in request.files:
        return jsonify({"error": "Missing template_image or test_image"}), 400
    try:
        template_pil = Image.open(request.files['template_image'].stream).convert('RGB')
        test_pil = Image.open(request.files['test_image'].stream).convert('RGB')

        diff_threshold = int(request.form.get('diffThreshold', 0))
        min_area = int(request.form.get('minArea', MIN_CONTOUR_AREA_DEFAULT))
        morph_iterations = int(request.form.get('morphIter', 2))

        # 1. Run the full analysis
        result = process_and_classify_defects(
            template_pil, test_pil, diff_threshold, morph_iterations, min_area
        )
        summary = result.get('summary', {})
        defects_list = result.get('defects', [])

        # 2. Generate the chart figures
        bar_fig = _create_bar_chart_fig(summary)
        pie_fig = _create_pie_chart_fig(summary)
        scatter_fig = _create_scatter_chart_fig(defects_list)

        # Generate the PDF

        pdf_bytes = create_pdf_report(
            template_pil,
            test_pil,
            result['diff_image_bgr'],
            result['mask_image_bgr'],
            result['annotated_image_bgr'],
            defects_list,
            summary,
            bar_fig,
            pie_fig,
            scatter_fig
        )


        #Send the PDF as a file
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='CircuitGuard_Report.pdf'
        )
    except Exception as e:
        logging.exception("/api/download_report failed")
        return jsonify({"error": str(e)}), 500