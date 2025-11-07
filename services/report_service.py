# services/report_service.py
from fpdf import FPDF
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

class PDFReport(FPDF):
    """
    Custom PDF class to define a header and footer (with border).
    """
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'CircuitGuard - Defect Analysis Report', 0, 1, 'C')
        # Reset Y to 20mm (inside the top margin) for content
        self.set_y(20)

    def footer(self):
        # --- PAGE BORDER ---
        self.set_draw_color(0, 0, 0) # Black
        self.set_line_width(0.3)
        # Draw rect at 10mm margins (from edge of page)
        self.rect(10, 10, self.w - 20, self.h - 20)

        # --- PAGE NUMBER ---
        self.set_y(-15) # Position 15mm from bottom
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_chapter_title(self, title):
        """Adds a formatted chapter title, handling page breaks."""
        if self.get_y() + 10 > self.page_break_trigger:
            self.add_page()
            self.set_y(20) # Reset Y pos

        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 6, title, 0, 1, 'L', False)
        self.ln(4) # Add a 4mm space after the title

    def add_body_text(self, text):
        """Adds a multi-line block of text, with auto-wrapping."""
        if self.get_y() + 10 > self.page_break_trigger:
            self.add_page()
            self.set_y(20)
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, text) # 5mm line height
        self.ln(2)

    def add_image_from_pil(self, pil_img, title, w=190):
        """Adds a PIL Image to the PDF, handling page breaks."""
        try:
            with io.BytesIO() as buf:
                pil_img.save(buf, format='PNG')
                buf.seek(0)
                img_h = (pil_img.height * w) / pil_img.width
                if self.get_y() + img_h + 10 > self.page_break_trigger:
                    self.add_page()
                    self.set_y(20) # Reset Y pos

                # Get current X to center the image block if it's not side-by-side
                current_x = self.get_x()
                if current_x < 16: # Check if we are at the left margin
                    self.set_x((self.w - w) / 2)

                self.image(buf, w=w)
                self.set_font('Helvetica', 'I', 8)
                self.set_x(current_x)
                self.cell(w, 10, title, 0, 1, 'C')
                self.ln(2)
        except Exception as e:
            print(f"Error adding PIL image {title}: {e}")

    def add_image_from_fig(self, fig, title, w=190):
        """Adds a Matplotlib Figure to the PDF, handling page breaks."""
        try:
            with io.BytesIO() as buf:
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_h = (fig.get_figheight() * w) / fig.get_figwidth()
                if self.get_y() + img_h + 10 > self.page_break_trigger:
                    self.add_page()
                    self.set_y(20) # Reset Y pos

                current_x = self.get_x()
                if current_x < 16:
                    self.set_x((self.w - w) / 2)

                self.image(buf, w=w)
                self.set_font('Helvetica', 'I', 8)
                self.set_x(current_x)
                self.cell(w, 10, title, 0, 1, 'C')
                self.ln(2)
        except Exception as e:
            print(f"Error adding Matplotlib fig {title}: {e}")

    def add_defect_table(self, defects):
        """Adds the table of defect details, handling page breaks."""
        if not defects:
            if self.get_y() + 10 > self.page_break_trigger:
                self.add_page()
                self.set_y(20)
            self.cell(0, 10, "No defects found.", 0, 1)
            return

        self.set_font('Helvetica', 'B', 10)
        col_width = self.epw / 6  # Effective page width / 6 columns
        headers = ['#', 'Class', 'Confidence', 'Position', 'Size', 'Area']

        if self.get_y() + 7 > self.page_break_trigger:
            self.add_page()
            self.set_y(20)

        for h in headers:
            self.cell(col_width, 7, h, 1, 0, 'C')
        self.ln()

        self.set_font('Helvetica', '', 9)
        for d in defects:
            if self.get_y() + 6 > self.page_break_trigger:
                self.add_page()
                self.set_y(20) # Reset Y pos
                self.set_font('Helvetica', 'B', 10)
                for h in headers:
                    self.cell(col_width, 7, h, 1, 0, 'C')
                self.ln()
                self.set_font('Helvetica', '', 9)

            self.cell(col_width, 6, str(d['id']), 1)
            self.cell(col_width, 6, d['label'], 1)
            self.cell(col_width, 6, f"{d['confidence']*100:.1f}%", 1)
            self.cell(col_width, 6, f"({d['x']}, {d['y']})", 1)
            self.cell(col_width, 6, f"({d['w']}, {d['h']})", 1)
            self.cell(col_width, 6, str(d['area']), 1)
            self.ln()

def create_pdf_report(template_pil, test_pil, diff_bgr, mask_bgr, annotated_bgr, defects, summary, bar_fig, pie_fig, scatter_fig):
    """
    Main function to generate the PDF report in the new user-specified order.
    """
    pdf = PDFReport()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page() # Start Page 1

    # --- 1. INPUT IMAGES (User Order 1) ---
    pdf.set_y(20) # Reset Y pos
    pdf.add_chapter_title('1. Input Images')
    page_width = pdf.epw / 2 - 5 # Effective page width / 2, minus gap

    y_start_inputs = pdf.get_y()
    y_after_template = y_start_inputs
    if template_pil:
        pdf.add_image_from_pil(template_pil, "Template Image", w=page_width)
        y_after_template = pdf.get_y()

    pdf.set_y(y_start_inputs)
    pdf.set_x(page_width + 20)
    y_after_test = y_start_inputs

    if test_pil:
        pdf.add_image_from_pil(test_pil, "Test Image", w=page_width)
        y_after_test = pdf.get_y()

    pdf.set_y(max(y_after_template, y_after_test))
    pdf.set_x(15)
    pdf.ln(5)

    # --- 2. PREPROCESSING STEPS (NEW SECTION) ---
    pdf.add_chapter_title('2. Preprocessing Steps')

    # Convert BGR (OpenCV) images to RGB (PIL)
    diff_pil = Image.fromarray(cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))

    y_start_process = pdf.get_y()
    y_after_diff = y_start_process
    if diff_pil:
        pdf.add_image_from_pil(diff_pil, "Difference Image", w=page_width)
        y_after_diff = pdf.get_y()

    pdf.set_y(y_start_process)
    pdf.set_x(page_width + 20)
    y_after_mask = y_start_process

    if mask_pil:
        pdf.add_image_from_pil(mask_pil, "Binary Mask", w=page_width)
        y_after_mask = pdf.get_y()

    pdf.set_y(max(y_after_diff, y_after_mask))
    pdf.set_x(15)
    pdf.ln(5)

    # --- 3. BACKEND PROCESS INSIGHT (NEW SECTION) ---
    pdf.add_chapter_title('3. Backend Process Insight')
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, "The analysis is a two-stage pipeline executed by the Python backend:\n")

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(5, 5, "1. ")
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, "**Defect Detection (OpenCV):** The 'Template Image' and 'Test Image' are loaded, converted to grayscale, and aligned. An absolute difference is calculated, resulting in the 'Difference Image'. This is converted to the 'Binary Mask' using Otsu's thresholding and morphological operations to remove noise. Finally, `cv2.findContours` identifies each separate defect region.")
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(5, 5, "2. ")
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, "**Defect Classification (PyTorch):** Each defect region (ROI) is cropped from the original Test Image and passed to a pre-trained **EfficientNet-B4** deep learning model. The model, trained on the DeepPCB dataset to **98% accuracy**, returns a final class label (e.g., 'short', 'spur') and a confidence score for each defect.")
    pdf.ln(5)

    # --- 4. ANALYSIS SUMMARY & DEFECT DETAILS ---
    pdf.add_chapter_title('4. Analysis Summary & Defect Details')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, f"Total Defects Found: {len(defects)}", 0, 1)
    if summary:
        for label, count in summary.items():
            pdf.cell(0, 6, f"  - {label.capitalize()}: {count}", 0, 1)
    pdf.ln(5)

    pdf.add_defect_table(defects)
    pdf.ln(5)

    # --- 5. VISUALIZATIONS ---
    pdf.add_chapter_title('5. Visualizations')

    y_start_charts = pdf.get_y()
    y_after_bar = y_start_charts

    if bar_fig:
        pdf.add_image_from_fig(bar_fig, "Defect Count per Class", w=page_width)
        y_after_bar = pdf.get_y()
        plt.close(bar_fig)

    pdf.set_y(y_start_charts)
    pdf.set_x(page_width + 20)
    y_after_pie = y_start_charts

    if pie_fig:
        pdf.add_image_from_fig(pie_fig, "Defect Class Distribution", w=page_width)
        y_after_pie = pdf.get_y()
        plt.close(pie_fig)

    pdf.set_y(max(y_after_bar, y_after_pie))
    pdf.set_x(15)

    if scatter_fig:
        pdf.ln(5)
        pdf.add_image_from_fig(scatter_fig, "Defect Scatter Plot", w=pdf.epw * 0.8) # 80% width
        plt.close(scatter_fig)
    pdf.ln(5)

    # --- 6. ANNOTATED IMAGE ---
    if annotated_bgr is not None:
        pdf.add_chapter_title('6. Annotated Image')
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
        # 75% width for "medium sized", and centered
        pdf.add_image_from_pil(annotated_pil, "Final Annotated Result", w=pdf.epw * 0.75)
        pdf.ln(5)

    # Return as bytes
    return pdf.output()