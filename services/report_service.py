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
        # Check if we have enough space for the title
        if self.get_y() + 10 > self.page_break_trigger:
            self.add_page()
            self.set_y(20) # Reset Y pos

        self.set_font('Helvetica', 'B', 12)
        # --- FIX: Removed background color ---
        # self.set_fill_color(230, 230, 230) # No more gray background
        self.cell(0, 6, title, 0, 1, 'L', False) # Set fill=False
        # --- END FIX ---
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
                # Check if adding this image will overflow the page
                if self.get_y() + img_h + 10 > self.page_break_trigger:
                    self.add_page()
                    self.set_y(20) # Reset Y pos
                self.image(buf, w=w)
                self.set_font('Helvetica', 'I', 8)
                self.cell(0, 10, title, 0, 1, 'C')
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
                self.image(buf, w=w)
                self.set_font('Helvetica', 'I', 8)
                self.cell(0, 10, title, 0, 1, 'C')
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

        # Check if we need to add a page for the header
        if self.get_y() + 7 > self.page_break_trigger:
            self.add_page()
            self.set_y(20)

        for h in headers:
            self.cell(col_width, 7, h, 1, 0, 'C')
        self.ln()

        self.set_font('Helvetica', '', 9)
        for d in defects:
            # Check if we need to add a new page for the next row
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

def create_pdf_report(template_pil, test_pil, annotated_bgr, defects, summary, bar_fig, pie_fig, scatter_fig):
    """
    Main function to generate the PDF report in a professional, auto-flowing layout.
    """
    pdf = PDFReport()
    # Set margins to 15mm (for the border at 10mm)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    # Enable auto page breaks with a 15mm bottom margin
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page() # Start Page 1

    # --- 1. PROJECT BACKGROUND (NEW STRUCTURE) ---
    pdf.set_y(20) # Reset Y pos
    pdf.add_chapter_title('1. Project Background')
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, "This report details the automated defect analysis performed by the CircuitGuard system. The system employs a two-stage computer vision pipeline:")
    pdf.ln(2)

    # Bullet point 1
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(5, 5, "1. ")
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, "**Defect Detection:** Uses OpenCV for image alignment, subtraction, and thresholding (Otsu's method) to isolate potential defect regions from a template image.")
    pdf.ln(1)

    # Bullet point 2
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(5, 5, "2. ")
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, "**Defect Classification:** Each isolated defect is classified by an **EfficientNet-B4** deep learning model. This model was trained on the DeepPCB dataset to an accuracy of **98%** and can identify six distinct defect classes (short, spur, open, etc.).")
    pdf.ln(4)

    # Date
    pdf.set_font('Helvetica', 'I', 9) # Italic for the date
    pdf.cell(0, 5, f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)

    # --- 2. INPUT IMAGES ---
    pdf.add_chapter_title('2. Input Images')
    page_width = pdf.epw / 2 - 5 # Effective page width / 2, minus gap

    y_start_inputs = pdf.get_y() # Get Y before images
    y_after_template = y_start_inputs
    if template_pil:
        pdf.add_image_from_pil(template_pil, "Template Image", w=page_width)
        y_after_template = pdf.get_y()

    # Reset Y to start, move X to the right column
    pdf.set_y(y_start_inputs)
    pdf.set_x(page_width + 20) # 15mm margin + page_width + 5mm gap
    y_after_test = y_start_inputs

    if test_pil:
        pdf.add_image_from_pil(test_pil, "Test Image", w=page_width)
        y_after_test = pdf.get_y()

    # Set Y to the bottom of the taller image
    pdf.set_y(max(y_after_template, y_after_test))
    pdf.set_x(15) # Reset X
    pdf.ln(5) # Add a small gap before next section

    # --- 3. ANALYSIS SUMMARY ---
    pdf.add_chapter_title('3. Analysis Summary')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, f"Total Defects Found: {len(defects)}", 0, 1)
    if summary:
        for label, count in summary.items():
            pdf.cell(0, 6, f"  - {label.capitalize()}: {count}", 0, 1)
    pdf.ln(5)

    # --- 4. DEFECT DETAILS ---
    pdf.add_chapter_title('4. Defect Details')
    pdf.add_defect_table(defects)
    pdf.ln(5)

    # --- 5. VISUALIZATIONS ---
    pdf.add_chapter_title('5. Visualizations')

    y_start_charts = pdf.get_y() # Get Y before charts
    y_after_bar = y_start_charts

    if bar_fig:
        pdf.add_image_from_fig(bar_fig, "Defect Count per Class", w=page_width)
        y_after_bar = pdf.get_y()
        plt.close(bar_fig) # Close fig individually

    # Reset Y to start, move X to the right column
    pdf.set_y(y_start_charts)
    pdf.set_x(page_width + 20)
    y_after_pie = y_start_charts

    if pie_fig:
        pdf.add_image_from_fig(pie_fig, "Defect Class Distribution", w=page_width)
        y_after_pie = pdf.get_y()
        plt.close(pie_fig) # Close fig individually

    # Set Y to the bottom of the *taller* of the two charts
    pdf.set_y(max(y_after_bar, y_after_pie))
    pdf.set_x(15) # Reset X

    if scatter_fig:
        pdf.add_image_from_fig(scatter_fig, "Defect Scatter Plot", w=pdf.epw * 0.8) # 80% width
        plt.close(scatter_fig) # Close fig individually
    pdf.ln(5)

    # --- 6. ANNOTATED IMAGE ---
    if annotated_bgr is not None:
        pdf.add_chapter_title('6. Annotated Image')
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
        # Use 75% of the effective page width for "medium sized"
        pdf.add_image_from_pil(annotated_pil, "Final Annotated Result", w=pdf.epw * 0.75)
        pdf.ln(5)

    # Return as bytes
    return pdf.output()