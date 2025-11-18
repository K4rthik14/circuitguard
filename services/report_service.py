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

    # --- NEW, ROBUST IMAGE FUNCTIONS ---
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

                # Get current X/Y
                current_x = self.get_x()
                current_y = self.get_y()

                # Draw the image at the current X, Y
                self.image(buf, x=current_x, y=current_y, w=w)

                # Set font for caption
                self.set_font('Helvetica', 'I', 8)

                # Move Y *below* the image
                self.set_y(current_y + img_h + 2)
                # Set X back to the image's X for centered caption
                self.set_x(current_x)
                self.cell(w, 10, title, 0, 0, 'C') # ln=0 to not move down

                # Return the total height consumed
                return img_h + 10
        except Exception as e:
            print(f"Error adding PIL image {title}: {e}")
            return 0

    def add_image_from_fig(self, fig, title, w=190):
        """Adds a Matplotlib Figure to the PDF, handling page breaks."""
        try:
            with io.BytesIO() as buf:
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                # Calculate image aspect ratio
                img_props = plt.imread(buf)
                fig_h, fig_w, _ = img_props.shape
                img_h = (fig_h * w) / fig_w

                if self.get_y() + img_h + 10 > self.page_break_trigger:
                    self.add_page()
                    self.set_y(20) # Reset Y pos

                # Get current X/Y
                current_x = self.get_x()
                current_y = self.get_y()

                # Reset buffer and draw image
                buf.seek(0)
                self.image(buf, x=current_x, y=current_y, w=w)

                self.set_font('Helvetica', 'I', 8)

                # Move Y *below* the image
                self.set_y(current_y + img_h + 2)
                # Set X back to the image's X for centered caption
                self.set_x(current_x)
                self.cell(w, 10, title, 0, 0, 'C') # ln=0 to not move down

                # Return the total height consumed
                return img_h + 10
        except Exception as e:
            print(f"Error adding Matplotlib fig {title}: {e}")
            return 0
    # --- END NEW IMAGE FUNCTIONS ---

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

    def check_page_break(self, height_needed):
        """Checks if height_needed fits, if not, adds a new page."""
        if self.get_y() + height_needed > self.page_break_trigger:
            self.add_page()
            self.set_y(20)

def create_pdf_report(template_pil, test_pil, diff_bgr, mask_bgr, annotated_bgr, defects, summary, bar_fig, pie_fig, scatter_fig):
    """
    Main function to generate the PDF report in a professional, side-by-side layout.
    """
    pdf = PDFReport()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page() # Start Page 1

    # --- 1. PROJECT BACKGROUND (FIXED BOLDING) ---
    pdf.set_y(20) # Reset Y pos
    pdf.add_chapter_title('1. Project Background')
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, "This report details the automated defect analysis performed by the CircuitGuard system. The system employs a two-stage computer vision pipeline:")
    pdf.ln(2)

    # Bullet point 1
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(5, 5, "1. ")
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 5, "Defect Detection:", 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_x(20) # Indent
    pdf.multi_cell(0, 5, "Uses OpenCV for image alignment, subtraction, and thresholding (Otsu's method) to isolate potential defect regions from a template image.")
    pdf.ln(1)

    # Bullet point 2
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(5, 5, "2. ")
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 5, "Defect Classification:", 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_x(20) # Indent
    pdf.multi_cell(0, 5, "Each isolated defect is classified by an EfficientNet-B4 deep learning model. This model was trained on the DeepPCB dataset to an accuracy of 98% and can identify six distinct defect classes (short, spur, open, etc.).")
    pdf.ln(4)

    # Date
    pdf.set_font('Helvetica', 'I', 9) # Italic for the date
    pdf.cell(0, 5, f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)

    # --- 2. INPUT IMAGES (Side-by-side) ---
    pdf.check_page_break(80)
    pdf.add_chapter_title('2. Input Images')
    page_width = pdf.epw / 2 - 5 # Effective page width / 2, minus gap

    y_start_inputs = pdf.get_y() # Get Y before images
    h1 = 0
    if template_pil:
        h1 = pdf.add_image_from_pil(template_pil, "Template Image", w=page_width)

    pdf.set_xy(page_width + 20, y_start_inputs) # 15mm margin + page_width + 5mm gap
    h2 = 0
    if test_pil:
        h2 = pdf.add_image_from_pil(test_pil, "Test Image", w=page_width)

    pdf.set_y(max(y_start_inputs + h1, y_start_inputs + h2) + 5) # Move Y to below the tallest image
    pdf.set_x(15) # Reset X
    pdf.ln(5)

    # --- 3. PREPROCESSING STEP
    pdf.check_page_break(80)
    pdf.add_chapter_title('3. Preprocessing Steps')

    diff_pil = Image.fromarray(cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))

    y_start_process = pdf.get_y()
    h1_proc = 0
    if diff_pil:
        h1_proc = pdf.add_image_from_pil(diff_pil, "Difference Image", w=page_width)

    pdf.set_xy(page_width + 20, y_start_process)
    h2_proc = 0
    if mask_pil:
        h2_proc = pdf.add_image_from_pil(mask_pil, "Binary Mask", w=page_width)

    pdf.set_y(max(y_start_process + h1_proc, y_start_process + h2_proc) + 5)
    pdf.set_x(15)
    pdf.ln(5)

    # ANALYSIS SUMMARY
    pdf.check_page_break(40)
    pdf.add_chapter_title('4. Analysis Summary')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, f"Total Defects Found: {len(defects)}", 0, 1)
    if summary:
        for label, count in summary.items():
            pdf.cell(0, 6, f"  - {label.capitalize()}: {count}", 0, 1)
    pdf.ln(5)

    #DEFECT DETAILS
    pdf.check_page_break(50)
    pdf.add_chapter_title('5. Defect Details')
    pdf.add_defect_table(defects)
    pdf.ln(5)

    # VISUALIZATIONS
    pdf.check_page_break(100)
    pdf.add_chapter_title('6. Visualizations')

    y_start_charts = pdf.get_y()
    h1_chart = 0
    if bar_fig:
        h1_chart = pdf.add_image_from_fig(bar_fig, "Defect Count per Class", w=page_width)
        plt.close(bar_fig)

    pdf.set_xy(page_width + 20, y_start_charts)
    h2_chart = 0
    if pie_fig:
        h2_chart = pdf.add_image_from_fig(pie_fig, "Defect Class Distribution", w=page_width)
        plt.close(pie_fig)

    pdf.set_y(max(y_start_charts + h1_chart, y_start_charts + h2_chart) + 5)
    pdf.set_x(15)

    if scatter_fig:
        pdf.check_page_break(80)
        pdf.ln(5)
        # Center the scatter plot
        pdf.set_x((pdf.w - (pdf.epw * 0.8)) / 2)
        pdf.add_image_from_fig(scatter_fig, "Defect Scatter Plot", w=pdf.epw * 0.8)
        plt.close(scatter_fig)
    pdf.ln(5)

    # ANNOTATED IMAGE
    if annotated_bgr is not None:
        pdf.check_page_break(100)
        pdf.add_chapter_title('7. Annotated Image')
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
        pdf.set_x((pdf.w - (pdf.epw * 0.75)) / 2) # Center the image
        pdf.add_image_from_pil(annotated_pil, "Final Annotated Result", w=pdf.epw * 0.75)
        pdf.ln(5)


    return pdf.output()