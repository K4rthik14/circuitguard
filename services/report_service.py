# services/report_service.py
from fpdf import FPDF
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import cv2

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

        #PAGE BORDER
        self.set_draw_color(0, 0, 0) # Black
        self.set_line_width(0.3)
        # Draw rect at 10mm margins (from edge of page)
        self.rect(10, 10, self.w - 20, self.h - 20)
        # --- END NEW BORDER ---

    def add_chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 6, title, 0, 1, 'L', True)
        self.ln(4)

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
            self.cell(0, 10, "No defects found.", 0, 1)
            return

        self.set_font('Helvetica', 'B', 10)
        col_width = self.epw / 6  # Effective page width / 6 columns
        headers = ['#', 'Class', 'Confidence', 'Position', 'Size', 'Area']
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
    Main function to generate the PDF report in the user-specified order.
    """
    pdf = PDFReport()
    # Set margins to 15mm (for the border at 10mm)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page() # Start Page 1

    # --- 1. INPUT IMAGES (User Order 1) ---
    pdf.set_y(20) # Reset Y pos
    pdf.add_chapter_title('Input Images')
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

    #SUMMARY & DEFECT DETAILS
    pdf.add_page() # Start Page 2
    pdf.set_y(20) # Reset Y pos
    pdf.add_chapter_title('Summary')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, f"Total Defects Found: {len(defects)}", 0, 1)
    if summary:
        for label, count in summary.items():
            pdf.cell(0, 6, f"  - {label.capitalize()}: {count}", 0, 1)
    pdf.ln(5)

    pdf.add_chapter_title('Defect Details')
    pdf.add_defect_table(defects)
    pdf.ln(5)


    # --- 3. VISUALIZATIONS (User Order 3) ---
    pdf.add_page() # Start Page 3
    pdf.set_y(20) # Reset Y pos
    pdf.add_chapter_title('Visualizations')

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
        # pdf.ln(5) # <-- FIX 1: REMOVED this line to tighten layout
        pdf.add_image_from_fig(scatter_fig, "Defect Scatter Plot", w=pdf.epw * 0.8)
        plt.close(scatter_fig) # Close fig individually

    # --- 4. ANNOTATED IMAGE (User Order 4) ---
    if annotated_bgr is not None:
        pdf.add_page() # Start Page 4
        pdf.set_y(20) # Reset Y pos
        pdf.add_chapter_title('Annotated Image')
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
        # --- FIX 2: CHANGED to 0.6 from 0.75 to make the image smaller ---
        pdf.add_image_from_pil(annotated_pil, "Final Annotated Result", w=pdf.epw * 0.6)
        pdf.ln(5)

    # Return as bytes
    return pdf.output()