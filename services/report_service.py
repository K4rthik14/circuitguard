# services/report_service.py
from fpdf import FPDF
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import cv2

class PDFReport(FPDF):
    """
    Custom PDF class to define a header and footer for the report.
    """
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'CircuitGuard - Defect Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 6, title, 0, 1, 'L', True)
        self.ln(4)

    def add_image_from_pil(self, pil_img, title, w=190):
        """Adds a PIL Image to the PDF."""
        try:
            with io.BytesIO() as buf:
                pil_img.save(buf, format='PNG')
                buf.seek(0)
                # Check if adding this image will overflow the page
                if self.get_y() + 100 > self.page_break_trigger: # 100 is an estimate
                    self.add_page()
                self.image(buf, w=w)
                self.set_font('Helvetica', 'I', 8)
                self.cell(0, 10, title, 0, 1, 'C')
                self.ln(2)
        except Exception as e:
            print(f"Error adding PIL image {title}: {e}")

    def add_image_from_fig(self, fig, title, w=190):
        """Adds a Matplotlib Figure to the PDF."""
        try:
            with io.BytesIO() as buf:
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                if self.get_y() + 100 > self.page_break_trigger:
                    self.add_page()
                self.image(buf, w=w)
                self.set_font('Helvetica', 'I', 8)
                self.cell(0, 10, title, 0, 1, 'C')
                self.ln(2)
        except Exception as e:
            print(f"Error adding Matplotlib fig {title}: {e}")

    def add_defect_table(self, defects):
        """Adds the table of defect details."""
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
            if self.get_y() + 6 > self.page_break_trigger:
                self.add_page()
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
    Main function to generate the PDF report.
    """
    pdf = PDFReport()
    pdf.add_page()

    # 1. Summary
    pdf.add_chapter_title('Summary')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, f"Total Defects Found: {len(defects)}", 0, 1)
    for label, count in summary.items():
        pdf.cell(0, 6, f"  - {label.capitalize()}: {count}", 0, 1)
    pdf.ln(5)

    # 2. Defect Table
    pdf.add_chapter_title('Defect Details')
    pdf.add_defect_table(defects)
    pdf.ln(5)

    # 3. Annotated Image
    pdf.add_chapter_title('Annotated Image')
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
    pdf.add_image_from_pil(annotated_pil, "Final Annotated Result")
    pdf.ln(5)

    # 4. Charts
    pdf.add_page()
    pdf.add_chapter_title('Visualizations')
    # Place charts side-by-side
    page_width = pdf.epw / 2 - 5
    pdf.add_image_from_fig(bar_fig, "Defect Count per Class", w=page_width)
    y_after_bar = pdf.get_y() # Save Y position
    pdf.set_y(y_after_bar - (bar_fig.get_figheight() * 25.4) - 12) # Move Y back up
    pdf.set_x(page_width + 15)
    pdf.add_image_from_fig(pie_fig, "Defect Class Distribution", w=page_width)
    pdf.set_y(y_after_bar) # Move Y down past the tallest chart

    pdf.add_image_from_fig(scatter_fig, "Defect Scatter Plot")
    plt.close('all') # Close all figs to save memory

    # 5. Input Images
    pdf.add_page()
    pdf.add_chapter_title('Input Images')
    pdf.add_image_from_pil(template_pil, "Template Image", w=page_width)
    y_after_template = pdf.get_y()
    pdf.set_y(y_after_template - (template_pil.height * page_width / template_pil.width * 0.264583) - 12) # Move Y back up
    pdf.set_x(page_width + 15)
    pdf.add_image_from_pil(test_pil, "Test Image", w=page_width)

    # Return as bytes
    return pdf.output(dest='S').encode('latin-1')