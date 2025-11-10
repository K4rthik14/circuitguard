# CircuitGuard: AI-Powered PCB Defect Detection

CircuitGuard is a full-stack web application designed for automated detection and classification of defects on Printed Circuit Boards (PCBs). It uses computer vision to compare a "test" image against a "template" image and employs a deep learning model to classify any found anomalies.

The application provides a web-based user interface for uploading images, visualizing results, and generating detailed PDF reports.

### ✨ Features

* **AI-Powered Classification:** Uses a pre-trained **EfficientNet-B4** model (98% accuracy) to classify 6 types of PCB defects (short, spur, open, copper, mousebite, pin-hole).
* **Web Interface:** A clean, responsive UI built with HTML, CSS, and JavaScript for uploading images and viewing results.
* **Adjustable Parameters:** Users can fine-tune the detection pipeline by adjusting the Difference Threshold, Minimum Defect Area, and Noise Filter Strength.
* **Rich Result Visualization:** Displays the final annotated image, intermediate processing steps (Difference & Mask), and three summary charts (Bar, Pie, Scatter).
* **Professional PDF Reporting:** Dynamically generates a multi-page PDF report on the backend, including all images, charts, and a detailed defect summary.
* **Data Export:** Allows users to download the annotated image, the CSV log of defects, and the full PDF report.

---

### 🛠️ Tech Stack & Architecture

This project uses a modular client-server architecture.



* **Backend:**
    * **Framework:** **Flask** (Python)
    * **AI/Deep Learning:** **PyTorch** & **Timm** (for EfficientNet-B4)
    * **Computer Vision:** **OpenCV** (for image subtraction, thresholding, contours)
    * **Graphing:** **Matplotlib** (for server-side chart generation)
    * **PDF Generation:** **fpdf2** (for professional, backend-generated reports)

* **Frontend:**
    * **Structure:** **HTML5**
    * **Styling:** **CSS3**
    * **Interactivity:** **JavaScript (ES6+)** (using `fetch` for API calls)

---

### 📁 Project Structure