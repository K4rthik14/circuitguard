# app.py (Main Flask application)
from flask import Flask, render_template, send_from_directory
from controllers.detection_routes import detection_bp # Import the Blueprint
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Register the Blueprint (already includes '/api' prefix from detection_routes.py)
app.register_blueprint(detection_bp)

# --- Route for the main HTML page ---
@app.route('/')
def index():
    """Serves the main HTML page (index.html)."""
    logging.info("Serving index page.")
    if not os.path.exists(os.path.join(app.template_folder, 'index.html')):
        logging.error("index.html not found in templates folder.")
        return "Error: index.html not found in templates folder.", 404
    return render_template('index.html')

if __name__ == '__main__':
    logging.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True)