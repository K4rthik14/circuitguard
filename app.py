# app.py (Main Flask application)
from flask import Flask, render_template, send_from_directory
from controllers.detection_routes import detection_bp # Import the Blueprint
import os
import logging # Added for logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
# template_folder='templates' tells Flask where to find index.html
# static_folder='static' tells Flask where to find CSS, JS files
app = Flask(__name__, template_folder='templates', static_folder='static')

# Register the Blueprint from controllers/detection_routes.py
# All routes defined in detection_bp will now be accessible under the '/api' prefix
# e.g., the '/detect' route becomes '/api/detect'
app.register_blueprint(detection_bp) # url_prefix is now set in the Blueprint itself

# --- Route for the main HTML page ---
@app.route('/')
def index():
    """Serves the main HTML page (index.html) from the templates folder."""
    logging.info("Serving index page.")
    # Check if index.html exists to provide a helpful error if it's missing
    if not os.path.exists(os.path.join(app.template_folder, 'index.html')):
        logging.error("index.html not found in templates folder.")
        return "Error: index.html not found in templates folder.", 404
    # render_template looks inside the 'templates' folder by default
    return render_template('index.html')

# --- Routes for static files (CSS, JS) ---
# Flask automatically handles serving files from the 'static' folder
# if you use url_for() correctly in your HTML, as shown in the updated index.html.
# This explicit route is usually not needed but can be helpful for debugging.
# @app.route('/static/<path:filename>')
# def serve_static(filename):
#     return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # Run the Flask development server
    # host='0.0.0.0' makes the server accessible on your local network
    # port=5000 is the default Flask port
    # debug=True enables auto-reloading when code changes (DISABLE FOR PRODUCTION)
    logging.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True)