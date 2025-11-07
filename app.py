# app.py - Clean Flask bootstrap
from flask import Flask, render_template
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_app() -> Flask:
    app = Flask(__name__, template_folder='templates', static_folder='static')

    # Defer import to avoid circulars
    from controllers.detection_routes import detection_bp
    app.register_blueprint(detection_bp)

    @app.route('/')
    def index():
        if not os.path.exists(os.path.join(app.template_folder, 'index.html')):
            return "index.html missing", 500
        return render_template('index.html')
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
