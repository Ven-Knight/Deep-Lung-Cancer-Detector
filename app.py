# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Core Flask Modules, CORS Handling, and Internal Pipeline Utilities
# ────────────────────────────────────────────────────────────────────────────────────────
import os                                                                                 # Environment variable setup
from flask                             import Flask, request, jsonify, render_template    # Flask app and API routing
from flask_cors                        import CORS, cross_origin                          # Enable cross-origin requests
from cnnClassifier.utils.common        import decodeImage                                 # Base64 image decoding utility
from cnnClassifier.pipeline.prediction import PredictionPipeline                          # Prediction pipeline wrapper


# ────────────────────────────────────────────────────────────────────────────────────────
# Environment Locale Setup: Ensures UTF-8 Compatibility for Image and Log Handling
# ────────────────────────────────────────────────────────────────────────────────────────
os.putenv('LANG',   'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


# ────────────────────────────────────────────────────────────────────────────────────────
# Flask App Initialization and CORS Configuration
# ────────────────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


# ────────────────────────────────────────────────────────────────────────────────────────
# ClientApp Wrapper: Holds Filename and Prediction Pipeline Instance
# ────────────────────────────────────────────────────────────────────────────────────────
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"                          # Default filename for incoming image
        self.classifier = PredictionPipeline(self.filename)       # Initialize prediction pipeline


# ────────────────────────────────────────────────────────────────────────────────────────
# Route: Home Page - Serves Frontend UI from Jinja2 Template
# ────────────────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')                          # Assumes templates/index.html exists


# ────────────────────────────────────────────────────────────────────────────────────────
# Route: Training Trigger - Executes Full Pipeline via main.py
# ────────────────────────────────────────────────────────────────────────────────────────
@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")                                   # Executes pipeline (can replace with subprocess)
    # os.system("dvc repro")                                      # Optional: reproducible pipeline via DVC
    return "Training done successfully!"


# ────────────────────────────────────────────────────────────────────────────────────────
# Route: Prediction API - Accepts Base64 Image and Returns Classification Result
# ────────────────────────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']                                 # Expect base64-encoded image in JSON
    decodeImage(image, clApp.filename)                            # Decode and save image to disk
    result = clApp.classifier.predict()                           # Run prediction pipeline
    return jsonify(result)                                        # Return result as JSON response


# ────────────────────────────────────────────────────────────────────────────────────────
# Entry Point: Launch Flask App on Port 8080 (AWS-Compatible Host Binding)
# ────────────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clApp = ClientApp()                                           # Instantiate prediction wrapper
    app.run(host='0.0.0.0', port=8080)                            # Run app on all interfaces