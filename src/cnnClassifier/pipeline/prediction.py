# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: standard libraries
# ────────────────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
from   tensorflow.keras.models        import load_model
from   tensorflow.keras.preprocessing import image


# ────────────────────────────────────────────────────────────────────────────────────────
# PredictionPipeline Class: Handles model inference on input image
# ────────────────────────────────────────────────────────────────────────────────────────
class PredictionPipeline:
    def __init__(self, filename):
        """
        Initializes the prediction pipeline with the input image filename.

        Args:
            filename (str): Path to the image file to be classified.
        """
        self.filename = filename

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Predict Method: Loads model, preprocesses image, performs inference
    # ────────────────────────────────────────────────────────────────────────────────────────
    def predict(self):
        """
        Executes the prediction workflow:
        - Loads trained model from disk
        - Preprocesses input image to match model input shape
        - Performs inference and returns class label

        Returns:
            list[dict]: Prediction result wrapped in a dictionary for downstream use
        """
        # Load trained model from disk

        # model      = load_model(os.path.join("model", "model.h5"))                  # Active path for inference
        model      = load_model(os.path.join("artifacts", "training", "model.h5"))  # Loading trained & saved model

        # Load and preprocess input image
        imagename  = self.filename
        test_image = image.load_img     (imagename, target_size=(224, 224))   # Resize to model input
        test_image = image.img_to_array (test_image)                          # Convert to NumPy array
        test_image = np.expand_dims     (test_image, axis=0)                  # Add batch dimension

        # Perform prediction and extract class index
        result     = np.argmax(model.predict(test_image), axis=1)
        print(result)        # Optional: log raw prediction index

        # Map prediction index to human-readable label
        class_map = {
                        0 : "adeno_carcinoma",
                        1 : "large_cell_carcinoma",
                        2 : "normal",
                        3 : "squamous_cell_carcinoma"
                    }

        # class_map = {
        #                 0: "colon_adenocarcinoma",
        #                 1: "colon_normal",
        #                 2: "lung_adenocarcinoma",
        #                 3: "lung_normal",
        #                 4: "lung_squamous_cell_carcinoma"
        #             }                  # for LC25000 data


        prediction = class_map.get(result[0], "Unknown")
        return [{"image" : prediction}]
