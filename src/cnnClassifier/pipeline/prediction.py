# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: standard libraries
# ────────────────────────────────────────────────────────────────────────────────────────
import os
import numpy      as np
import tensorflow as tf

from   tensorflow.keras.preprocessing import image
from   pathlib                        import Path

from   cnnClassifier.utils.common     import read_yaml
        
# ────────────────────────────────────────────────────────────────────────────────────────
# PredictionPipeline Class: Handles model inference on input image
# ────────────────────────────────────────────────────────────────────────────────────────
class PredictionPipeline:
    def __init__(self, filename, model=None):
        """
        Initializes the prediction pipeline with the input image filename and model.

        Args:
            filename (str)         : Path to the image file to be classified.
            model (tf.keras.Model) : Preloaded model instance (optional).
        """
        self.filename = filename
        self.model    = model or self._load_default_model()
        
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot perform prediction.")


    def _load_default_model(self):
        """
        Fallback method to load model from disk if not provided externally.
        """
        raise RuntimeError("No model provided and fallback is disabled in production.")



    # ────────────────────────────────────────────────────────────────────────────────────────
    # Predict Method: Loads model, preprocesses image, performs inference
    # ────────────────────────────────────────────────────────────────────────────────────────
    def predict(self):
        """
        Executes the prediction workflow:
        - Preprocesses input image to match model input shape
        - Performs inference and returns class label

        Returns:
            list[dict]: Prediction result wrapped in a dictionary for downstream use
        """
        
        # Load image size from params.yaml
        params     = read_yaml(Path("params.yaml"))                               # Read the parms file
        image_size = tuple(params["IMAGE_SIZE"][:2])                              # Extract (height, width)
        
        # Load and preprocess input image
        test_image = image.load_img     (self.filename, target_size=image_size)   # Resize to model input
        test_image = image.img_to_array (test_image)                              # Convert to NumPy array
        test_image = np.expand_dims     (test_image, axis=0)                      # Add batch dimension

        # Perform prediction and extract class index
        result     = np.argmax(self.model.predict(test_image), axis=1)
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
