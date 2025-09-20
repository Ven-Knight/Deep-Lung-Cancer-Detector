# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Standard Libraries
# ────────────────────────────────────────────────────────────────────────────────────────
import tensorflow   as tf
import mlflow
import mlflow.keras
from   pathlib      import Path
from   urllib.parse import urlparse

# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Project Utilities
# ────────────────────────────────────────────────────────────────────────────────────────
from cnnClassifier.entity.config_entity import EvaluationConfig                          # Typed config object
from cnnClassifier.utils.common         import read_yaml, create_directories, save_json  # Utility functions

# ────────────────────────────────────────────────────────────────────────────────────────
# Evaluation Class: Handles model loading, validation, scoring, and MLflow logging
# ────────────────────────────────────────────────────────────────────────────────────────
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        """
        Initialize with structured config containing paths, parameters, and MLflow URI.

        Args:
            config (EvaluationConfig): Configuration entity for evaluation stage.
        """
        self.config = config

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Setup Validation Data Generator
    # ────────────────────────────────────────────────────────────────────────────────────────
    def _valid_generator(self):
        """
        Creates a validation data generator using ImageDataGenerator.
        Applies rescaling and splits data for evaluation.
        """
        datagenerator_kwargs = dict(
                                        rescale          = 1./255,
                                        validation_split = 0.20
                                   )

        dataflow_kwargs      = dict(
                                        target_size   = self.config.params_image_size[:-1],  # Exclude channel dimension
                                        batch_size    = self.config.params_batch_size,
                                        interpolation = "bilinear"
                                   )

        valid_datagenerator  = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.valid_generator = valid_datagenerator.flow_from_directory(
                                                                           directory = self.config.training_data,
                                                                           subset    = "validation",
                                                                           shuffle   = False,
                                                                           **dataflow_kwargs
                                                                      )

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Load Trained Model from Disk
    # ────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Loads a trained Keras model from the specified path.

        Args:
            path (Path): Path to the saved model (.h5 or SavedModel format)

        Returns:
            tf.keras.Model: Loaded model instance
        """
        return tf.keras.models.load_model(path)


    # ────────────────────────────────────────────────────────────────────────────────────────
    # Save Evaluation Metrics Locally
    # ────────────────────────────────────────────────────────────────────────────────────────
    def save_score(self):
        """
        Saves evaluation metrics (loss and accuracy) to a local JSON file.
        """
        scores = {"loss" : self.score[0], "accuracy" : self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Evaluate Model on Validation Data
    # ────────────────────────────────────────────────────────────────────────────────────────
    def evaluation(self):
        """
        Loads the model, prepares validation data, evaluates performance,
        and saves the score locally.
        """
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    
    # ────────────────────────────────────────────────────────────────────────────────────────
    # Log Metrics and Model into MLflow
    # ────────────────────────────────────────────────────────────────────────────────────────
    def log_into_mlflow(self):
        """
        Logs evaluation metrics and model artifacts into MLflow.
        Registers model if remote tracking URI is used.
        """
        # Set remote MLflow tracking URI (hosted on EC2)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)

        # Ensure experiment exists or create it
        experiment_name = self.config.experiment_name

        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)

        # Determine backend store type
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(self.config.all_params)

            # Log evaluation metrics
            mlflow.log_metrics({
                                    "loss"     : self.score[0],
                                    "accuracy" : self.score[1]
                               })

            # Log model to S3 (via MLflow)
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                                        self.model,
                                        artifact_path         = "model",
                                        registered_model_name = self.config.registered_model_name
                                      )
            else:
                mlflow.keras.log_model(self.model, artifact_path="model")
