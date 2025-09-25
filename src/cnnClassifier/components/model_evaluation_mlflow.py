# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Standard Libraries
# ────────────────────────────────────────────────────────────────────────────────────────
import os
import mlflow
import mlflow.keras
import tensorflow        as tf
import matplotlib.pyplot as plt
import seaborn           as sns

from   sklearn.metrics import confusion_matrix, classification_report
from   pathlib         import Path
from   urllib.parse    import urlparse
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
    def _test_generator(self):
        """Creates a test data generator using ImageDataGenerator (no split)."""
        datagenerator_kwargs = dict(rescale=1./255)

        dataflow_kwargs      = dict(
                                        target_size   = self.config.params_image_size[:-1],    # Exclude channel dimension
                                        batch_size    = self.config.params_batch_size,
                                        interpolation = "bilinear"
                                   )

        test_datagenerator   = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.test_generator  = test_datagenerator.flow_from_directory(
                                                                        directory = self.config.test_data,
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
    # confusion matrics creation
    # ────────────────────────────────────────────────────────────────────────────────────────
    def log_confusion_matrix(self, y_true, y_pred, dataset_name="Test Data"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_file_path = f'confusion_matrix_{dataset_name.replace(" ", "_")}.png'
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        plt.close()

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Save Evaluation Metrics Locally
    # ────────────────────────────────────────────────────────────────────────────────────────
    def save_score(self):
        """Saves loss, accuracy, and class-wise metrics from classification report to scores.json."""
        # Start with loss and accuracy (evaluation metrics)
        scores = {
                    "loss"     : float(self.score[0]),
                    "accuracy" : float(self.score[1])
                 }

        # Add class-wise metrics
        cr = classification_report(self.y_true, self.y_pred_classes, output_dict=True)
        for label, metrics in cr.items():
            clean_label = label.replace(" avg", "")
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    scores[f"{clean_label}_{metric}"] = float(value)
            else:
                scores[clean_label] = float(metrics)

        # Save to scores.json
        save_json(path=Path("scores.json"), data=scores)

        # store for MLflow logging
        self.metric_store = scores
    
    # ────────────────────────────────────────────────────────────────────────────────────────
    # Evaluate Model on Validation Data
    # ────────────────────────────────────────────────────────────────────────────────────────
    def evaluation(self):
        """
        Loads the model, prepares test data, evaluates performance,
        and saves the score locally.
        """
        self.model          = self.load_model(self.config.path_of_model)
        self._test_generator()
        self.score          = self.model.evaluate(self.test_generator)
        self.y_pred         = self.model.predict(self.test_generator)
        self.y_pred_classes = self.y_pred.argmax(axis=1)
        self.y_true         = self.test_generator.classes

        self.save_score()

    
    # ────────────────────────────────────────────────────────────────────────────────────────
    # Log Metrics and Model into MLflow
    # ────────────────────────────────────────────────────────────────────────────────────────
    def log_into_mlflow(self):
        """
        Logs evaluation metrics and model artifacts into MLflow.
        Registers model if remote tracking URI is used.
        """
        
        # Set remote MLflow tracking URI (hosted on EC2) in secured way
        from dotenv import load_dotenv
        load_dotenv()
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")

        
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_registry_uri(mlflow_uri)

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
            
            # Log all evaluation metrics from scores.json
            mlflow.log_metrics(self.metric_store)

            # Log confusion matrix
            self.log_confusion_matrix(self.y_true, self.y_pred_classes)

            # Log scores.json as an artifact
            mlflow.log_artifact("scores.json")

            # Log model to S3 (via MLflow)
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                                        self.model,
                                        artifact_path         = "model",
                                        registered_model_name = self.config.registered_model_name
                                      )
            else:
                mlflow.keras.log_model(self.model, artifact_path="model")
