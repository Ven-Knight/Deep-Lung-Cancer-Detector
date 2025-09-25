# ────────────────────────────────────────────────────────────────────────────────────────
# Typed Configuration Entities : Configuration Entities define immutable, structured objects 
# that encapsulate stage-specific parameters and paths for reproducible pipeline execution.
# ────────────────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass          # For immutable, structured config objects
from pathlib     import Path               # OS-agnostic path handling

# ────────────────────────────────────────────────────────────────────────────────────────
# Configuration Entity: Data Ingestion Stage
# ────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir                   : Path      # Root directory for data ingestion artifacts
    source_URL                 : str       # Remote URL to download raw dataset
    local_data_file            : Path      # Path to store downloaded file locally
    unzip_dir                  : Path      # Directory to extract and organize raw data

# ────────────────────────────────────────────────────────────────────────────────────────
# Configuration Entity: Base Model Preparation Stage
# ────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    base_model_type            : str       # Architecture type (e.g., 'vgg16', 'resnet50')
    root_dir                   : Path      # Directory to store base model artifacts
    base_model_path            : Path      # Path to original pre-trained model
    updated_base_model_path    : Path      # Path to updated model with custom layers
    params_image_size          : list      # Input image dimensions [height, width, channels]
    params_learning_rate       : float     # Learning rate for fine-tuning
    params_include_top         : bool      # Whether to include top layers of base model
    params_weights             : str       # Pre-trained weights source (e.g., 'imagenet')
    params_classes             : int       # Number of output classes
    params_freeze_all          : bool      # If True, freezes all layers of base model during initial training 
    params_freeze_till         : int       # Number of layers (from the end) to keep trainable during fine-tuning


# ────────────────────────────────────────────────────────────────────────────────────────
# Configuration Entity: Model Training Stage
# ────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TrainingConfig:
    root_dir                   : Path      # Directory to store training outputs
    trained_model_path         : Path      # Path to save final trained model
    model_export_path          : Path      # Path to save final trained model for docker visibility
    updated_base_model_path    : Path      # Path to fine-tuned base model
    training_data              : Path      # Path to training dataset
    params_batch_size          : int       # Batch size for training
    params_is_augmentation     : bool      # Flag to enable/disable data augmentation
    params_image_size          : list      # Input image dimensions [height, width, channels]

    # New fields for VGG16 fine-tuning
    params_num_classes         : int       # Number of output classes
    params_epochs_head         : int       # Epochs for training the classification head
    params_epochs_fine         : int       # Epochs for fine-tuning top layers
    params_learning_rate_head  : float     # Learning rate for head training
    params_learning_rate_fine  : float     # Learning rate for fine-tuning
    params_freeze_all          : bool      # Whether to freeze all layers initially
    params_freeze_till         : int       # Number of layers to unfreeze from the end

# ────────────────────────────────────────────────────────────────────────────────────────
# Configuration Entity: Model Evaluation Stage
# ────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model              : Path      # Path to trained model for evaluation
    test_data                  : Path      # Path of the test data
    all_params                 : dict      # Dictionary of all hyperparameters used
    mlflow_uri                 : str       # MLflow tracking URI for logging metrics
    params_image_size          : list      # Input image dimensions [height, width, channels]
    params_batch_size          : int       # Batch size for evaluation
    experiment_name            : str       # experiment name to set in mlflow
    registered_model_name      : str       # final model name to set in mlflow model registry
