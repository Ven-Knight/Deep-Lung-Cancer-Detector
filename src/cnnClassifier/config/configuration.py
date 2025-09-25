# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Constants, Utilities, and Config Entities
# ────────────────────────────────────────────────────────────────────────────────────────
import os

from cnnClassifier.constants            import *                              # Centralized constant paths
from cnnClassifier.utils.common         import read_yaml, create_directories  # Utility functions
from cnnClassifier.entity.config_entity import ( DataIngestionConfig,
                                                 PrepareBaseModelConfig,
                                                 TrainingConfig,
                                                 EvaluationConfig
                                               )                              # Typed config dataclasses

# ────────────────────────────────────────────────────────────────────────────────────────
# Configuration Manager: Loads and Structures Pipeline Configs
# ────────────────────────────────────────────────────────────────────────────────────────
class ConfigurationManager:
    def __init__(
                    self,
                    config_filepath = CONFIG_FILE_PATH,                       # Path to config.yaml
                    params_filepath = PARAMS_FILE_PATH                        # Path to params.yaml
                ):
        # Load configuration and hyperparameters from YAML files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Create root artifact directory for storing pipeline outputs
        create_directories([self.config.artifacts_root])

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Data Ingestion Config: Setup for downloading and extracting dataset
    # ────────────────────────────────────────────────────────────────────────────────────────
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        # Create ingestion-specific directory
        create_directories([config.root_dir])

        # Return structured config object for ingestion stage
        data_ingestion_config = DataIngestionConfig(
                                                     root_dir        = config.root_dir,
                                                     source_URL      = config.source_URL,
                                                     local_data_file = config.local_data_file,
                                                     unzip_dir       = config.unzip_dir
                                                   )
        return data_ingestion_config

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Base Model Preparation Config: Setup for loading and customizing pretrained model
    # ────────────────────────────────────────────────────────────────────────────────────────
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        # Create model preparation directory
        create_directories([config.root_dir])

        # Return structured config object for base model setup
        prepare_base_model_config = PrepareBaseModelConfig(
                                                    base_model_type         = config.base_model_type,
                                                    root_dir                = Path(config.root_dir),
                                                    base_model_path         = Path(config.base_model_path),
                                                    updated_base_model_path = Path(config.updated_base_model_path),
                                                    params_image_size       = self.params.IMAGE_SIZE,
                                                    params_learning_rate    = self.params.LEARNING_RATE_HEAD,
                                                    params_include_top      = self.params.INCLUDE_TOP,
                                                    params_weights          = self.params.WEIGHTS,
                                                    params_classes          = self.params.CLASSES,
                                                    params_freeze_all       = self.params.FREEZE_ALL,
                                                    params_freeze_till      = self.params.FREEZE_TILL
                                                          )
        return prepare_base_model_config

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Training Config: Setup for model training parameters and paths
    # ────────────────────────────────────────────────────────────────────────────────────────
    def get_training_config(self) -> TrainingConfig:
        
        training           = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params             = self.params

        # Construct training data path from ingestion output
        training_data      = os.path.join(self.config.data_ingestion.unzip_dir, self.config.data_ingestion.source_dir_name, "Train_and_Validation_Set")

        # Create training-specific directory
        create_directories([Path(training.root_dir)])

        # Return structured config object for training stage
        training_config    = TrainingConfig(
                                                    root_dir                   = Path(training.root_dir),
                                                    trained_model_path         = Path(training.trained_model_path),
                                                    model_export_path          = Path(training.model_export_path),
                                                    updated_base_model_path    = Path(prepare_base_model.updated_base_model_path),
                                                    training_data              = Path(training_data),
                                                    params_batch_size          = params.BATCH_SIZE,
                                                    params_is_augmentation     = params.AUGMENTATION,
                                                    params_image_size          = params.IMAGE_SIZE,

                                                    # New fields for VGG16 fine-tuning
                                                    params_num_classes         = params.CLASSES,
                                                    params_epochs_head         = params.EPOCHS_HEAD,
                                                    params_epochs_fine         = params.EPOCHS_FINE,
                                                    params_learning_rate_head  = params.LEARNING_RATE_HEAD,
                                                    params_learning_rate_fine  = params.LEARNING_RATE_FINE,
                                                    params_freeze_all          = params.FREEZE_ALL,
                                                    params_freeze_till         = params.FREEZE_TILL
                                           )
        return training_config


    # ────────────────────────────────────────────────────────────────────────────────────────
    # Evaluation Config: Setup for model evaluation and MLflow logging
    # ────────────────────────────────────────────────────────────────────────────────────────
    def get_evaluation_config(self) -> EvaluationConfig:
        # Return structured config object for evaluation stage

        # Construct testing data path from ingestion output
        testing_data  = os.path.join(self.config.data_ingestion.unzip_dir, self.config.data_ingestion.source_dir_name, "Test_Set")
        
        eval_config   = EvaluationConfig(
                         path_of_model         = "artifacts/training/model.h5",                  # Path to trained model
                         test_data             = Path(testing_data),
                         mlflow_uri            = os.environ.get("MLFLOW_TRACKING_URI"),          # MLflow tracking URI
                         all_params            = self.params,                                    # Full parameter dictionary
                         params_image_size     = self.params.IMAGE_SIZE,
                         params_batch_size     = self.params.BATCH_SIZE,
                         experiment_name       = self.config.mlflow.experiment_name,                         
                         registered_model_name = self.config.mlflow.registered_model_name

                                      )
        return eval_config
      