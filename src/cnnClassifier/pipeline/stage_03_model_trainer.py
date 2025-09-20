# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Configuration Manager, Training Component, and Logger
# ────────────────────────────────────────────────────────────────────────────────────────
from cnnClassifier                          import logger                # Centralized logger
from cnnClassifier.config.configuration     import ConfigurationManager  # Loads config entities
from cnnClassifier.components.model_trainer import Training              # Training logic


# ────────────────────────────────────────────────────────────────────────────────────────
# Stage Identifier for Logging and Traceability
# ────────────────────────────────────────────────────────────────────────────────────────
STAGE_NAME = "Training"

# ────────────────────────────────────────────────────────────────────────────────────────
# Pipeline Class: Orchestrates Model Training Workflow
# ────────────────────────────────────────────────────────────────────────────────────────
class ModelTrainingPipeline:
    def __init__(self):
        """
        Initializes the pipeline class.
        No state is maintained here—execution is handled in `main()`.
        """
        pass

    def main(self):
        """
        Executes the model training workflow:
        - Loads training configuration
        - Loads updated base model from disk
        - Prepares training and validation data generators
        - Trains the model and saves final weights
        """
        config          = ConfigurationManager()
        training_config = config.get_training_config()
        training        = Training(config=training_config)
                                 #(config=ConfigurationManager().get_training_config())
        training.get_base_model()
        training.train_valid_generator()
        training.train()

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry Point: Executes Pipeline with Logging and Exception Handling
# ────────────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> STAGE : {STAGE_NAME} started   <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> STAGE : {STAGE_NAME} completed <<<<<<\n{'x'*50}")
    except Exception as e:
        logger.exception(e)  # Logs full traceback for debugging
        raise e              # Propagates error for upstream visibility