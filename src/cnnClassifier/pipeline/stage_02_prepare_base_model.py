# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Configuration Manager, Component Logic, and Logger
# ────────────────────────────────────────────────────────────────────────────────────────
from cnnClassifier                               import logger                # Centralized logger instance
from cnnClassifier.config.configuration          import ConfigurationManager  # Loads config entities
from cnnClassifier.components.prepare_base_model import PrepareBaseModel      # Base model logic


# ────────────────────────────────────────────────────────────────────────────────────────
# Stage Identifier for Logging and Traceability
# ────────────────────────────────────────────────────────────────────────────────────────
STAGE_NAME = "Prepare_Base_Model"

# ────────────────────────────────────────────────────────────────────────────────────────
# Pipeline Class: Orchestrates Base Model Preparation Workflow
# ────────────────────────────────────────────────────────────────────────────────────────
class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        """
        Initializes the pipeline class.
        No state is maintained here—execution is handled in `main()`.
        """
        pass

    def main(self):
        """
        Executes the base model preparation workflow:
        - Loads configuration
        - Downloads and loads pretrained base model
        - Applies custom layers and compiles updated model
        - Saves both base and updated models to disk
        """
        config                    = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model        = PrepareBaseModel(config=prepare_base_model_config)
                                                   #(config=ConfigurationManager().get_prepare_base_model_config())
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry Point: Executes Pipeline with Logging and Exception Handling
# ────────────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> STAGE : {STAGE_NAME} started   <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> STAGE : {STAGE_NAME} completed <<<<<<\n{'x'*50}")
    except Exception as e:
        logger.exception(e)  # Logs full traceback for debugging
        raise e              # Propagates error for upstream visibility