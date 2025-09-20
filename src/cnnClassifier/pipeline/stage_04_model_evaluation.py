# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Configuration Manager, Evaluation Component, and Logger
# ────────────────────────────────────────────────────────────────────────────────────────
from cnnClassifier                                    import logger                # Centralized logger instance
from cnnClassifier.config.configuration               import ConfigurationManager  # Loads config entities
from cnnClassifier.components.model_evaluation_mlflow import Evaluation            # Evaluation logic


# ────────────────────────────────────────────────────────────────────────────────────────
# Stage Identifier for Logging and Traceability
# ────────────────────────────────────────────────────────────────────────────────────────
STAGE_NAME = "Evaluation"

# ────────────────────────────────────────────────────────────────────────────────────────
# Pipeline Class: Orchestrates Model Evaluation Workflow
# ────────────────────────────────────────────────────────────────────────────────────────
class EvaluationPipeline:
    def __init__(self):
        """
        Initializes the pipeline class.
        No state is maintained here—execution is handled in `main()`.
        """
        pass

    def main(self):
        """
        Executes the model evaluation workflow:
        - Loads evaluation configuration
        - Loads trained model from disk
        - Prepares validation data generator
        - Evaluates model performance
        - Saves evaluation metrics locally
        - (Optional) Logs metrics and model into MLflow
        """
        config      = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation  = Evaluation(eval_config)
                               #(ConfigurationManager().get_evaluation_config())
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()  # Uncomment to enable MLflow logging

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry Point: Executes Pipeline with Logging and Exception Handling
# ────────────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> STAGE : {STAGE_NAME} started   <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> STAGE : {STAGE_NAME} completed <<<<<<\n{'x'*50}")
    except Exception as e:
        logger.exception(e)  # Logs full traceback for debugging
        raise e              # Propagates error for upstream visibility