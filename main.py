# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Logger and Stage-Specific Pipeline Classes
# ────────────────────────────────────────────────────────────────────────────────────────
from cnnClassifier                                      import logger  # Centralized logger instance
from cnnClassifier.pipeline.stage_01_data_ingestion     import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_trainer      import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation   import EvaluationPipeline

# ────────────────────────────────────────────────────────────────────────────────────────
# STAGE 01: Data Ingestion
# ────────────────────────────────────────────────────────────────────────────────────────
STAGE_NAME = "STAGE 01: Data Ingestion    "
try:
    logger.info("\n" + "*" * 90)
    logger.info(f"\n\t\t\t>>>>>> {STAGE_NAME} started   <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"\n\t\t\t>>>>>> {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)  # Logs full traceback for debugging
    raise e              # Propagates error for upstream visibility

# ────────────────────────────────────────────────────────────────────────────────────────
# STAGE 02: Prepare Base Model
# ────────────────────────────────────────────────────────────────────────────────────────
STAGE_NAME = "STAGE 02: Prepare Base Model"
try:
    logger.info("\n" + "*" * 90)
    logger.info(f"\n\t\t\t>>>>>> {STAGE_NAME} started   <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f"\n\t\t\t>>>>>> {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

# ────────────────────────────────────────────────────────────────────────────────────────
# STAGE 03: Model Training
# ────────────────────────────────────────────────────────────────────────────────────────
STAGE_NAME = "STAGE 03: Model Training    "
try:
    logger.info("\n" + "*" * 90)
    logger.info(f"\n\t\t\t>>>>>> {STAGE_NAME} started   <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f"\n\t\t\t>>>>>> {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

# ────────────────────────────────────────────────────────────────────────────────────────
# STAGE 04: Model Evaluation
# ────────────────────────────────────────────────────────────────────────────────────────
STAGE_NAME = "STAGE 04: Model Evaluation  "
try:
    logger.info("\n" + "*" * 90)
    logger.info(f"\n\t\t\t>>>>>> {STAGE_NAME} started   <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    model_evaluation = EvaluationPipeline()
    model_evaluation.main()
    logger.info(f"\n\t\t\t>>>>>> {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e