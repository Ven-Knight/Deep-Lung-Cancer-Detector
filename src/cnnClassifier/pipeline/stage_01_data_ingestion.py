# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Configuration Manager, Component Logic, and Logger
# ────────────────────────────────────────────────────────────────────────────────────────
from cnnClassifier                           import logger                # Centralized logger
from cnnClassifier.config.configuration      import ConfigurationManager  # Loads config entities
from cnnClassifier.components.data_ingestion import DataIngestion         # Ingestion logic


# ────────────────────────────────────────────────────────────────────────────────────────
# Stage Identifier for Logging and Traceability
# ────────────────────────────────────────────────────────────────────────────────────────
STAGE_NAME = "STAGE 01: Data Ingestion    "

# ────────────────────────────────────────────────────────────────────────────────────────
# Pipeline Class: Orchestrates the Data Ingestion Workflow
# ────────────────────────────────────────────────────────────────────────────────────────
class DataIngestionTrainingPipeline:
    def __init__(self):
        """
        Initializes the pipeline class.
        No state is maintained here—execution is handled in `main()`.
        """
        pass

    def main(self):
        """
        Executes the data ingestion workflow:
        - Loads config
        - Downloads dataset from remote source
        - Extracts contents to target directory
        """
        config                = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion        = DataIngestion(config=data_ingestion_config)
                                            #(config=ConfigurationManager().get_data_ingestion_config())
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

# ────────────────────────────────────────────────────────────────────────────────────────
# Entry Point: Executes Pipeline with Logging and Exception Handling
# ────────────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        logger.info("\n" + "*" * 90)
        logger.info(f"\n\t\t\t>>>>>> {STAGE_NAME} started   <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"\n\t\t\t>>>>>> {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)  # Logs full traceback for debugging
        raise e              # Propagates error for upstream visibility
