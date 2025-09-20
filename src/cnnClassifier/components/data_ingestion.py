# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Standard Libraries
# ────────────────────────────────────────────────────────────────────────────────────────
import os
import zipfile
import gdown
# ────────────────────────────────────────────────────────────────────────────────────────
# Imports: Project Modules
# ────────────────────────────────────────────────────────────────────────────────────────
from cnnClassifier                      import logger               # Centralized logger instance
from cnnClassifier.utils.common         import get_size             # Utility to get file size
from cnnClassifier.entity.config_entity import DataIngestionConfig  # Typed config object

# ────────────────────────────────────────────────────────────────────────────────────────
# DataIngestion Class: Handles downloading and extracting dataset
# ────────────────────────────────────────────────────────────────────────────────────────
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize with a structured config object containing paths and URLs.

        Args:
            config (DataIngestionConfig): Configuration entity for ingestion stage.
        """
        self.config = config

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Download File: Fetches dataset from Google Drive using gdown
    # ────────────────────────────────────────────────────────────────────────────────────────
    def download_file(self) -> str:
        """
        Downloads the dataset zip file from the configured Google Drive URL.

        Returns:
            str: Path to the downloaded zip file.
        """
        try:
            dataset_url      = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            # Ensure ingestion directory exists
            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            # Extract file ID from Google Drive URL and construct direct download link
            file_id          = dataset_url.split("/")[-2]
            prefix           = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(f"Downloading Completed")
            return str(zip_download_dir)

        except Exception as e:
            raise e                                

    # ────────────────────────────────────────────────────────────────────────────────────────
    # Extract Zip File: Unzips the downloaded dataset into target directory
    # ────────────────────────────────────────────────────────────────────────────────────────
    def extract_zip_file(self):
        """
        Extracts the downloaded zip file into the configured directory.

        Returns:
            None
        """
        unzip_path = self.config.unzip_dir

        # Ensure target directory exists
        os.makedirs(unzip_path, exist_ok=True)

        # Extract contents of the zip file
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        logger.info(f"Extracted zip file to directory: {unzip_path}")

