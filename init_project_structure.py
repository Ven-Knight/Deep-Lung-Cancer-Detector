import os
import logging
from   pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ────────────────────────────────────────────────────────────────────────────────────────

# Configure logging to include timestamps for auditability
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# ────────────────────────────────────────────────────────────────────────────────────────
# Project Configuration
# ────────────────────────────────────────────────────────────────────────────────────────

# Define the root module name 
project_name  = "cnnClassifier"

# List of essential files and directories to scaffold the project structure
list_of_files = [
    ".github/workflows/.gitkeep",                   # Placeholder for GitHub Actions workflows
    f"src/{project_name}/__init__.py",              # Root module init
    f"src/{project_name}/components/__init__.py",   # Model components (e.g., layers, blocks)
    f"src/{project_name}/utils/__init__.py",        # Utility functions (e.g., logging, metrics)
    f"src/{project_name}/config/__init__.py",       # Config module init
    f"src/{project_name}/config/configuration.py",  # Config loading logic
    f"src/{project_name}/pipeline/__init__.py",     # Pipeline orchestration
    f"src/{project_name}/entity/__init__.py",       # Data/model entities (e.g., dataclasses)
    f"src/{project_name}/constants/__init__.py",    # Global constants
    "config/config.yaml",                           # Central config file (DVC/MLflow compatible)
    "dvc.yaml",                                     # DVC pipeline definition
    "params.yaml",                                  # Hyperparameters and runtime configs
    "requirements.txt",                             # Python dependencies
    "setup.py",                                     # Package setup for pip installability
    "research/trials.ipynb",                        # Experimental notebook for prototyping
    "templates/index.html"                          # Optional frontend template (Flask/SageMaker UI)
               ]

# ────────────────────────────────────────────────────────────────────────────────────────
# Directory and File Creation Logic
# ────────────────────────────────────────────────────────────────────────────────────────

for filepath in list_of_files:
    filepath          = Path(filepath)             # Convert string path to OS-agnostic Path object to handle '/' and '\' across platforms
    filedir, filename = os.path.split(filepath)

    # Create directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create empty file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Empty file placeholder
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")