# ────────────────────────────────────────────────────────────────────────────────────────
# Standard Library and Third-Party Imports
# ────────────────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import yaml
import joblib
import base64

from pathlib        import Path
from typing         import Any

from box            import ConfigBox            # converts dict to box for easy element access
from box.exceptions import BoxValueError
from ensure         import ensure_annotations   # avoids mismatch of func args and return types

from cnnClassifier  import logger               # Centralized logger instance

# ────────────────────────────────────────────────────────────────────────────────────────
# YAML Utilities
# ────────────────────────────────────────────────────────────────────────────────────────

@ensure_annotations   # Ensures all function arguments and return types match their type annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
             path_to_yaml (Path): Path to the YAML file.

    Raises:
             ValueError         : If the YAML file is empty.
             Exception          : For any other unexpected errors.

    Returns:
             ConfigBox          : Parsed YAML content with dot-accessible keys.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully : {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e

# ────────────────────────────────────────────────────────────────────────────────────────
# Directory Management
# ────────────────────────────────────────────────────────────────────────────────────────

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates a list of directories if they don't already exist.

    Args:
        path_to_directories (list) : List of directory paths.
        verbose (bool)             : Whether to log creation messages.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at : {path}")

# ────────────────────────────────────────────────────────────────────────────────────────
# JSON Utilities
# ────────────────────────────────────────────────────────────────────────────────────────

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dictionary as a JSON file.

    Args:
           path (Path) : Destination path for the JSON file.
           data (dict) : Data to be saved.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns its contents as a ConfigBox.

    Args:
             path (Path) : Path to the JSON file.

    Returns:
             ConfigBox   : Parsed JSON content with dot-accessible keys.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

# ────────────────────────────────────────────────────────────────────────────────────────
# Binary File Utilities (e.g., model artifacts, encoders)
# ────────────────────────────────────────────────────────────────────────────────────────

@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves Python object as a binary file using joblib.

    Args:
             data (Any)  : Object to be serialized.
             path (Path) : Destination path.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads a binary file and returns the stored object.

    Args:
             path (Path) : Path to the binary file.

    Returns:
             Any         : Deserialized Python object.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from : {path}")
    return data

# ────────────────────────────────────────────────────────────────────────────────────────
# File Size Utility
# ────────────────────────────────────────────────────────────────────────────────────────

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of a file in kilobytes.

    Args:
             path (Path) : Path to the file.

    Returns:
             str         : Approximate size in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

# ────────────────────────────────────────────────────────────────────────────────────────
# Base64 Image Encoding/Decoding (for API or UI integration)
# ────────────────────────────────────────────────────────────────────────────────────────

def decodeImage(imgstring: str, fileName: str):
    """
    Decodes a base64 image string and writes it to disk.

    Args:
           imgstring (str) : Base64-encoded image string.
            fileName (str) : Destination filename.
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)


def encodeImageIntoBase64(croppedImagePath: str) -> bytes:
    """
    Encodes an image file into base64 format.

    Args:
               croppedImagePath (str)  : Path to the image file.

    Returns:
               bytes                   : Base64-encoded image content.
    """
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())