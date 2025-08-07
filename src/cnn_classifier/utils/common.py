import os
from box.exceptions import BoxValueError
import yaml

import json
import joblib

from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

from cnn_classifier import logger



@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:

    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    parameters : 
        (a) path_to_yaml (Path): Path to the YAML configuration file.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: For all other exceptions.

    Returns:
        ConfigBox: Parsed YAML content as an object with dot-accessible keys.
    """

    try:
        with open(path_to_yaml, mode='r') as yaml_file :
            
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} loaded successfully")

            return ConfigBox(content)

    # First error we are trying to catch is BoxValueError    
    except BoxValueError:
        raise ValueError("yaml file is empty")
    
    except Exception as e:

        raise e

   
@ensure_annotations
def create_directories(path_to_directories : list, verbose=True):

    """
    Creates directories if they don't already exist.

    Parameters : 
        (a) path_to_directories (list[Path]): List of directory paths to create.
        (b) verbose (bool, optional): Whether to log the creation. Defaults to True.
    """

    for path in path_to_directories:

        os.makedirs(path, exist_ok=True)

        if verbose:
            logger.info(f"Created directory at : {path}")



@ensure_annotations
def save_json(path:Path, data:dict):

    """
    Saves a dictionary to a JSON file.

    Parameters : 
        (a) path (Path): Path where the JSON file should be saved.
        (b) data (dict): Data to save in JSON format.
    """
    
    with open(path, mode="w") as f:

        json.dump(obj=data, fp=f, indent=4)

    logger.info(f"json file saved at : {path}")


@ensure_annotations
def load_json(path:Path) -> ConfigBox:

    """
    Loads a JSON file and returns the content as a ConfigBox.

    Parameters : 
        (a) path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Parsed JSON content with dot-accessible keys.
    """

    with open(path, mode='r') as f:

        content=json.load(f)

    logger.info(f"json file loaded successfully from {path}")

    return ConfigBox(content)


@ensure_annotations
def save_bin(data : Any, path:Path):

    """
    Saves data to a binary file using joblib.

    Parameters :
        (a) data (Any): Data to be saved.
        (b) path (Path): File path for the binary output.
    
    """

    joblib.dump(value=data, filename=path)

    logger.info(f"binary file saved at : {path}")



@ensure_annotations
def load_bin(path:Path) -> Any:

    """
    Loads data from a binary file using joblib.

    Parameters :
        (a) path (Path): Path to the binary file.

    Returns:
        Any: Loaded data object.
    """
    
    data=joblib.load(path)
    logger.info(f"binary file loaded from : {path}")

    return data

@ensure_annotations
def get_size(path:Path) -> str:

    """
    Gets the size of a file in kilobytes (KB).

    Parameters : 
        (a) path (Path): Path to the file.

    Returns:
        str: File size in KB (rounded).
    """

    size_in_kb = round( os.path.getsize(path) / 1024 )

    return f" ~ {size_in_kb} KB"


def decodeImage(imgstring, filename):
    
    """
    Decodes a base64-encoded image string and saves it as an image file.

    Parameters : 
        (a) imgstring (str): The base64-encoded image string.
        (b) filename (str): The name (with path) of the file to save the decoded image to.

    Example:
        decodeImage(imgstring, "output_image.png")
    """

    imgdata=base64.b64decode(imgstring)

    with open(filename, mode='wb') as f:

        f.write(imgdata)
    

def encodeImageIntoBase64(croppedImagePath):
    
    """
    Reads an image file from the given path and encodes its content into a base64 string.

    Parameters : 
        (a) croppedImagePath (str): The file path of the image to be encoded.

    Returns:
        bytes: The base64-encoded representation of the image.

    Example:
        encoded_string = encodeImageIntoBase64("cropped_image.jpg")
    """

    with open(croppedImagePath, mode="rb") as f:

        return base64.b64encode(f.read())
    
    





