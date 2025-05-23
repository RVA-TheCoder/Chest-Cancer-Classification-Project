import os
from box.exceptions import BoxValueError
import yaml


from cnn_classifier import logger
import json
import joblib


from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:

    """
    reads yaml file and returns ConfigBox type output

    Parameters : 
    (a) path_to_yaml : (str) : path to yaml file

    Raises : 
          ValueError : if yaml file is empty
                       e : empty file

    Returns : 
            ConfigBox : ConfigBox type

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
    Create directories from path_to_directories.

    Paramters :

    (a) path_to_directories : (list) : list of path directories

    (b) verbose : (bool) : (optional): logs about directory creation.
                                       Defaults to True.

    """

    for path in path_to_directories:

        os.makedirs(path, exist_ok=True)

        if verbose:
            logger.info(f"Created directory at : {path}")



@ensure_annotations
def save_json(path:Path, data:dict):

    """
    save json data
    
    Parameters : 
    (a) path : (Path) : path to json file
    (b) data : (dict) : data to be saved in json file format.
    
    """
    with open(path, mode="w") as f:

        json.dump(obj=data, fp=f, indent=4)

    logger.info(f"json file saved at : {path}")


@ensure_annotations
def load_json(path:Path) -> ConfigBox:

    """
    
    load json file data at given path and returns a ConfigBox type output.
    
    Returns :
        ConfigBox : data as class attributes instead of dict
    """

    with open(path, mode='r') as f:

        content=json.load(f)

    logger.info(f"json file loaded successfully from {path}")

    return ConfigBox(content)


@ensure_annotations
def save_bin(data : Any, path:Path):

    """
    save binary file
    
    Parameters : 
    (a) data : (Any) : data to be saved in binary file format.
    (b) path : (Path) : path to binary file
    
    """

    joblib.dump(value=data, filename=path)

    logger.info(f"binary file saved at : {path}")



@ensure_annotations
def load_bin(path:Path) -> Any:

    """
    load binary data
    
    Parameters : 
    (a) path : (Path) : path to binary file.
    
    Returns : (Any) : object stored in the file
    
    """
    
    data=joblib.load(path)
    logger.info(f"binary file loaded from : {path}")

    return data

@ensure_annotations
def get_size(path:Path) -> str:

    """
    get the size of file in (KB)

    Parameters : 
    (a) path : (Path) : path to a file

    Returns : 
        str : size in KB

    """

    size_in_kb = round( os.path.getsize(path) / 1024 )

    return f" ~ {size_in_kb} KB"


def decodeImage(imgstring, filename):

    imgdata=base64.b64decode(imgstring)

    with open(filename, mode='wb') as f:

        f.write(imgdata)
    

def encodeImageIntoBase64(croppedImagePath):

    with open(croppedImagePath, mode="rb") as f:

        return base64.b64encode(f.read())
    
    





