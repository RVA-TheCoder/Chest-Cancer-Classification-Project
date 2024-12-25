# Update the src/cnn_classifier/entity

from dataclasses import dataclass
from pathlib import Path

# For dataIngestion stage
@dataclass(frozen=True)
class DataIngestionConfig:

    """
    dataclasses provides a decorator (@dataclass) to automatically generate methods like
    __init__, __repr__, and __eq__ for classes, thus simplifying the creation
     of data containers.

    root_dir, source_url, local_data_file, unzip_dir : arguments to __init__ method.

    """
    root_dir : Path
    source_url : str
    local_data_file : Path
    unzip_dir : Path


# For Base model stage

@dataclass(frozen=True)
class PrepareBasseModelConfig:

    root_dir: Path
    base_model_path : Path
    custom_base_model_path: Path
    params_include_top : bool
    params_weights : str
    params_image_size : list
    params_learning_rate : float
    params_classes : int
    






