from dataclasses import dataclass
from pathlib import Path


# For DataIngestion stage
@dataclass(frozen=True)
class DataIngestionConfig:

    """
    dataclasses provides a decorator (@dataclass) to automatically generate methods like
    __init__, __repr__, and __eq__ for classes, thus simplifying the creation of data containers.
     

    @dataclass(frozen=True) : Makes the DataIngestionConfig instances immutable. 
                              Once an object is created, its attributes cannot be changed.
                              Attempting to modify an attribute will raise a FrozenInstanceError.
                              
    Parameters : 
        (a) root_dir : root directory path for dataingestion stage.
        (b) source_url : google drive url where data is stored.
        (c) local_data_file : local path where data file being downloaded will be stored.
        (d) unzip_dir : dir path where data file unzip operation will be done.
    """
    root_dir : Path
    source_url : str
    local_data_file : Path
    unzip_dir : Path


# For Base model stage
@dataclass(frozen=True)
class PrepareBaseModelConfig:

    root_dir: Path
    base_model_path : Path
    custom_base_model_path: Path
    params_include_top : bool
    params_weights : str
    params_image_size : list
    params_learning_rate : float
    params_classes : int




# For training custom_base model
@dataclass(frozen=True)
class TrainingConfig:
    root_dir:Path
    trained_model_path:Path
    custom_base_model_path:Path
    training_data:Path
    testing_data:Path
    params_epochs:int
    params_batch_size:int
    params_is_augmentation:bool
    params_image_size:list | tuple
    params_learning_rate:float
    params_metrices:list



# For Evaluation of trained model
@dataclass(frozen=True)
class EvaluationConfig:
    trained_model_path:Path
    training_data:Path
    testing_data:Path
    all_params:dict
    mlflow_uri:str
    params_image_size:list | tuple
    params_batch_size:int
    
    



