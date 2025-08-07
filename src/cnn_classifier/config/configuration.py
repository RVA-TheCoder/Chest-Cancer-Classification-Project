import os
import dagshub
from cnn_classifier.constants import *
from cnn_classifier.utils.common import read_yaml, create_directories, save_json
from cnn_classifier.entity.config_entity import (DataIngestionConfig ,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)






# Update the src/cnn_classifier/config/configuration.
class ConfigurationManager:
     
    """
    Reads and parses the 'config' and 'parameter' YAML files, sets up necessary directories,
    and provides configuration objects for different pipeline stages like
    data ingestion, model preparation, training, and evaluation.
    """

    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):

        """
        Initializes the ConfigurationManager by reading YAML configuration files
        and creating the root artifact directory.

        Parmeters : 
            (a) config_filepath (str): Path to the 'config/config.yaml' file.
            (b) params_filepath (str): Path to the 'params.yaml' file.
        
        """
        
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)

        # Creating directory
        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:

        """
        Generates and returns the DataIngestionConfig object by reading data ingestion
        section from config file and ensuring directories exist.

        Returns:
            DataIngestionConfig: Configuration object for data ingestion.
        """
        config = self.config.data_ingestion  

        # Create a directory
        create_directories([config.root_dir])

        # Creating an object of DataIngestionConfig class
        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
            )

        return data_ingestion_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        
        """
        Generates and returns the PrepareBaseModelConfig object using values from the
        'config' and 'params' files.

        Returns:
            PrepareBaseModelConfig: Configuration for preparing the base CNN model.
        """

        config=self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config= PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            custom_base_model_path=Path(config.custom_base_model_path),
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_image_size=self.params.INPUT_SHAPE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_classes=self.params.CLASSES
                )
        
        return prepare_base_model_config
    
    
    def get_training_config(self) -> TrainingConfig:

        """
        Generates and returns the TrainingConfig object with necessary paths
        and parameters for model training.

        Returns:
            TrainingConfig: Configuration for model training.
        """
        
        params=self.params
        training=self.config.training
        prepare_base_model=self.config.prepare_base_model
        
        # Donot give local system related absoute path, otherwise it will throw error when the app will run on the production server
        training_data=Path(os.path.join(self.config.data_ingestion.unzip_dir, r"data/train") )
        testing_data=Path(os.path.join(self.config.data_ingestion.unzip_dir, r"data/test") )

        create_directories( [Path(training.root_dir)] )

        
        training_config=TrainingConfig(

            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            custom_base_model_path=Path(prepare_base_model.custom_base_model_path),
            training_data=Path(training_data),
            testing_data=Path(testing_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.INPUT_SHAPE,
            params_learning_rate=params.LEARNING_RATE,
            params_metrices=params.METRICES
        )

        return training_config
    
    # Part of get_evaluation_config(self) method below
    def initialize_dagshub(self,
                           Repo_owner:str='Aakash00004' ,
                           Repo_name:str='Chest-Cancer-Classification-Project' ,
                           Mlflow:bool=True):
        
        """
        Below details of Repo_owner , Repo_name & Mlflow value are collected from the dagshub account 
        under 'remote' option of the repo : Chest-Cancer-Classification-Project
        
        Repo_owner='Aakash00004' ,
        Repo_name='Chest-Cancer-Classification-Project' ,
        Mlflow=True
        
        This code initialize a DagsHub repository or DagsHub-related functionality for experiment tracking using MLflow..
        
        Initialization includes:
        Creates a repository on DagsHub if it doesn’t exist yet.

        If dvc flag is set, adds the DagsHub repository as a dvc remote.

        If mlflow flag is set, initializes MLflow environment variables to enable 
        logging experiments into the DagsHub hosted MLflow. That means that if you call 
        dagshub.init() in your script, then any MLflow function called later in the script
        will log to the DagsHub hosted MLflow.

        Parameters : 
            (a) Repo_owner (str): Username or organization name on DagsHub.
            (b) Repo_name (str): Repository name on DagsHub.
            (c) Mlflow (bool): Whether to enable MLflow logging to DagsHub-hosted MLflow.
        
        """
        dagshub.init(repo_owner=Repo_owner, 
                     repo_name=Repo_name, 
                     mlflow=Mlflow)
        
    
    def get_evaluation_config(self) -> EvaluationConfig:

        """
        Generates and returns the EvaluationConfig object for model evaluation.
        Also initializes DagsHub for MLflow logging.

        Returns:
            EvaluationConfig: Configuration for evaluation and MLflow logging.
        """
        
        # Donot give local system related absoute path, otherwise it will throw error when the app will run on the production server
        training_data=Path(os.path.join(self.config.data_ingestion.unzip_dir, r"data/train") )
        testing_data=Path(os.path.join(self.config.data_ingestion.unzip_dir, r"data/test") )
        
        # calling method
        self.initialize_dagshub()

        eval_config=EvaluationConfig(
            # Donot give local system related absoute path, otherwise it will throw error when the app will run on the production server
            trained_model_path="trained_model/training/trained_model.keras",
            
            training_data=Path(training_data),
            testing_data=Path(testing_data),
            
            #mlflow_uri="https://dagshub.com/Aakash00004/Chest-Cancer-Classification-Project.mlflow",
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI"),
            
            all_params=self.params,
            params_image_size=self.params.INPUT_SHAPE,
            params_batch_size=self.params.BATCH_SIZE
        )
        
        return eval_config
    
    
    
    
    