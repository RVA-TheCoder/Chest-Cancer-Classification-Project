import os
from cnn_classifier.constants import *
from cnn_classifier.utils.common import read_yaml, create_directories
from cnn_classifier.entity.config_entity import (DataIngestionConfig ,
                                                PrepareBasseModelConfig,
                                                TrainingConfig)


# Update the src/cnn_classifier/config/configuration.
class ConfigurationManager:

    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):

        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)

        # Creating directory
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        """
        returns the object of DataIngestionConfig class
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
    

    def get_prepare_base_model_config(self) -> PrepareBasseModelConfig:

        config=self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config= PrepareBasseModelConfig(
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

        params=self.params
        training=self.config.training
        prepare_base_model=self.config.prepare_base_model
        

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
            params_learning_rate=params.LEARNING_RATE
        )

        return training_config