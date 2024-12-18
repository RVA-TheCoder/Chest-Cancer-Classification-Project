from cnn_classifier.constants import *
from cnn_classifier.utils.common import read_yaml, create_directories
from cnn_classifier.entity.config_entity import DataIngestionConfig

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