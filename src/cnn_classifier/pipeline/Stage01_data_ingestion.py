from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.data_ingestion import DataIngestion
from cnn_classifier import logger




STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    
    """
    Stage 01: Data Ingestion Pipeline

    This module defines the pipeline for data ingestion, including downloading and extracting
    dataset files. It uses configuration management to get relevant paths and URLs.

    The stage includes:
    - Downloading the dataset ZIP file from a specified URL.
    - Extracting the ZIP file to a specified directory.
    """

    def __init__(self):

        pass

    def main(self):
        
        """
        Executes the data ingestion pipeline:
        - Retrieves configuration settings.
        - Downloads the dataset.
        - Extracts the ZIP file.
        """
        
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":

    try:
        
        logger.info(f">>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
        
        obj=DataIngestionTrainingPipeline()
        obj.main()

        logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<")

    except Exception as e :
        
        logger.exception(e)
        raise e


