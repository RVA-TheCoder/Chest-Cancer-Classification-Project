import os
import zipfile
import gdown
from cnn_classifier import logger
from cnn_classifier.utils.common import get_size
from cnn_classifier.entity.config_entity import DataIngestionConfig




class DataIngestion:

    def __init__(self, config:DataIngestionConfig):

        self.config = config

    def download_file(self) ->str :

        """
        fetch data from the url.
        """
        
        try :
            
            # url from where data will be downloaded
            dataset_url=self.config.source_url
            # where data will be saved
            zip_download_dir=self.config.local_data_file
            # creating root directory for data ingestion if not already been created
            os.makedirs(name=self.config.root_dir , exist_ok=True)

            file_id=dataset_url.split("/")[-2]
            prefix="https://drive.google.com/uc?/export=download&id="
            # downloading the file from gdrive
            file_url=prefix+file_id
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
            gdown.download(url=file_url,output=zip_download_dir)
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    def extract_zip_file(self):

        """
        This method extracts the zip file.
        """
        unzip_dir_path=self.config.unzip_dir

        # Creating the directory where data zip file will be extracted, if not created already
        os.makedirs(name=unzip_dir_path, exist_ok=True)

        with zipfile.ZipFile(file=self.config.local_data_file, mode='r') as zip_ref:
            
            # path : specifies a  directory to extract to.
            zip_ref.extractall(path=unzip_dir_path)

