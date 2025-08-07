import os
import zipfile
import gdown
from cnn_classifier import logger
from cnn_classifier.utils.common import get_size
from cnn_classifier.entity.config_entity import DataIngestionConfig




class DataIngestion:
    
    """
    Component to handle data ingestion for the CNN classifier project.
    Includes downloading the dataset from a URL(gdrive) and extracting it.
    """

    def __init__(self, config:DataIngestionConfig):
        
        """
        Initializes the DataIngestion class with the provided configuration.

        Parameters : 
            (a) config (DataIngestionConfig): Configuration object containing paths and URLs.
        """

        self.config = config

    def download_file(self) ->str :

        """
        Downloads a ZIP file from a Google Drive URL and saves it locally.

        Returns:
            str: Path to the downloaded ZIP file.

        Raises:
            Exception: If the download fails or the URL is incorrect.
        """
        
        try :
            
            # url from where data will be downloaded
            dataset_url=self.config.source_url
            # where data will be saved
            zip_download_dir=self.config.local_data_file
            # creating root directory for data ingestion if not already been created
            os.makedirs(name=self.config.root_dir , exist_ok=True)

            # Convert Google Drive shareable link to direct download link
            file_id=dataset_url.split("/")[-2]
            prefix="https://drive.google.com/uc?/export=download&id="
            file_url=prefix+file_id
            
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
            # downloading the file from gdrive
            gdown.download(url=file_url,output=zip_download_dir)
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    def extract_zip_file(self):

        """
        Extracts the contents of the downloaded ZIP file to the specified directory.

        Raises:
            Exception: If the zip extraction fails due to a corrupted file or path issue.
        """
        unzip_dir_path=self.config.unzip_dir

        # Creating the directory where data zip file will be extracted, if not created already
        os.makedirs(name=unzip_dir_path, exist_ok=True)

        with zipfile.ZipFile(file=self.config.local_data_file, mode='r') as zip_ref:
            
            # path : specifies a  directory to extract to.
            zip_ref.extractall(path=unzip_dir_path)
            logger.info(f"Extracted zip file to {unzip_dir_path}")
