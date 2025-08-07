from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.prepare_base_model import PrepareBaseModel
from cnn_classifier import logger


STAGE_NAME = "Prepare BaseModel Stage"

class PrepareBaseModelPipeline:
    
    """
    Stage 02: Prepare Base Model Pipeline

    This module defines the pipeline to prepare the base CNN model used for training.
    The base model is typically a pre-trained model (e.g., from Keras Applications),
    which is then customized (e.g., by adding new top layers) for the specific classification task.

    The stage includes:
        - Loading the base model architecture.
        - Modifying/customizing the base model as per the configuration.
    """

    def __init__(self):

        pass

    def main(self):
        
        """
        Executes the prepare base model pipeline:
            - Loads configuration settings.
            - Retrieves and loads the base model.
            - Customizes the base model (e.g., adds top layers).
        """
       
        config=ConfigurationManager()
        prepare_base_model_config=config.get_prepare_base_model_config()
        prepare_base_model=PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.custom_base_model()



if __name__ == "__main__":

    try:
        
        logger.info(f">>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
        
        obj=PrepareBaseModelPipeline()
        obj.main()

        logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<\n\nx================x")

    except Exception as e :
        
        logger.exception(e)
        raise e





