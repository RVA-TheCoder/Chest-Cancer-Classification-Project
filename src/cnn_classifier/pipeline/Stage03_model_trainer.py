from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_trainer import Training

from cnn_classifier import logger




STAGE_NAME = "Training"

class ModelTrainingPipeline:
    
    """
    Stage 03: Model Training Pipeline

    This module defines the pipeline responsible for training the CNN model.
    It includes the following steps:
        - Loading a customized base model.
        - Preprocessing the training and validation datasets.
        - Training the model using the provided configuration.

    This pipeline is part of the CNN-based lung cancer classification system.
    """

    def __init__(self):
        pass

    def main(self):
        
        """
        Executes the model training pipeline:
            - Loads the training configuration.
            - Loads the customized base model from previous stage.
            - Preprocesses training and validation data.
            - Trains the model and saves the results.
        """
        
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_custom_base_model()
        training.preprocess_data()
        training.train()

    

if __name__ == '__main__':

    try:
        logger.info(f"*******************")
        logger.info(f">>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")

        obj = ModelTrainingPipeline()
        obj.main()

        logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e