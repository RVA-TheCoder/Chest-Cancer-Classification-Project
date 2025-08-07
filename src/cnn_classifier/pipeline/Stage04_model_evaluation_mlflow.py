from cnn_classifier.config.configuration import ConfigurationManager
from cnn_classifier.components.model_evaluation_mlflow import Evaluation

from cnn_classifier import logger





STAGE_NAME="Evaluation Stage"

class Evaluation_Pipleine:
    
    """
    Stage 04: Model Evaluation with MLflow

    This module handles the evaluation of the trained CNN model.
    It performs:
        - Model evaluation using test data.
        - (Optional) Logging of evaluation metrics and model artifacts to MLflow for experiment tracking.

    This stage is critical for validating model performance before production deployment.
    """

    def __init__(self):
        
        pass

    def main(self):
        
        """
        Executes the evaluation pipeline:
            - Loads the evaluation configuration.
            - Runs model evaluation.
            - (Optional) Logs metrics and model to MLflow for experiment tracking.
        """

        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.model_evaluation()
        # comment below line while deploying the project to production because there we dont want experiment tracking and model logging
        #evaluation.log_into_mlflow()
        
    


if __name__ == '__main__':

    try:
        
        logger.info(f"*******************")
        logger.info(f">>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")

        # Creating object of class Evaluation_Pipleine
        obj = Evaluation_Pipleine()
        # calling method
        obj.main()

        logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<\n\n")

    except Exception as e:
        logger.exception(e)
        raise e





