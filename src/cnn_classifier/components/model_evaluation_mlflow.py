import tensorflow as tf
from pathlib import Path
import mlflow
import dagshub
import mlflow.keras
from urllib.parse import urlparse
from tensorflow.keras.utils import image_dataset_from_directory as Images
from cnn_classifier.entity.config_entity import EvaluationConfig
from cnn_classifier.utils.common import read_yaml, create_directories, save_json

class Evaluation:

    def __init__(self, config:EvaluationConfig):

        self.config = config

    def get_trained_model(self):

        return tf.keras.models.load_model( self.config.trained_model_path
                                         )   
                                            
    def get_test_data(self):


        self.images_test = Images(
                            directory=self.config.testing_data,
                            labels='inferred',
                            image_size = self.config.params_image_size[:-1],
                            batch_size = self.config.params_batch_size
                            )


    def save_score(self):

        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def model_evaluation(self):

        self.trained_model = self.get_trained_model()
        self.get_test_data()
        self.score = self.trained_model.evaluate(self.images_test)
        
        self.save_score()

    def log_into_mlflow(self):

        """
        This initializes the integration with DagsHub for the specified repository.

        The mlflow=True argument ensures MLflow logs (parameters, metrics, artifacts) are 
        synchronized with the DagsHub repository i.e., it sets the registry_uri as well for 
        MLflow tracking server where runs, parameters, metrics, and artifacts will be logged..

        """
        dagshub.init(repo_owner='Aakash00004', repo_name='Chest-Cancer-Classification-Project', mlflow=True)
        #print("Mlflow tracking URI",mlflow.get_tracking_uri())
       
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print("After dagshub.init of mlflow",tracking_url_type_store)
        
        with mlflow.start_run():

            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                                {"loss": self.score[0], "accuracy": self.score[1]}
                              )
            # Model registry does not work with file store
            #print("Before if statment of mlflow",tracking_url_type_store)
            if tracking_url_type_store != "file":

                print("Inside if statement of mlflow")

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.trained_model, "model", registered_model_name="Custom_VGG16_Model")

            else:

                #print("Inside else statment of mlflow")
                mlflow.keras.log_model(self.trained_model, "model")

   
