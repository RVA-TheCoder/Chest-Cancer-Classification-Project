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
                            label_mode="categorical", # use loss=tf.keras.losses.CategoricalCrossentropy() because label_mode is set to 'categorical' 
                            image_size = self.config.params_image_size[:-1],
                            batch_size = self.config.params_batch_size
                            )


    def save_eval_score(self):

        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("ModelEvaluation_scores.json"), data=scores)

    def model_evaluation(self):

        # calling method
        self.trained_model = self.get_trained_model()
        # calling method
        self.get_test_data()
        
        # Evaluating the trained model performance on test data
        self.score = self.trained_model.evaluate(self.images_test)
        
        # calling method
        self.save_eval_score()


    def log_into_mlflow(self ,Repo_Owner:str='Aakash00004',
                        Repo_Name:str='Chest-Cancer-Classification-Project' ,
                        MlFlow:bool=True ):

        """
        This initializes the integration with DagsHub for the specified repository.

        The mlflow=True argument ensures MLflow logs (parameters, metrics, artifacts) are 
        synchronized with the DagsHub repository i.e., it sets the registry_uri as well for 
        MLflow tracking server where runs, parameters, metrics, and artifacts will be logged..

        """
        # No need of below code for  Repository Aakash00004/Chest-Cancer-Classification-Project to be
        # initialized! because we've initialized the repo inside the 
        # method 'get_evaluation_config' of class 'ConfigurationManager' inside src/cnn_classifier/config/configuration.py file
          
        # dagshub.init(repo_owner=Repo_Owner,
        #              repo_name=Repo_Name,
        #              mlflow=MlFlow)
        
        
        #For debugging purpose : 
        # print("Mlflow tracking URI",mlflow.get_tracking_uri())
       
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        #For debugging purpose :
        #print("After dagshub.init of mlflow : ",tracking_url_type_store)
        
        # Set the experiment name
        mlflow.set_experiment("My Chest Cancer Experiment")  
        
        with mlflow.start_run():

            #For debugging purpose :
            #print("inside mlflow.start_run() ")
            
            #For debugging purpose :
            #print(self.config.all_params)
            
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                                {"loss": self.score[0], "accuracy": self.score[1] }
                              )
            
            # Model registry does not work with file store
            
            #For debugging purpose :
            #print("Before if statement of mlflow tracking_url_type_store : ",tracking_url_type_store)
            if tracking_url_type_store != "file":

                #For debugging purpose :
                #print("Inside if statement of mlflow tracking_url_type_store : ")

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.trained_model,
                                       "model",
                                       registered_model_name="Custom_VGG16_Model")

            else:

                #For debugging purpose :
                #print("Inside else statment of mlflow")
                mlflow.keras.log_model(self.trained_model, "model")

   
