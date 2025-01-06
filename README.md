# Chest-Cancer-Classification-Project
CNN DL project with MLFow and DVC with AWS deployment

# For transfer Learning in tensorflow with data augmentation 
link : https://www.tensorflow.org/guide/keras/transfer_learning

## Project WorkFlow

1. Update config/config.yaml file
2. update secrets.yaml file [optional]
3. Update params.yaml file
4. Update the src/cnn_classifier/entity/config_entity.py.
5. Update the src/cnn_classifier/config/configuration.py file.
6. Update the src/cnn_classifier/components.
7. Update the src/cnn_classifier/pipeline.
8. Update the main.py file.
9. Update the dvc.yaml file.


##  dagshub

MLFLOW_TRACKING_URI=https://dagshub.com/Aakash00004/Chest-Cancer-Classification-Project.mlflow

MLFLOW_TRACKING_USERNAME=Aakash00004

MLFLOW_TRACKING_PASSWORD=620c670a001ca0eca0af7e36c8140bf200e2e4ed



### Run below commands at the terminal to export  env variables in the powershell or gitbash current session state:

##### In powershell we use ''set'' and in gitbash we can use ''export''
set MLFLOW_TRACKING_URI=https://dagshub.com/Aakash00004/Chest-Cancer-Classification-Project.mlflow

set MLFLOW_TRACKING_USERNAME=Aakash00004 

set MLFLOW_TRACKING_PASSWORD=620c670a001ca0eca0af7e36c8140bf200e2e4ed










