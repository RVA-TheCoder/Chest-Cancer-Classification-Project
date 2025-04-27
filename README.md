# Chest-Cancer-Classification-Project
CNN DL project with MLFow and DVC with AWS deployment

# For transfer Learning in tensorflow with data augmentation 
link : https://www.tensorflow.org/guide/keras/transfer_learning

## Project WorkFlow

1. Update config/config.yaml file
2. Update params.yaml file
3. Update the src/cnn_classifier/entity/config_entity.py.
4. Update the src/cnn_classifier/config/configuration.py file.
5. Update the src/cnn_classifier/components.
6. Update the src/cnn_classifier/pipeline.
7. Update the main.py file.
8. Update the dvc.yaml file.


##  dagshub

MLFLOW_TRACKING_URI=https://dagshub.com/Aakash00004/Chest-Cancer-Classification-Project.mlflow

MLFLOW_TRACKING_USERNAME=Aakash00004

MLFLOW_TRACKING_PASSWORD=620c670a001ca0eca0af7e36c8140bf200e2e4ed



### Run below commands at the terminal to export  env variables in the powershell or gitbash current session state:

##### In powershell we use ''set'' and 
set MLFLOW_TRACKING_URI=https://dagshub.com/Aakash00004/Chest-Cancer-Classification-Project.mlflow

set MLFLOW_TRACKING_USERNAME=Aakash00004 

set MLFLOW_TRACKING_PASSWORD=620c670a001ca0eca0af7e36c8140bf200e2e4ed

##### In gitbash we use ''export''
export MLFLOW_TRACKING_URI=https://dagshub.com/Aakash00004/Chest-Cancer-Classification-Project.   mlflow

export MLFLOW_TRACKING_USERNAME=Aakash00004 

export MLFLOW_TRACKING_PASSWORD=620c670a001ca0eca0af7e36c8140bf200e2e4ed

## dvc : for pipeline tracking 

#### After creating the dvc.yaml file , Run 'dvc repro' at the terminal to execute the file.
#### Above command will generate the 'dvc.lock' file that has all the details regarding our pipeline.
#### Run 'dvc dag' at the terminal for creating the dependency graph of the pipeline stages using dvc.






