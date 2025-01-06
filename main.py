from cnn_classifier import logger
import subprocess
import os
from pathlib import Path


STAGE_NAME="Data Ingestion stage"

try:

    #cwd=os.getcwd()
    cwd=r"E:/STUDY/TENSORFLOW/Projects/1_CNN_Project"
    script_path=cwd+"/src/cnn_classifier/pipeline/Stage01_data_ingestion.py"
    script_path = Path(script_path)
    #print(script_path)

    subprocess.run(["python", script_path])
    
except Exception as e:
    raise e



STAGE_NAME="Prepare BaseModel Stage"

try:

    #cwd=os.getcwd()
    cwd=r"E:/STUDY/TENSORFLOW/Projects/1_CNN_Project"
    script_path=cwd+"/src/cnn_classifier/pipeline/Stage02_prepare_base_model.py"
    script_path = Path(script_path)
    #print(script_path)

    subprocess.run(["python", script_path])
    
except Exception as e:
    raise e



STAGE_NAME="Model Training"

try:

    #cwd=os.getcwd()
    cwd=r"E:/STUDY/TENSORFLOW/Projects/1_CNN_Project"
    script_path=cwd+"/src/cnn_classifier/pipeline/Stage03_model_trainer.py"
    script_path = Path(script_path)
    #print(script_path)

    subprocess.run(["python", script_path])
    
except Exception as e:
    raise e



STAGE_NAME="Evaluation Stage"

try:

    #cwd=os.getcwd()
    cwd=r"E:/STUDY/TENSORFLOW/Projects/1_CNN_Project"
    script_path=cwd+"/src/cnn_classifier/pipeline/Stage04_model_evaluation_mlflow.py"
    script_path = Path(script_path)
    #print(script_path)

    subprocess.run(["python", script_path])
    
except Exception as e:
    raise e














