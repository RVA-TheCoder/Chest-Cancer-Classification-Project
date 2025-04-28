from cnn_classifier import logger
import subprocess
import os
from pathlib import Path
import dagshub


STAGE_NAME="Data Ingestion stage"

try:

    # For local Windows system project run, use below script_path 
    cwd = os.getcwd()
    cwd = cwd.replace("\\","/") # windows system uses "\\" backward slash in the filepath
    #print(cwd)
    script_path=cwd+"/src/cnn_classifier/pipeline/Stage01_data_ingestion.py"
    
    # For production deployment , use below script path
    #script_path="src/cnn_classifier/pipeline/Stage01_data_ingestion.py"
    
    script_path = Path(script_path)
    #print("Complete script path : ",script_path) 

    subprocess.run(["python", script_path])
    
except Exception as e:
    raise e



STAGE_NAME="Prepare BaseModel Stage"

try:

    # For local Windows system project run, use below script_path
    cwd = os.getcwd()
    cwd = cwd.replace("\\","/")    # windows system uses "\\" backward slash in the filepath
    #print(cwd)
    script_path=cwd+"/src/cnn_classifier/pipeline/Stage02_prepare_base_model.py"
    
    # For production deployment , use below script path
    #script_path="src/cnn_classifier/pipeline/Stage02_prepare_base_model.py"
    
    script_path = Path(script_path)
    #print("Complete script path : ",script_path)

    subprocess.run(["python", script_path])
    
except Exception as e:
    raise e



STAGE_NAME="Model Training"

try:

    # For local Windows system project run, use below script_path
    cwd = os.getcwd()
    cwd = cwd.replace("\\","/")     # windows system uses "\\" backward slash in the filepath
    #print(cwd)
    script_path=cwd+"/src/cnn_classifier/pipeline/Stage03_model_trainer.py"
  
    # For production deployment , use below script path
    #script_path="src/cnn_classifier/pipeline/Stage03_model_trainer.py"
    
    script_path = Path(script_path)
    #print("Complete script path : ",script_path)

    subprocess.run(["python", script_path])
    
except Exception as e:
    raise e



STAGE_NAME="Evaluation Stage"

try:

    # For local Windows system project run, use below script_path
    cwd = os.getcwd()
    cwd = cwd.replace("\\","/")     # windows system uses "\\" backward slash in the filepath
    #print(cwd)
    script_path=cwd+"/src/cnn_classifier/pipeline/Stage04_model_evaluation_mlflow.py"
    
    # For production deployment , use below script path
    #script_path="src/cnn_classifier/pipeline/Stage04_model_evaluation_mlflow.py"
  
    script_path = Path(script_path)
    #print("Complete script path : ",script_path)

    subprocess.run(["python", script_path])
    
except Exception as e:
    raise e 














