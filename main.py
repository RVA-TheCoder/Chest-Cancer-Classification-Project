import subprocess
import os
from pathlib import Path
import dagshub

from cnn_classifier import logger



"""
Main script to sequentially run all pipeline stages of the Adenocarcinoma CNN project.
Each stage is executed as a separate subprocess. 
Useful for both local testing and production deployment.
"""


# ---------------------- #
#     Stage 01: Data Ingestion
# ---------------------- #
STAGE_NAME="Data Ingestion stage"

try:

    # Get current working directory and standardize path format
    cwd = os.getcwd().replace("\\", "/")  # Windows path fix : windows system uses "\\" backward slash in the filepath
    
    #print(cwd)
    
    # For production deployment , use below script path
    #script_path="src/cnn_classifier/pipeline/Stage01_data_ingestion.py"
    
    # Define script path for local system (adjust for production if needed)
    script_path = Path(f"{cwd}/src/cnn_classifier/pipeline/Stage01_data_ingestion.py")
    #print("Complete script path : ",script_path) 

    subprocess.run(["python", script_path])
    
except Exception as e:
    raise e



# ---------------------- #
#     Stage 02: Prepare Base Model
# ---------------------- #
STAGE_NAME="Prepare BaseModel Stage"

try:

    # Get current working directory and standardize path format
    cwd = os.getcwd().replace("\\", "/")  # Windows path fix : windows system uses "\\" backward slash in the filepath
    #print(cwd)
    
    # For production deployment , use below script path
    #script_path="src/cnn_classifier/pipeline/Stage02_prepare_base_model.py"
    
    script_path = Path(f"{cwd}/src/cnn_classifier/pipeline/Stage02_prepare_base_model.py")
    #print("Complete script path : ",script_path)
    
    
    logger.info(f">>> Running {STAGE_NAME}")
    subprocess.run(["python", script_path])
    logger.info(f">>> Completed {STAGE_NAME}")
    
except Exception as e:
    raise e



# ---------------------- #
#     Stage 03: Model Training
# ---------------------- #
STAGE_NAME="Model Training"

try:

    # Get current working directory and standardize path format
    cwd = os.getcwd().replace("\\", "/")  # Windows path fix : windows system uses "\\" backward slash in the filepath
    #print(cwd)
    
    # For production deployment , use below script path
    #script_path="src/cnn_classifier/pipeline/Stage03_model_trainer.py"
    
    
    script_path = Path(f"{cwd}/src/cnn_classifier/pipeline/Stage03_model_trainer.py")
    #print("Complete script path : ",script_path)

    
    logger.info(f">>> Running {STAGE_NAME}")
    subprocess.run(["python", script_path])
    logger.info(f">>> Completed {STAGE_NAME}")
    
except Exception as e:
    raise e




# ---------------------- #
#     Stage 04: Model Evaluation
# ---------------------- #
STAGE_NAME="Evaluation Stage"

try:

    # Get current working directory and standardize path format
    cwd = os.getcwd().replace("\\", "/")  # Windows path fix : windows system uses "\\" backward slash in the filepath
    #print(cwd)
    
    # For production deployment , use below script path
    #script_path="src/cnn_classifier/pipeline/Stage04_model_evaluation_mlflow.py"
  
    script_path = Path(f"{cwd}/src/cnn_classifier/pipeline/Stage04_model_evaluation_mlflow.py")
    #print("Complete script path : ",script_path)

    
    logger.info(f">>> Running {STAGE_NAME}")
    subprocess.run(["python", script_path])
    logger.info(f">>> Completed {STAGE_NAME}")
    
except Exception as e:
    raise e 














