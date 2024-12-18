from cnn_classifier import logger
from cnn_classifier.pipeline.Stage01_data_ingestion import DataIngestionTrainingPipeline
import subprocess
import os
from pathlib import Path



STAGE_NAME="Data Ingestion stage"

try:

    cwd=os.getcwd()
    script_path=cwd+"/src/cnn_classifier/pipeline/Stage01_data_ingestion.py"
    script_path = Path(script_path)
    #print(script_path)

    subprocess.run(["python", script_path])
    
except Exception as e:
    raise e











