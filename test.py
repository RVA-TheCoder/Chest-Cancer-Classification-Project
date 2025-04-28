# Do not include this file while commiting to github because it is used for testing purpose only.
import os
from pathlib import Path
# import dagshub
# dagshub.init(repo_owner='Aakash00004', repo_name='Chest-Cancer-Classification-Project', mlflow=True)


# print(os.getenv("MLFLOW_TRACKING_URI"))
# print(os.getenv("MLFLOW_TRACKING_USERNAME"))
# print(os.getenv("MLFLOW_TRACKING_PASSWORD"))

# print("Inside test.py file")

cwd = os.getcwd()
cwd = cwd.replace("\\","/")
print(cwd)


