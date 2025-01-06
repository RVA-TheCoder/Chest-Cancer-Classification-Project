import os

import dagshub
dagshub.init(repo_owner='Aakash00004', repo_name='Chest-Cancer-Classification-Project', mlflow=True)


print(os.getenv("MLFLOW_TRACKING_URI"))
print(os.getenv("MLFLOW_TRACKING_USERNAME"))
print(os.getenv("MLFLOW_TRACKING_PASSWORD"))

print("Inside test.py file")

