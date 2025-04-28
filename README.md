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


##  Dagshub

### (i) For initializing the dagshub in your project use below code (available inside `remote` option of dagshub project repo ):

import dagshub

dagshub.init(repo_owner=<project repo owner>, repo_name=< project repo name>, mlflow=True)

### (ii) After initializing the dagshub, use below code for finding mlflow details:
print(os.getenv("MLFLOW_TRACKING_URI"))

print(os.getenv("MLFLOW_TRACKING_USERNAME"))

print(os.getenv("MLFLOW_TRACKING_PASSWORD"))


## dvc : for pipeline tracking 

#### After creating the dvc.yaml file ,
- first, Run `dvc init` at the terminal : for initializing the dvc in your project, then
- Run `dvc repro` at the terminal to execute the `dvc.yaml` file.
#### Above command will generate the 'dvc.lock' file that has all the details regarding our pipeline.
#### Run 'dvc dag' at the terminal for creating the dependency graph of the pipeline stages using dvc.

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the ECR URI: for eg., 56637...765.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine by running below commands at the EC2 terminal:
	
	#optional

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    - inside github project repo goto :->  setting>actions>runner>new self hosted runner> choose os> 
	- then run commands one by one at the EC2 terminal mentioned by github while creating the `self-hosted` runner.

# 7. Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to AWS ECR

	3. Launch Your AWS EC2 instance

	4. Pull Your image from ECR into EC2

	5. Lauch your docker image in EC2 instance

# 8. Setup github secrets:

    AWS_ACCESS_KEY_ID=<from iam user .csv file>

    AWS_SECRET_ACCESS_KEY=<from iam user .csv file>

    AWS_REGION = ap-south-1

    AWS_ECR_LOGIN_URI = <your ecr uri>

    ECR_REPOSITORY_NAME = <your-ecr-name>






