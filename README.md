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

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=<from iam user .csv file>

    AWS_SECRET_ACCESS_KEY=<from iam user .csv file>

    AWS_REGION = ap-south-1

    AWS_ECR_LOGIN_URI = <your ecr uri>

    ECR_REPOSITORY_NAME = <your-ecr-name>






