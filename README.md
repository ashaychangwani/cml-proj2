# Cloud & ML Project 2
Using the GitHub Repo
=====================

This section provides a step-by-step guide on how to use the GitHub repo containing the improved implementation of the Multi-ResNet model with support for single and multiple GPUs, as well as branches for API integration, Docker deployment, and the original implementation.

Branches Overview
-----------------

The repo contains the following branches:

- `api`: This branch contains a FastAPI server for model deployment and serving.
- `main`: This branch contains the improved single GPU implementation of the Multi-ResNet model.
- `multi-gpu`: This branch contains the improved Multi-ResNet model with support for multiple GPUs.
- `original-impl`: This branch contains the original single GPU implementation as described in the reference paper.
- `training-docker`: This branch contains all the Docker deployment YAML files required for training the model in a containerized environment.

Steps to Use the Repo
---------------------

1. **Clone the repo**: First, clone the repository to your local machine using the following command:

```git clone https://github.com/ashaychangwani/cml-proj2.git```


2. **Navigate to the repo**: Change your current working directory to the cloned repo:

```cd cml-proj2```


3. **Select the desired branch**: Depending on the desired functionality, switch to the appropriate branch. For example, to switch to the `main` branch, run:

```git checkout main```


4. **Install dependencies**: Before running the training script, ensure that you have all the required dependencies installed. You can install them using the following command:

```pip install -r requirements.txt```


5. **Run the training script**: Now, you can train the Multi-ResNet model using the provided command. Make sure to replace `/imagenet` with the path to your ImageNet dataset directory.

```python train.py train multi_resnet18_kd --data-dir /imagenet --epoch 1 --batch-size 64```


This command trains the `multi_resnet18_kd` model on the ImageNet dataset for one epoch with a batch size of 64.

You can also select `multi_resnet50_kd` to create the knowledge distilled model of ResNet50 instead of ResNet18.

6. **Explore other branches**: To use the functionalities provided by other branches, switch to the desired branch using `git checkout` and follow the instructions provided in the corresponding `README.md` file.

By following these steps, you can utilize the different branches of the GitHub repo to train and deploy the Multi-ResNet model with single and multiple GPU support, as well as leverage the API and Docker features for efficient model serving and containerized deployment.
