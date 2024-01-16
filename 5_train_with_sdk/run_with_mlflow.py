import os
from azure.ai.ml import MLClient, command
from azure.identity import AzureCliCredential

from azure.ai.ml.entities import Environment, BuildContext

RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")
TRAIN_DATA_PATH = "https://azuremlexamples.blob.core.windows.net/datasets/iris.csv"

TAGS = {
    "created_by": "gabriel_r"
}

ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)

# You need to reference or create a new env with mlflow and azure-mlflow
env_docker_conda = Environment(
    name="grc-conda-mlflow",
    description="Environment created from a Docker image plus Conda environment.",
    tags=TAGS,

    conda_file="custom_ymls/mlflow.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
)

ml_client.environments.create_or_update(env_docker_conda)


job = command(
    code="./src",
    command=f"python train_args_mlflow.py --train_data_path {TRAIN_DATA_PATH}",
    display_name="train-model",
    experiment_name="train-classification-model",
    tags=TAGS,

    # Compute
    timeout=15,
    compute="instance",

    environment="grc-conda-mlflow",
)

returned_job = ml_client.create_or_update(job)