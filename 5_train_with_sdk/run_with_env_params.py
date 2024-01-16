import os
from azure.ai.ml import MLClient, command
from azure.identity import AzureCliCredential

RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")

TAGS = {
    "created_by": "gabriel_r"
}

ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)

job = command(
    code="./src",
    command="python train_env.py",
    display_name="train-model",
    experiment_name="train-classification-model",
    tags=TAGS,

    # Compute
    timeout=15,
    compute="instance",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    environment_variables={"PATH": "https://azuremlexamples.blob.core.windows.net/datasets/iris.csv"}
)

returned_job = ml_client.create_or_update(job)