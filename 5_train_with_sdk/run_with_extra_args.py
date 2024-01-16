import os
from azure.ai.ml import MLClient, command
from azure.identity import AzureCliCredential

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

job = command(
    code="./src",
    
    command=f"python train_args.py --train_data_path {TRAIN_DATA_PATH}",

    display_name="train-model",
    experiment_name="train-classification-model",
    tags=TAGS,

    # Compute
    timeout=15,
    compute="instance",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
)

returned_job = ml_client.create_or_update(job)