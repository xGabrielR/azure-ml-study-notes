"""
Is necessary a MLFLOW registred Model!
You can register from a run, pipeline...
Just need azure-mlflow and register a basic model, its easy. 
"""

import os
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.identity import AzureCliCredential 
from azure.ai.ml.entities import (
    Model,
    Environment,
    CodeConfiguration,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment
)

RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")

ENDPOINT_NAME = "grc_15012024_iris_online_endpoint_custom"

TAGS = {
    "created_by": "gabriel_r"
}

ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)


# Create Endpoint
endpoint = ManagedOnlineEndpoint(
    location="eastus",
    name=ENDPOINT_NAME,
    description="Online Endpoint for Iris Model",
    auth_mode="key",
    tags=TAGS
)
endpoint.traffic = {"blue": 100}
print(endpoint)

endpoint_result = ml_client.begin_create_or_update(endpoint).result()
print(endpoint_result)


# Create New Env
env = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./src/conda.yml",
    name="deployment-environment",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env)


# Inference Model
model = Model(
    path="./model",
    type=AssetTypes.MLFLOW_MODEL,
    description="my sample mlflow model",
)


# Create Deployment
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    description="Iris dataset deployment",
    tags=TAGS,

    model=model,
    environment="deployment-environment",
    code_configuration=CodeConfiguration(
        code="./src",
        scoring_script="score.py"
    ),

    endpoint_name=ENDPOINT_NAME,
    instance_type="Standard_F4s_v2",
    instance_count=1,
)

blue_deployment_results = ml_client.online_deployments.begin_create_or_update(blue_deployment).result()
print(blue_deployment)

ml_client.online_endpoints.invoke(
    endpoint_name=ENDPOINT_NAME,
    deployment_name="blue",
    request_file={
        "input_data": {
            "columns": ["sepal.length","sepal.width","petal.length","petal.width"],
            "index": [1],
            "data": [[5.1,3.5,1.4,.2]]
        }
    },
)

ml_client.online_endpoints.begin_delete(name=ENDPOINT_NAME)