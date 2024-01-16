"""
Is necessary a MLFLOW registred Model!
You can register from a run, pipeline...
Just need azure-mlflow and register a basic model, its easy. 
"""

import os
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential 
from azure.ai.ml.entities import (
    Model,
    Environment,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment
)


RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")

ENDPOINT_NAME = "grc_15012024_iris_online_endpoint"

TAGS = {
    "created_by": "gabriel_r"
}

ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)

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

# Put a model inside the endpoint
latest_model_version = max([int(m.version) for m in ml_client.models.list(name="registered_model_name")])
model = ml_client.models.get(name="registered_model_name", version=latest_model_version)

# create an online deployment.
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    description="Iris dataset deployment",
    tags=TAGS,

    model=model,
    environment=None,

    endpoint_name=ENDPOINT_NAME,
    instance_type="Standard_F4s_v2",
    instance_count=1,
)

blue_deployment_results = ml_client.online_deployments.begin_create_or_update(blue_deployment).result()
print(blue_deployment)

print(f"Deployment {blue_deployment_results.name} provisioning state: {blue_deployment_results.provisioning_state}")


ml_client.online_endpoints.invoke(
    endpoint_name=ENDPOINT_NAME,
    request_file={
        "input_data": {
            "columns": ["sepal.length","sepal.width","petal.length","petal.width"],
            "index": [1],
            "data": [[5.1,3.5,1.4,.2]]
        }
    },
    deployment_name="blue",
)

ml_client.online_endpoints.begin_delete(name=ENDPOINT_NAME)