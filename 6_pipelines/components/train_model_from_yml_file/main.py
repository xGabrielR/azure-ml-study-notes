import os
from azure.ai.ml import MLClient, load_component
from azure.identity import AzureCliCredential

RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")

ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)

iris_model_logistic = load_component("./iris_model_logistic.yml")
print(iris_model_logistic)

prep = ml_client.components.create_or_update(iris_model_logistic)
print(prep)