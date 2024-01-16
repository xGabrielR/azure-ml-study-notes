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

prep_comp = load_component("./prep.yml")
print(prep_comp)

prep = ml_client.components.create_or_update(prep_comp)
print(prep)