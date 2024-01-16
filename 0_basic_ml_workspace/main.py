"""

Objectives:

1. Create Resource Group.
2. Create ML Workspace.

You can create Resource Group and ML Workspace with:
- Interface;
- CLI;
- Python SDK;
- Terraform.

Using Terraform: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace-terraform?view=azureml-api-2&tabs=publicworkspace
Install CLI: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux?pivots=apt
Add ML Extention to CLI: https://learn.microsoft.com/en-us/training/modules/create-azure-machine-learning-resources-cli-v2/2-use-azure-cli-v2


## Using CLI

First Install Cli:
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

And add "az ml" extension to CLI
az extension add --name ml -y

    az login

    az -h

    az ml -h

    You can setup Subscription to dont need to 
    explicity provide on azure cli commands with

    az account set -s "YOUR_SUBSCRIPTION_NAME_OR_ID"

    az group --help

    az group create --help

    az group create --name "grcMlRg"
                    --location "eastus"
                    --subscription "TODO"
                    --output "json"
                    --tags "created_by=gabriel_r"
    
    az ml workspace -h

    az ml workspace create -h

    az ml workspace create --name "grcMlWs"
                           --subscription "TODO"
                           --resource-group "grcMlRg"
                           --location "eastus"
                           --description "Azure ML WS with CLI."
                           --tags "created_by=gabriel_r" date +"created_at=%y-%m-%d %H:%M:%S"

     az ml workspace delete --name "grcMlWs"
                            --subscription "TODO"
                            --resource-group "grcMlRg"
                            --all-resources
                            --yes

    az group delete --name "grcMlRg"
                    --subscription "TODO"
                    --yes

"""

## Using Python
# pip install azure-ai-ml

import os
from datetime import datetime

from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Workspace

RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")

TAGS = {
    "created_by": "gabriel_richter",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

resource_manager = ResourceManagementClient(
    AzureCliCredential(), subscription_id=SUBSCRIPTION_ID
)

rgr = resource_manager.resource_groups.create_or_update(
    resource_group_name=RG_NAME, 
    parameters={
        "location": "eastus",
        "tags": TAGS
    }
)

print(rgr)

ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)

ws = Workspace(
    name="grcMlWs",
    location="eastus",
    hbi_workspace=True,
    resource_group=RG_NAME,
    description="Azure ML WS with Python.",
    tags=TAGS
)

wsr = ml_client.workspaces.begin_create(ws)

print(ws)
print(wsr.status)

wsr = ml_client.workspaces.begin_delete(ws, delete_dependent_resources=True)

print(wsr)

rgr = resource_manager.resource_groups.begin_delete(RG_NAME)

print(rgr)