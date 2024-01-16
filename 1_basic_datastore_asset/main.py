"""

Objectives:

1. Create Blob Storage.
2. Send Data to Blob Storage.
3. Create a Datastore to Link with New Blob Storange.

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
    
    az storage account create --name "grcBlobStg"
                              --resource-group "grcMlRg"
                              --location "eastus"
                              --encryption-services "blob"
                              --sku "Standard_ZRS"

    az storage container create --name "iris"
                                --resource-group "grcMlRg"
                                --account-name "grcBlobStg"
                                --auth-mode "login"

    az storage blob upload --account-name "grcBlobStg"
                           --container-name "iris"
                           --name "iris.csv"
                           --file "./iris/iris.csv"
                           --auth-mode "login"

    az storage container delete --name "iris"
                                --storage-account "grcBlobStg"
                                --yes

    az storage account delete --name "grcBlobStg"
                              --resource-group "grcMlRg"
                              --yes


For creation of Data asset and datastore

    az ml data -h
    az ml datastore -h

    az ml datastore create --file data_yml/datastore_blob.yml
                           --subscription "TODO"
                           --resource-group "grcMlRg"
                           --workspace-name "grcMlWs"

    az ml datastore delete --name "NAME-FROM-YML-OR-CUSTOM-NAME"
                           --resource-group "grcMlRg"
                           --workspace-name "grcMlWs"

    You can create versions, registry names, and more

    az ml data create --file data_yml/data_folder_cloud.yml
                      --subscription "TODO"
                      --registry-name "grcCloudData"
                      ...

    You not necessary delete the data asset.
    Instead you can archive older versions
    
    az ml data archive -h

"""

## Using Python
# pip install azure-ai-ml

import os
from datetime import datetime

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    Data,
    AzureBlobDatastore,
    AccountKeyConfiguration
)

RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")

CONTAINER_NAME = "iris"
STORAGEACCOUNTNAME = "grcBlobStg"
BLOB_ACCOUNT_URL = f"https://{STORAGEACCOUNTNAME}.blob.core.windows.net"

TAGS = {
    "created_by": "gabriel_richter",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# You need to az login before
blob_s_client = BlobServiceClient(
    credential=DefaultAzureCredential(),
    account_url=BLOB_ACCOUNT_URL
)

blob_s_client.create_container(CONTAINER_NAME)

blob_client = blob_s_client.get_blob_client(
    container=CONTAINER_NAME,
    blob="./iris/iris.csv"
)

blob_client.upload_blob("./iris/iris.csv")


abd = AzureBlobDatastore(
    name="blob_datastore",
    description="Azure Blob Datastore link.",
    account_name=STORAGEACCOUNTNAME,
    container_name=CONTAINER_NAME,
    credentials=AccountKeyConfiguration(account_key="TODO")
)


# Create Data Asset from local path 
ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)


path = "./iris/iris.csv"

dt = Data(
    path=path,
    version=1,
    name="iris_dataset_csv",
    type=AssetTypes.URI_FILE,
    description="Local Gabriel R iris dataset path.",
    tags={"created_by": "gabriel_r"}
)

mdt = ml_client.data.begin_create(dt)
print(mdt)

# Create MLTable with yaml file
# https://learn.microsoft.com/pt-br/azure/machine-learning/reference-yaml-mltable?view=azureml-api-2
#
# az ml data create --name "iris-from-https"
#                   --version 1
#                   --type mltable
#                   --path ./iris 

dt = Data(
    path="./iris",
    version=1,
    name="iris_dataset_mltable",
    type=AssetTypes.MLTABLE,
    description="Local MLTable Yaml definition.",
    tags=TAGS
)

mdt = ml_client.data.create_or_update(dt)
print(mdt)

adt = ml_client.data.archive(name="iris_dataset_csv", version=1)
print(adt)

# Clean Workspace
bcr = blob_s_client.delete_container(CONTAINER_NAME)
print(bcr)

mdt = ml_client.data.begin_delete(mdt)
print(mdt)