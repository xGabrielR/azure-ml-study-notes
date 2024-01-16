"""

Objectives:

1. Create a Compute Instance to a existing Workspace

Using CLI:

    az ml compute create -h

    az ml compute create --resource-group "TODO"
                         --workspace-name "TODO"
                         --location "eastus"
                         --name "grcComputeInstance"
                         --description "Compute Instance from CLI."
                         --type "ComputeInstance"
                         --size "STANDARD_DS3_v2"
                         --admin-password ""
                         --admin-username ""
                         --tags "created_by=gabriel_r"

    az ml compute create --resource-group "TODO"
                         --workspace-name "TODO"
                         --location "eastus"
                         --name "grcComputeCluster"
                         --description "Compute Cluster from CLI."
                         --type "AmlCluster"
                         --size "STANDARD_DS3_v2"
                         --min-instances 0
                         --max-instances 2
                         --idle-time-before-scale-down 120
                         --admin-password ""
                         --admin-username ""
                         --tags "created_by=gabriel_r"

    az ml compute stop --name "grcComputeInstance"
                       --resource-group "TODO"
                       --workspace-name "TODO"

    az ml compute delete --name "grcComputeInstance"
                         --resource-group "TODO"
                         --workspace-name "TODO"
                         --yes

    az ml compute delete --name "grcComputeCluster"
                         --resource-group "TODO"
                         --workspace-name "TODO"
                         --yes
                         
"""

import os
from azure.ai.ml import MLClient, command
from azure.identity import AzureCliCredential
from azure.ai.ml.entities import (
    ComputeInstance,
    AmlCompute,
    ComputeInstanceSshSettings
)

RG_NAME = os.getenv("RG_NAME")
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")

ci = ComputeInstance(
    location="eastus",
    name="grcComputeInstance",
    description="Compute Instance from Python.",

    size="STANDARD_DS3_v2",
    idle_time_before_shutdown=5,
    
    tags={
        "created_by": "gabriel",
        "engine": "python"
    }
)

ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)

cir = ml_client.compute.begin_create_or_update(ci)
print(cir)

cir = ml_client.compute.begin_stop("grcComputeInstance")
print(cir)

cir = ml_client.compute.begin_delete("grcComputeInstance")
print(cir)

cl = AmlCompute(
    location="eastus",
    name="grcComputeCluster",
    description="Compute Cluster from Python.",

    size="STANDARD_DS3_v2",
    min_instances=0,
    max_instances=2,
    tags={
        "created_by": "gabriel_r",
        "engine": "python"
    }
)

ccr = ml_client.compute.begin_create_or_update(cl)
print(ccr)

cmd = command(
    code="./src",
    command="python train.py",
    display_name="train-model",
    display_name="iris-logistic-regression-train",
    experiment_name="iris-logisticregression-training",

    # Compute
    timeout=15,
    compute="grcComputeCluster",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    environment_variables={"PATH": "https://azuremlexamples.blob.core.windows.net/datasets/iris.csv"}
)

jobr = ml_client.create_or_update(cmd)
print(f"Model At: ")
print(jobr.studio_url)