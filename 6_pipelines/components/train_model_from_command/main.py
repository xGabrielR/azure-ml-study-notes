import os
from azure.identity import AzureCliCredential
from azure.ai.ml import (
    MLClient,
    Input,
    command
)

RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")

#ml_client = MLClient(
#    AzureCliCredential(),
#    resource_group_name=RG_NAME,
#    subscription_id=SUBSCRIPTION_ID,
#)

train_path = "..."
test_path = "..."

train_model_component = command(
    name="iris-train-logistic",
    display_name="iris-train-logistic",
    description="Train Logistic Regression model to Iris preprocessed dataset",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",

    inputs={
        "train_input_data": Input(type="uri_file", path=train_path),
        "test_input_data": Input(type="uri_file", path=test_path)
    },
    
    code="./src",
    command="""python train_model.py \
                --train_input_data ${{inputs.train_input_data}}
                --test_input_data ${{inputs.test_input_data}}
    """
)

print(train_model_component.component)