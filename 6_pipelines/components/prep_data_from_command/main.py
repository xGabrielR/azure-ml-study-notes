import os
from azure.identity import AzureCliCredential

from azure.ai.ml import (
    MLClient,
    Input,
    Output,
    command
)

RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")

ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)


input_data = "..."

data_prep_component = command(
    name="iris-data-preparation",
    display_name="iris-data-preparation",
    description="Preprocessing iris dataset",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",

    inputs={
        "input_data": Input(type="uri_file", path=input_data),
    },
    outputs={
        "train_output_data": Output(type="uri_file", path=""),
        "test_output_data": Output(type="uri_file", path="")
    },

    code="./src",
    command="""python prep_data.py \
                --input_data ${{inputs.input_data}}
                --train_output_data ${{outputs.train_output_data}}
                --test_output_data ${{outputs.test_output_data}}
    """
)

print(data_prep_component.component)
data_prep_component = ml_client.create_or_update(data_prep_component.component)

# Create (register) the component in your workspace
print(f"Component {data_prep_component.name} with Version {data_prep_component.version} is registered")