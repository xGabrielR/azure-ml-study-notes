"""
In Azure Machine Learning, a pipeline is a
workflow of machine learning tasks in which each task is defined as a component.
"""
import os
from azure.identity import AzureCliCredential

from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.constants import AssetTypes

from azure.ai.ml.dsl import pipeline

from azure.ai.ml.entities import JobSchedule
from azure.ai.ml.entities import RecurrenceTrigger

RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")
TAGS = {
    "created_by": "gabriel_r"
}

ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)

# Data Assets Paths
dataset_path = "azureml:iris_dataset:1"

# Ymls Components Path
prep_yml = "components/prep_data_from_yml_file/prep.yml"
train_yml = "components/train_model_from_yml_file/iris_model_logistic.yml"

# Load Components
loaded_component_prep_data = load_component(prep_yml)
loaded_component_train_model = load_component(train_yml)

@pipeline(
    name="iris_logistic_pipeline",
    description="iris logistic pipeline",
    experiment_name="iris_logistic_pipeline",

    compute="aml-cluster",

    version="1",
    tags=TAGS
)
def pipeline_prep_train_iris(pipeline_job_input):

    prep_data_job = loaded_component_prep_data(
        input_data=pipeline_job_input
    )

    train_model_job = loaded_component_train_model(
        train_input_data=prep_data_job.outputs.train_output_data,
        test_input_data=prep_data_job.outputs.test_output_data
    )

    return {
        "pipeline_job_train_data": prep_data_job.outputs.train_output_data,
        "pipeline_job_test_data": prep_data_job.outputs.test_output_data
    }

job = pipeline_prep_train_iris(
    Input(type=AssetTypes.URI_FILE, path=dataset_path)
)

print(job)

job = ml_client.job.create_or_update(job)
print(job.studio_url)


schedule_name = "iris_every_hour"

recurrence_trigger = RecurrenceTrigger(
    frequency="hour",
    interval=1,
)

job_schedule = JobSchedule(
    name=schedule_name, trigger=recurrence_trigger, create_job=job
)

job_schedule = ml_client.schedules.begin_create_or_update(
    schedule=job_schedule
).result()

ml_client.schedules.begin_disable(name=schedule_name).result()

ml_client.schedules.begin_delete(name=schedule_name).result()