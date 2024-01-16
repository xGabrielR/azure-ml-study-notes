import os
from azure.identity import AzureCliCredential

from azure.ai.ml import MLClient, command
from azure.ai.ml.sweep import Choice, Uniform

RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")
TRAIN_DATA_PATH = "https://azuremlexamples.blob.core.windows.net/datasets/iris.csv"

TAGS = {
    "created_by": "gabriel_r"
}

ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)

python_command = """
python train_tuning.py --train_data_path ${{inputs.training_data}}
                       --solver ${{inputs.solver}}
                       --penalty ${{inputs.penalty}}
                       --c ${{inputs.c}}
""".replace("\n", "")

job = command(
    code="./src",
    display_name="train-model",
    experiment_name="train-classification-model",
    tags=TAGS,

    command=python_command,
    inputs={
        "train_data_path": TRAIN_DATA_PATH,
        "c": 1,
        "penalty": "l1",
        "solver": "saga",
    },

    # Compute
    timeout=15,
    compute="instance",
    environment="grc-conda-mlflow",
)

# Define Search Space
command_job_for_sweep = job(
    c=Uniform(1, 5),
    solver=Choice(values=["saga"]),
    penalty=Choice(values=["l1", "l2", "elasticnet"])
)

sweep_job = command_job_for_sweep.sweep(
    goal="Maximize",
    primary_metric="Accuracy",
    sampling_algorithm="bayesian",
    experiment_name="sweep-iris-logistic"
)

sweep_job.set_limits(
    timeout=10,
    max_total_trials=10,
    max_concurrent_trials=1,
)

returned_job = ml_client.create_or_update(sweep_job)
print(returned_job)