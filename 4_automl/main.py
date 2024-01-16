"""

Objectives:
1. Run Automl experiment

Before you run the automl, you need a:
1. Workspace;
2. Active Compute;
3. Asset Data;

# You can choose to have AutoML apply preprocessing transformations, such as:
# 
# - Missing value imputation to eliminate nulls in the training dataset.
# - Categorical encoding to convert categorical features to numeric indicators.
# - Dropping high-cardinality features, such as record IDs.
# - Feature engineering (for example, deriving individual date parts from DateTime features)

"""

import os
from azure.identity import AzureCliCredential

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import (
    MLClient,
    Input,
    automl
)


RG_NAME = "grcMlRg"
SUBSCRIPTION_ID = os.getenv("AZ_SUB_ID")

ml_client = MLClient(
    AzureCliCredential(),
    resource_group_name=RG_NAME,
    subscription_id=SUBSCRIPTION_ID,
)

training_asset_data = Input(
    path="azureml:input-data-automl:1",
    type=AssetTypes.MLTABLE
)

job_classification = automl.classification(
    # Overviews 
    compute="my-instance",
    name="automl-diabetic",
    description="automl from python sdk.",
    tags={"created_by": "gabriel_r"},

    # Ml Params
    experiment_name="diabetic-automl",
    training_data=training_asset_data,
    target_column_name="Diabetic",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True,
    test_data=None,
    validation_data=None,
)

job_classification.set_limits(
    max_trials=4,
    timeout_minutes=60,
    max_concurrent_trials=2,
    trial_timeout_minutes=15,
    enable_early_termination=True,
)

print(job_classification)

jr = ml_client.jobs.create_or_update(job_classification)

print(jr)