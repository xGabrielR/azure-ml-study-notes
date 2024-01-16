"""
Components allow you to create reusable scripts that can easily be shared across users
within the same Azure Machine Learning workspace.
You can also use components to build an Azure Machine Learning pipeline.

Source:
- https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-component-command?view=azureml-api-2
- https://learn.microsoft.com/en-us/training/modules/run-pipelines-azure-machine-learning/2-create-components

"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", dest="input_data", type=str)
    parser.add_argument("--train_output_data", dest="output_data", type=str)
    parser.add_argument("--test_output_data", dest="output_data", type=str)
    # parser.add_argument("--artifacts_output", dest="output_data", type=str)

    args = parser.parse_args()

    return args

def read_data(args) -> pd.DataFrame:

    df = pd.read_csv(args.input_data)

    return df

def data_preprocessing(
    df: pd.DataFrame
) -> pd.DataFrame:
    
    df = df.dropna()

    return df

def split_data(df: pd.DataFrame, split_size: int = 50):
    
    df_test = df.sample(split_size)
    df_train = df[~df.index.isin(df_test.index)]

    df_test.reset_index(inplace=True, drop=True)
    df_train.reset_index(inplace=True, drop=True)

    return df_train, df_test
    
def data_reescaling(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> pd.DataFrame:
    num_cols = ["sepal.length", "sepal.width",
                "petal.length", "petal.width"]

    scaler = MinMaxScaler()

    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols] = scaler.transform(df_test[num_cols])

    return df_train, df_test

def data_store(
    args,
    df_train,
    df_test
) -> None:
    
    df_train.to_csv(
        (Path(args.output_data) / "train_data.csv"), 
        index = False
    )

    df_test.to_csv(
        (Path(args.output_data) / "test_data.csv"), 
        index = False
    )

    return None

if __name__ == "__main__":
    args = get_args()

    df = read_data(args)
    df = data_preprocessing(df)

    df_train, df_test = split_data(df, split_size=50)
    df_train, df_test = data_reescaling(df_train, df_test)
    
    data_store(args, df_train, df_test)
