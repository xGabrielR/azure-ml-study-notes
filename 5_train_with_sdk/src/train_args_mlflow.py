import mlflow
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def get_args() -> dict:
    p = argparse.ArgumentParser()
    
    p.add_argument("--train_data_path", dest="training_data", type=str)

    args = p.parse_args()

    return args


def read_data(
    path: str
) -> pd.DataFrame:
    df = pd.read_csv(path)

    return df


def split_data(
    df: pd.DataFrame,
    sample_size: int = 50,
    random_state: int = 33
):
    df_test = df.sample(sample_size, random_state=random_state)
    df_train = df[~df.index.isin(df_test.index)]

    df_test.reset_index(drop=True, inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    return df_test, df_train


def train_model(
    df_train: pd.DataFrame,
    target: str = "species"
) -> LogisticRegression:
    
    mlflow.autolog()

    X = df_train.drop(columns=[target], axis=1)
    Y = df_train[target]

    lr = LogisticRegression().fit(X, Y)

    return lr


def make_inference(
    model,
    df_test: pd.DataFrame,
    drop_target: bool = True,
):
    if drop_target:
        yhat = model.predict(df_test.drop(columns="species", axis=1))

    else:
        yhat = model.predict(df_test)

    return yhat


def evaluate_inference(
    ytrue,
    yhat
) -> dict:
    acc = accuracy_score(ytrue, yhat)

    print(f"Accuracy: {acc}")

    return {"acc": acc}


if __name__ == "__main__":
    args = get_args()

    df = read_data(path=args.training_data)

    df_test, df_train = split_data(df=df)

    model = train_model(df_train=df_train)

    yhat = make_inference(model=model, df_test=df_test)

    metrics = evaluate_inference(df_test["species"], yhat)