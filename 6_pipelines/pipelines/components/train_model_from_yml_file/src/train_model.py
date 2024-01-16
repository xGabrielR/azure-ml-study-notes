"""
You can improve with mlflow and register a model for future deployment.
lets make the "pattern" simple
"""

import os
import argparse
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


def get_args():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input_data", type=str, help="path to train data")
    parser.add_argument("--test_input_data", type=str, help="path to test data")
    
    #parser.add_argument("--model", type=str, help="path to model file")
    #parser.add_argument("--registered_model_name", type=str, help="model name")

    parser.add_argument("--c", required=False, default=1, type=int)
    

    args = parser.parse_args()

    return args

def get_data(path):
    df = pd.read_csv(path)

    y = df["variety"]
    x = df.drop(columns=["variety"], axis=1)

    return x, y

def train_model(
    args,
    x, y
):
    model = LogisticRegression(C=args.c).fit(x, y)
    
    return model

def get_predictions(
    model, x
):
    yhat = model.predict(x)

    return yhat

def evaluate_model(
    ytrue, yhat
):
    
    print(classification_report(ytrue, yhat))

if __name__ == "__main__":
    args = get_args()

    x_train, y_train = get_data(args.train_input_data)
    x_test, y_test = get_data(args.test_input_data)

    model = train_model(args, x_train, y_train)

    yhat = get_predictions(model, x_test)

    evaluate_model(y_test, yhat)
