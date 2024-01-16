import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("https://azuremlexamples.blob.core.windows.net/datasets/iris.csv")

df_test = df.sample(50, random_state=33)
df_train = df[~df.index.isin(df_test.index)]

df_test.reset_index(drop=True, inplace=True)
df_train.reset_index(drop=True, inplace=True)

X = df_train.drop(columns=["species"], axis=1)
Y = df_train["species"]

lr = LogisticRegression().fit(X, Y)

yhat = lr.predict(df_test.drop(columns="species", axis=1))

acc = accuracy_score(df_test["species"], yhat)
print(f"Accuracy: {acc}")
