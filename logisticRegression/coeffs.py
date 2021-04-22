import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from logisticRegression import LogisticRegression
from metrics import accuracy

data = load_breast_cancer()

df = pd.DataFrame(data=data.data, columns=data.feature_names)
df = (df - df.min()) / (df.max() - df.min())
X_train = df
df['target'] = data.target
y_train = df['target']
n_rows = len(df.index)
n_cols = len(df.columns)

lr = LogisticRegression(fit_intercept = True)
lr.fit_autograd_lr(X_train, y_train, regularisation="l1", lamda=10)

print(lr.coef_)

