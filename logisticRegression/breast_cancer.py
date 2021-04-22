import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from logisticRegression import LogisticRegression
from metrics import accuracy

data = load_breast_cancer()

df = pd.DataFrame(data=data.data, columns=data.feature_names)
df = (df - df.min()) / (df.max() - df.min())
df['target'] = data.target
n_rows = len(df.index)
n_cols = len(df.columns)

# 3 fold cross validation
folds = 3
cum_acc = 0.0
for i in range(folds):
    validation = df.iloc[int(n_rows * i / folds):int(n_rows * (i + 1) / folds), :]
    if i != folds - 1:
        train = df.iloc[np.r_[0:int(n_rows * i / folds), int(n_rows * (i + 1) / folds):n_rows], :]
    else:
        train = df.iloc[np.r_[0:int(n_rows * i / folds)], :]
    X_validation = validation.iloc[:, 0:n_cols - 1] # Dataframe
    y_validation = validation.iloc[:, n_cols- 1] # Series
    X_train = train.iloc[:, 0: n_cols - 1] # Dataframe
    y_train = train.iloc[:, n_cols - 1] # Series
    lr = LogisticRegression(fit_intercept = True)
    #lr.fit_autograd_lr(X_train, y_train, regularisation="l1", lamda=0.01)
    lr.fit_unregularised_lr_vec(X_train, y_train)
    y_hat = lr.predict(X_validation)
    if i == 0:
        lr.boundary(X_validation, y_hat)

    acc = accuracy(y_hat, y_validation)
    cum_acc += acc
    # print(acc)

print(cum_acc / folds)
