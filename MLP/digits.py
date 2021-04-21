from mlp import MLP, sigmoid, relu, softmax, identity, accuracy
import pandas as pd
import math
from autograd import grad
import autograd.numpy as np

from sklearn.datasets import load_digits
from sklearn import preprocessing

data2 = load_digits()

df2 = pd.DataFrame(data2.data)
labels = data2.target


# 3 fold cross validation
folds = 3
cum_acc = 0.0
n_rows = len(df2.index)
n_cols = len(df2.columns)
n_classes = len(np.unique(labels))
for i in range(folds):
    X_validation = df2.iloc[int(n_rows * i / folds):int(n_rows * (i + 1) / folds), :]
    y_validation = labels[int(n_rows * i / folds):int(n_rows * (i + 1) / folds)]
    if i != folds - 1:
        X_train = df2.iloc[np.r_[0:int(n_rows * i / folds), int(n_rows * (i + 1) / folds):n_rows], :]
        y_train = labels[np.r_[0:int(n_rows * i / folds), int(n_rows * (i + 1) / folds):n_rows]]
    else:
        X_train = df2.iloc[np.r_[0:int(n_rows * i / folds)], :]
        y_train = labels[np.r_[0:int(n_rows * i / folds)]]
    mlp = MLP(np.array(X_train), y_train, 1, n_cols, [20], [sigmoid], clf=True, n_classes=10)
    mlp.fit()
    y_hat = mlp.predict(np.array(X_validation))
    acc = accuracy(y_hat, y_validation)
    cum_acc += acc

    # if i == 0:
        # print(mlr.coef_[2])
    print(acc)

print(cum_acc / folds)
