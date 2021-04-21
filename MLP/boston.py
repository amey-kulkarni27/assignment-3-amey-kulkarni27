from mlp import MLP, sigmoid, relu, softmax, identity, accuracy
import pandas as pd
import math
from autograd import grad
import autograd.numpy as np

from sklearn.datasets import load_boston

data2 = load_boston()

df2 = pd.DataFrame(data2.data)
labels = data2.target

# 3 fold cross validation
folds = 3
cum_err = 0.0
n_rows = len(df2.index)
n_cols = len(df2.columns)
for i in range(folds):
    X_validation = df2.iloc[int(n_rows * i / folds):int(n_rows * (i + 1) / folds), :]
    y_validation = labels[int(n_rows * i / folds):int(n_rows * (i + 1) / folds)]
    if i != folds - 1:
        X_train = df2.iloc[np.r_[0:int(n_rows * i / folds), int(n_rows * (i + 1) / folds):n_rows], :]
        y_train = labels[np.r_[0:int(n_rows * i / folds), int(n_rows * (i + 1) / folds):n_rows]]
    else:
        X_train = df2.iloc[np.r_[0:int(n_rows * i / folds)], :]
        y_train = labels[np.r_[0:int(n_rows * i / folds)]]
    mlp = MLP(np.array(X_train), y_train, 1, n_cols, [30], [sigmoid], clf=False)
    mlp.fit()
    y_hat = mlp.predict(np.array(X_validation))
    err = rmse(y_hat, y_validation)
    cum_err += err

    # if i == 0:
        # print(mlr.coef_[2])
    print(err)

print(cum_err / folds)

