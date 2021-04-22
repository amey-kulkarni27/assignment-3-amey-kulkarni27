from mlp import MLP, sigmoid, relu, softmax, identity, accuracy, rmse
import pandas as pd
import math
from autograd import grad
import autograd.numpy as np

from sklearn.datasets import load_boston
from sklearn import preprocessing

data2 = load_boston()
min_max_scaler = preprocessing.MinMaxScaler()

df2 = pd.DataFrame(data2.data)
df2 = min_max_scaler.fit_transform(df2)
labels = data2.target

# 3 fold cross validation
folds = 3
cum_err = 0.0
n_rows = df2.shape[0]
n_cols = df2.shape[1]
for i in range(folds):
    X_validation = df2[int(n_rows * i / folds):int(n_rows * (i + 1) / folds), :]
    y_validation = labels[int(n_rows * i / folds):int(n_rows * (i + 1) / folds)]
    if i != folds - 1:
        X_train = df2[np.r_[0:int(n_rows * i / folds), int(n_rows * (i + 1) / folds):n_rows], :]
        y_train = labels[np.r_[0:int(n_rows * i / folds), int(n_rows * (i + 1) / folds):n_rows]]
    else:
        X_train = df2[np.r_[0:int(n_rows * i / folds)], :]
        y_train = labels[np.r_[0:int(n_rows * i / folds)]]
    mlp = MLP(np.array(X_train), y_train, 1, n_cols, [10], [sigmoid], clf=False)
    mlp.fit()
    y_hat = mlp.predict(np.array(X_validation))
    err = rmse(y_hat, y_validation)
    cum_err += err

    # if i == 0:
        # print(mlr.coef_[2])
    print(err)

print(cum_err / folds)

