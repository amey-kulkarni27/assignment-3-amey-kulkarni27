import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn import preprocessing
from logisticRegression import Multi_Class_LR
from metrics import accuracy

min_max_scaler = preprocessing.MinMaxScaler()

data2 = load_digits()
df2 = pd.DataFrame(data2.data)
min_max_scaler.fit_transform(df2)
df2['target'] = data2.target


# 4 fold cross validation
folds = 4
cum_acc = 0.0
n_rows = len(df2.index)
n_cols = len(df2.columns)
n_classes = len(df2.iloc[:, -1].unique())
for i in range(folds):
    validation = df2.iloc[int(n_rows * i / folds):int(n_rows * (i + 1) / folds), :]
    if i != folds - 1:
        train = df2.iloc[np.r_[0:int(n_rows * i / folds), int(n_rows * (i + 1) / folds):n_rows], :]
    else:
        train = df2.iloc[np.r_[0:int(n_rows * i / folds)], :]
    X_validation = validation.iloc[:, 0:n_cols - 1] # Dataframe
    y_validation = validation.iloc[:, n_cols- 1] # Series
    X_train = train.iloc[:, 0: n_cols - 1] # Dataframe
    y_train = train.iloc[:, n_cols - 1] # Series
    mlr = Multi_Class_LR(fit_intercept = True, n_classes = n_classes)
    mlr.fit_unregularised_lr_vec(X_train, y_train)
    y_hat = mlr.predict(X_validation)
    acc = accuracy(y_hat, y_validation)
    cum_acc += acc

    # if i == 0:
        # print(mlr.coef_[2])
    # print(acc)

print(cum_acc / folds)
