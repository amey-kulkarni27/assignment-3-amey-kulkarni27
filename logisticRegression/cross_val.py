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
l1_acc = 0.0
l1_lam = None
l2_acc = 0.0
l2_lam = None
for reg in ["l1", "l2"]:
    for lamda in np.arange(0.01, 0.1, 0.01):
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
            lr.fit_autograd_lr(X_train, y_train, regularisation=reg, lamda=lamda)
            y_hat = lr.predict(X_validation)

            acc = accuracy(y_hat, y_validation)
            cum_acc += acc
        cum_acc /= 3
        print(reg, lamda, cum_acc)
        if reg == "l1":
            if cum_acc > l1_acc:
                l1_acc = cum_acc
                l1_lam = lamda
        if reg == "l2":
            if cum_acc > l2_acc:
                l2_acc = cum_acc
                l2_lam = lamda

#print(cum_acc / folds)
print("Best l1 lambda:", l1_lam)
print("Best l2 lambda:", l2_lam)

