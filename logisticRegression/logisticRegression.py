import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# Import Autograd modules here
import autograd.numpy as anp
from autograd import grad
from autograd import elementwise_grad as egrad
from mpl_toolkits.mplot3d import Axes3D


def select_batch(X, s, size):
    '''
    Select a batch of size 'size' starting from 's' in 'X'

    Return: A vector with the same number of rows as X, set to 1 wherever we select the row in X
            Also, return end e. Used for the next starting point
    '''

    n = len(X.index)
    e = (s + size - 1) % n
    choose = [0 for i in range(n)]
    if e >= s:
        for i in range(s, e + 1):
            choose[i] = 1
    else:
        for i in range(e, n):
            choose[i] = 1
        for i in range(s + 1):
            choose[i] = 1
    sel_vec = pd.Series(choose, dtype = bool).reindex_like(X)
    return sel_vec, e


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def J(X, theta, y):
    '''
    This is the cost function we will be minimising
    
    param X: Contains the X values, N x k values
    param theta: Learned coefficients, k coefficients, k x 1
    param y: Actual labels, N x 1

    Return: Cost function
    '''

    cost = 0.0
    for i in range(N):
        x_i = X.iloc[i, :]
        y_i = y.iloc[i]
        Xtheta = np.dot(x_i, theta)
        cost = cost - (y_i * np.log(sigmoid(Xtheta))) - ((1 - y_i) * np.log(1 - sigmoid(Xtheta)))
    return cost


class LogisticRegression():
    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None # Will be replaced by the learned coefficients, thetas
        pass

    def fit_unregularised_lr(self, X, y, batch_size = 1, num_iter = 100, lr = 0.01):
        X_copy = X.copy(deep = True)
        n_samples = len(X_copy.index)
        n_features = len(X_copy.columns)
        if(self.fit_intercept):
            X_copy.insert(0, column = "ones", value = [1 for i in range(len(X_copy.index))])
            thetas = np.array([0 for i in range(n_features + 1)], dtype = 'float64')
        else:
            thetas = np.array([0 for i in range(n_features)], dtype = 'float64')
        prev_used = -1 # Previously used sample, this indicates where to start the next batch from
        for k in range(n_iter):
            selection_vector, prev_used = select_batch(X_copy, (prev_used + 1) % n_samples, batch_size)
            X_train = X_copy[selection_vector] # Select only the batch
            y_train = y[selection_vector]
            prev_thetas = thetas
            if(self.fit_intercept):
                params = n_features + 1
            else:
                params = n_features
            for j in range(params):
                for i in range(batch_size):
                    x_i = X_train.iloc[i, :]
                    y_i = y_train.iloc[i]
                    x_i_j = X_train.iloc[i, j]
                    thetas[j] -= (lr * (sigmoid(np.dot(x_i, prev_thetas)) - y_i) * x_i_j)
        self.coef_ = thetas

    def fit_autograd_lr(self, X, y, batch_size = 1, num_iter = 100, lr = 0.01):
        X_copy = X.copy(deep = True)
        n_samples = len(X_copy.index)
        n_features = len(X_copy.columns)
        if(self.fit_intercept):
            X_copy.insert(0, column = "ones", value = [1 for i in range(len(X_copy.index))])
            thetas = np.array([0 for i in range(n_features + 1)], dtype = 'float64')
        else:
            thetas = np.array([0 for i in range(n_features)], dtype = 'float64')
        prev_used = -1 # Previously used sample, this indicates where to start the next batch from
        for k in range(n_iter):
            selection_vector, prev_used = select_batch(X_copy, (prev_used + 1) % n_samples, batch_size)
            X_train = X_copy[selection_vector] # Select only the batch
            y_train = y[selection_vector]
            del_J = egrad(J)
            update_vector = del_J(thetas, X_train, y_train)
            thetas[j] -= (lr * (sigmoid(np.dot(x_i, prev_thetas)) - y_i) * x_i_j)
        self.coef_ = thetas
