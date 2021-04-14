import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# Import Autograd modules here
import autograd.numpy as anp
from autograd import grad
from autograd import elementwise_grad as egrad
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit


def select_batch(X, s, size):
    '''
    Select a batch of size 'size' starting from 's' in 'X'

    Return: A vector with the same number of rows as X, set to 1 wherever we select the row in X
            Also, return end e. Used for the next starting point
    '''

    # print(s, size)
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
    sel_vec = pd.Series(choose, index = X.index, dtype = bool)
    return sel_vec, e

def sigmoid(x):
    return 0.5 * (anp.tanh(x / 2.) + 1)

def logistic_preds(X, thetas):
    return sigmoid(anp.dot(X, thetas))

def J(theta, inp, targets):
    '''
    This is the cost function we will be minimising

    param X: Contains the X values, N x k values
    param theta: Learned coefficients, k coefficients, k x 1
    param y: Actual labels, N x 1

    Return: Cost function
    '''

    preds = logistic_preds(inp, theta)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    cost = -anp.sum(anp.log(label_probs))
    return cost

class LogisticRegression():
    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None # Will be replaced by the learned coefficients, thetas
        pass

    def fit_unregularised_lr_vec(self, X, y, batch_size = 20, num_iter = 1000, lr = 0.02):
        X_copy = X.copy(deep = True)
        n_samples = len(X_copy.index)
        n_features = len(X_copy.columns)

        if(self.fit_intercept):
            X_copy.insert(0, column = "ones", value = [1 for i in range(len(X_copy.index))])
            thetas = np.array([0 for i in range(n_features + 1)], dtype = 'float64')
        else:
            thetas = np.array([0 for i in range(n_features)], dtype = 'float64')

        prev_used = -1 # Previously used sample, this indicates where to start the next batch from

        for k in range(num_iter):
            selection_vector, prev_used = select_batch(X_copy, (prev_used + 1) % n_samples, batch_size)
            X_train = X_copy[selection_vector] # Select only the batch
            y_train = y[selection_vector]
            prev_thetas = thetas
            if(self.fit_intercept):
                params = n_features + 1
            else:
                params = n_features
            Xt = X_train.transpose()
            update_vec = Xt.values.dot(expit(X_train.values.dot(thetas)) - y_train.values)
            thetas = thetas - lr * update_vec

        self.coef_ = thetas

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

        for k in range(num_iter):
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
                    thetas[j] -= (lr * (expit(np.dot(x_i, prev_thetas)) - y_i) * x_i_j)

        self.coef_ = thetas

    def fit_autograd_lr(self, X, y, batch_size = 40, num_iter = 1000, lr = 0.1):
        X_copy = X.copy(deep = True)
        n_samples = len(X_copy.index)
        n_features = len(X_copy.columns)
        if(self.fit_intercept):
            X_copy.insert(0, column = "ones", value = [1 for i in range(len(X_copy.index))])
            thetas = np.array([0 for i in range(n_features + 1)], dtype = 'float64')
        else:
            thetas = np.array([0 for i in range(n_features)], dtype = 'float64')
        prev_used = -1 # Previously used sample, this indicates where to start the next batch from
        for k in range(num_iter):
            selection_vector, prev_used = select_batch(X_copy, (prev_used + 1) % n_samples, batch_size)
            X_train = X_copy[selection_vector] # Select only the batch
            y_train = y[selection_vector]
            X_train = X_train.to_numpy(dtype = float)
            y_train = y_train.to_numpy(dtype = float)
            del_J = grad(J)
            update_vector = del_J(thetas, X_train, y_train)
            thetas -= (lr * update_vector)
        self.coef_ = thetas


    def predict(self, X):
        '''
        Function to run logistic regression on a given data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction(one of the K classes) for sample in corresponding row in X.

        '''
        X_copy = X.copy(deep = True)
        if(self.fit_intercept):
            X_copy.insert(0, column = "ones", value = [1 for i in range(len(X_copy.index))])
        thetas = self.coef_
        y = X_copy.apply(lambda row: logistic_preds(row, thetas), axis = 1)

        return np.rint(y)

    # def plot_decision_boundary()
