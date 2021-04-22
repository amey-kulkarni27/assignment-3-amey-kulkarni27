import autograd.numpy as np
import pandas as pd
import numpy
import math
from autograd import grad
from autograd import elementwise_grad as egrad

from sklearn.datasets import load_digits
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


def rmse(y_hat, y):
    return np.sqrt(np.mean((y-y_hat)**2))

def accuracy(y_hat, y):
    assert(y_hat.size == y.size)

    return ((y_hat == y).sum() / len(y))


def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def relu(x):
    return np.where(x > 0, x, 0)

def softmax(x):
    '''
    outputs: n x n_classes

    return set of probabilities, also n x n_classes
    '''
    Af = np.subtract(x.T, np.max(x, axis = 1)).T
    expAf = np.exp(Af)
    denAf = np.sum(expAf, axis=1)[:, np.newaxis]
    return expAf / denAf

# x = np.array([
#               [0, 0],
#               [0, 1],
#               [1, 0],
#               [1, 1]
# ])
# y = np.array([0, 1, 1, 0])

class MLP():
    def __init__(self, X, y, num_layers, inp_dim, hid_dims, activations, clf=True, n_classes=None):
        self.num_layers = num_layers
        self.clf = clf
        self.inp_dim = inp_dim
        print(inp_dim)
        self.dims = hid_dims
        self.dims.insert(0, inp_dim)
        if self.clf:
            self.dims.append(n_classes)
        else:
            self.dims.append(1)
        self.W = [np.random.normal(size=(self.dims[i], self.dims[i + 1])) for i in range(num_layers+1)]
        self.b = [np.random.normal(size=self.dims[i]) for i in range(1, num_layers+2)]
        self.Z = [None] * (num_layers + 1)
        self.A = [None] * (num_layers + 1)
        self.g = activations
        if self.clf:
            self.g.append(softmax)
        else:
            self.g.append(identity)
        self.X = X
        self.y = y


    def forward(self, W, b):
        '''

        X is n x d, n is the number of sample and d is the number of features

        '''
        prev = self.X
        for l in range(self.num_layers+1):
            self.Z[l] = np.matmul(prev, W[l]) + b[l]
            self.A[l] = self.g[l](self.Z[l])
            prev = self.A[l]

        # For classification
        preds = self.A[-1] # This is the softmax of the final layer
        return preds


    def backward(self, lr):
        
        if self.clf:
            del_costW = grad(self.cross_entropy, 0)
            del_costb = grad(self.cross_entropy, 1)
        else:
            del_costW = grad(self.rmse, 0)
            del_costb = grad(self.rmse, 1)

        update_vecW = del_costW(self.W, self.b)
        update_vecb = del_costb(self.W, self.b)
        for i in range(self.num_layers + 1):
            if self.clf:
                self.W[i] -= lr * update_vecW[i] / len(self.X)
                self.b[i] -= lr * update_vecb[i] / len(self.X)
            else:
                self.W[i] -= lr * update_vecW[i]
                self.b[i] -= lr * update_vecb[i]



    def cross_entropy(self, W, b):
        '''
        Af: n x n_classes
        labels: n x 1
        '''

        loss = 0.0
        preds = self.forward(W, b)

        labels = self.y
        for i in range(len(labels)):
            l = labels[i]
            p = preds[i][l]
            loss -= np.log(p + np.exp(-10))

        # print(loss)
        return loss
    
    def rmse(self, W, b):
        predictions = self.forward(W, b)
        targets = self.y
        print(np.sqrt(np.mean((predictions-targets)**2)))
        return np.sqrt(np.mean((predictions-targets)**2))

    def fit(self, n_iter=2, lr=5):
        for i in range(n_iter):
            self.backward(lr)


    def predict(self, inp):
        self.X = inp
        probs = self.forward(self.W, self.b)
        y_preds = np.argmax(probs, axis=1)

        return y_preds


