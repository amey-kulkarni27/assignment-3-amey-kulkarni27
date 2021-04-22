import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn import preprocessing
from logisticRegression import Multi_Class_LR
from metrics import accuracy
from sklearn.decomposition import PCA # as sklearnPCA
import matplotlib.pyplot as plt

min_max_scaler = preprocessing.MinMaxScaler()

data2 = load_digits()
df2 = pd.DataFrame(data2.data)
min_max_scaler.fit_transform(df2)
df2['target'] = data2.target


pca = PCA(n_components=2)
y_pca = pca.fit_transform(df2)

plt.scatter(y_pca[:, 0], y_pca[:, 1], c=df2['target'])
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("PCA analysis")
plt.show()
