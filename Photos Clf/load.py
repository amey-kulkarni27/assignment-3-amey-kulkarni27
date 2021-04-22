# load and confirm the shape
from numpy import load
photos = load('apples_vs_kiwi_photos.npy')
labels = load('apples_vs_kiwi_labels.npy')
print(photos.shape, labels.shape)
print(labels)
