# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# define location of dataset
foldera = 'apples/'
folderk = 'kiwi/'
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(foldera):
	# determine class
        output = 1.0
	# load image
        photo = load_img(foldera + file, target_size=(200, 200))
	# convert to numpy array
        photo = img_to_array(photo)
        # store
        photos.append(photo)
        labels.append(output)

for file in listdir(folderk):
	# determine class
        output = 0.0
	# load image
        photo = load_img(folderk + file, target_size=(200, 200))
        # convert to numpy array
        photo = img_to_array(photo)
        # store
        photos.append(photo)
        labels.append(output)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save('apples_vs_kiwi_photos.npy', photos)
save('apples_vs_kiwi_labels.npy', labels)
