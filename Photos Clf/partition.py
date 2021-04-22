from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
foldera = 'apples/'
folderk = 'kiwi/'
dataset_home = 'dataset_apples_vs_kiwi/'
for file in listdir(foldera):
    src = foldera + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    dst = dataset_home + dst_dir + 'apples/'  + file
    copyfile(src, dst)


for file in listdir(folderk):
    src = foldera + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    dst = dataset_home + dst_dir + 'kiwi/'  + file
    copyfile(src, dst)
