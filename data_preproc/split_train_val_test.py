# # Creating Train / Val / Test folders (One time use)
import os
import numpy as np
import shutil
import random
from PIL import Image

root = '/datasets/etis/' #choose dataset

images_dir = root + 'images/'
masks_dir = root + 'masks/'

val_ratio = 0.10
test_ratio = 0.10

src = images_dir  # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)* (1 - (val_ratio + test_ratio))),
                                                           int(len(allFileNames)* (1 - test_ratio))])

train_FileNames_images = [src+ name for name in train_FileNames.tolist()]
val_FileNames_images = [src + name for name in val_FileNames.tolist()]
test_FileNames_images = [src + name for name in test_FileNames.tolist()]

train_FileNames_masks = [masks_dir + "p" + name for name in train_FileNames.tolist()]
val_FileNames_masks = [masks_dir + "p" + name for name in val_FileNames.tolist()]
test_FileNames_masks = [masks_dir + "p" + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

os.makedirs(root + '/train/')
os.makedirs(root + '/val/')
os.makedirs(root + '/test/')
os.makedirs(root + '/train/images/')
os.makedirs(root + '/val/images/')
os.makedirs(root + '/test/images/')
os.makedirs(root + '/train/masks/')
os.makedirs(root + '/val/masks/')
os.makedirs(root + '/test/masks/')

# Copy-pasting images
for name in train_FileNames_images:
    shutil.copy(name, root +'/train/images/')
for name in train_FileNames_masks:
    shutil.copy(name, root +'/train/masks/')

for name in val_FileNames_images:
    shutil.copy(name, root +'/val/images/')
for name in val_FileNames_masks:
    shutil.copy(name, root +'/val/masks/')

for name in test_FileNames_images:
    shutil.copy(name, root +'/test/images/')
for name in test_FileNames_masks:
    shutil.copy(name, root +'/test/masks/')
