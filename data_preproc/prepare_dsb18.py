# # Creating Train / Val / Test folders (One time use)
import os
import numpy as np
import shutil
import random
from PIL import Image

root = '/datasets/segpc/'  #choose dataset

directories = os.listdir(root)
images_dir = [root + x + '/images/' for x in directories]
masks_dir = [root + x + '/masks/' for x in directories]

all_images = [x+y for x in images_dir for y in os.listdir(x)]
all_masks = [os.listdir(x) for x in masks_dir]

os.makedirs(root + 'images')
os.makedirs(root + 'masks')

for name in all_images:
    shutil.copy(name, root +'images/')

for idx in range(len(all_images)):
	temp = np.asarray(Image.open(masks_dir[idx] + all_masks[idx][0]))
	for im in all_masks[idx][1:]:
		image = np.asarray(Image.open(masks_dir[idx] + im))
		temp = np.add(image, temp)
	Image.fromarray(temp).save(root + 'masks/' + all_images[idx].split('/')[-1],"PNG")



"""
image = Image.open("image_path.jpg")
image.show()

images_dir = root + '/train/images' # data root path
masks_dir = root + '/train/masks' # data root path

#val_ratio = 0.10
test_ratio = 0.10



src = images_dir + '/' # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
leNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - test_ratio))])

train_FileNames_images = [src+'/'+ name for name in train_FileNames.tolist()]
#val_FileNames_images = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames_images = [src+'/' + name for name in test_FileNames.tolist()]

train_FileNames_masks = [masks_dir+'/'+ name for name in train_FileNames.tolist()]
#val_FileNames_masks = [masks_dir+'/' + name for name in val_FileNames.tolist()]
test_FileNames_masks = [masks_dir+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
#print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

os.makedirs(root + '/train2/')
#os.makedirs(root + '/val/')
#os.makedirs(root + '/test/')
os.makedirs(root + '/train2/images/')
#os.makedirs(root + '/val/images/')
os.makedirs(root + '/test/images/')
os.makedirs(root + '/train2/masks/')
#os.makedirs(root + '/val/masks/')
os.makedirs(root + '/test/masks/')

# Copy-pasting images
for name in train_FileNames_images:
    shutil.copy(name, root +'/train2/images/')
for name in train_FileNames_masks:
    shutil.copy(name.replace(".jpg","_segmentation.png"), root +'/train2/masks/')

#for name in val_FileNames_images:
#    shutil.copy(name, root +'/val/images/')
#for name in val_FileNames_masks:
#    shutil.copy(name, root +'/val/masks/')

for name in test_FileNames_images:
    shutil.copy(name, root +'/test/images/')
for name in test_FileNames_masks:
    shutil.copy(name.replace(".jpg","_segmentation.png"), root +'/test/masks/')
"""
