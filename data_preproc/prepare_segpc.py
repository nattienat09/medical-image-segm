# # Creating Train / Val / Test folders (One time use)
import os
import numpy as np
import shutil
import random
from PIL import Image
from glob import glob

root = '/datasets/segpc/'  #choose dataset

images = sorted(glob(os.path.join(root,"images","*.bmp")))
masks = []
bad_images = []
for i in images:
    mask = glob(os.path.join(root,"masks",i.split('/')[-1].split('.')[0]) + "_*.bmp")
    if len(mask) == 0:
        bad_images.append(i)
    masks.append(mask)

os.makedirs(root + 'new_masks',exist_ok=True)


for idx in range(len(images)):
    print(idx)
    if images[idx] in bad_images:
        continue
    temp = np.asarray(Image.open(masks[idx][0]))
    if len(temp.shape) > 2:
        temp = temp[:,:,0] 
    temp = (temp==40).astype(np.uint8) # timi 40 einai o pyrhnas
    if len(masks[idx]) == 1:
        Image.fromarray(255*temp).save(root + 'new_masks/' + images[idx].split('/')[-1])
    else:    
        for mask in masks[idx][1:]:
            image = np.asarray(Image.open(mask))
            if len(image.shape) > 2:
                image = image[:,:,0] 
            image = (image==40).astype(np.uint8) # timi 40 einai o pyrhnas
            temp = np.add(image, temp)
            Image.fromarray(255*temp).save(root + 'new_masks/' + images[idx].split('/')[-1])


print(bad_images)


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
