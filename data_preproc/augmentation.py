from skimage import io
import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen_images = ImageDataGenerator(        
        rotation_range = 40,
        shear_range = 0.3,
        zoom_range = [0.7,1.0],
        horizontal_flip = True,
        vertical_flip = True)

datagen_masks = ImageDataGenerator(        
        rotation_range = 40,
        shear_range = 0.3,
        zoom_range = [0.7,1.0],
        horizontal_flip = True,
        vertical_flip = True)


image_directory = '/datasets/dsb18/train/images/'
mask_directory = '/datasets/dsb18/train/masks/'
SIZE1 = 256
SIZE2 = 256

image_dataset = []
my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
	#if (image_name.split('.')[1] == 'jpg'):
	image = io.imread(image_directory + image_name)
	print("read image ",image_name, i)
	image = Image.fromarray(image)
	image = image.resize((SIZE1,SIZE2))
	image_dataset.append(np.array(image))
x = np.array(image_dataset)
print(x.shape)

mask_dataset = []
my_masks = os.listdir(mask_directory)
for i, image_name in enumerate(my_masks):
	#if (image_name.split('.')[1] == 'jpg'):
	image = io.imread(mask_directory + image_name)
	print("read mask ",image_name, i)
	image = Image.fromarray(image, 'L')
	image = image.resize((SIZE1,SIZE2))
	mask_dataset.append(np.array(image))
y = np.array(mask_dataset)
print(y.shape)
y = np.expand_dims(y,axis=3)

i = 0
for batch in datagen_images.flow(x, batch_size=8,
                          save_to_dir= r'/datasets/dsb18_augmented/images/',
                          save_prefix='dr',
                          save_format='png', seed = 538):    
	print(i)
	i += 1    
	if i > 30:        
		break

i = 0
for batch in datagen_masks.flow(y, batch_size=8,
                          save_to_dir= r'/datasets/dsb18_augmented/masks/',
                          save_prefix='dr',
                          save_format='png', seed = 538):    
	print(i)
	i += 1    
	if i > 30:        
		break
