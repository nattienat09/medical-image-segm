import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
from glob import glob
import warnings
np.random.seed(123)


def get_image(image_path, image_size_wight, image_size_height,gray=False):
    # load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    if gray==True:
        img = img.convert('L')
    # center crop
    img_center_crop = img
    # resize
    img_resized = img_center_crop.resize((image_size_height, image_size_wight), Image.ANTIALIAS)
    
    img_array = np.asarray(img_resized).astype(np.float32)/255.0
    #print(img_array)
    if gray==True:
        img_array=(img_array >=0.5).astype(int)
    img.close()
    return img_array

