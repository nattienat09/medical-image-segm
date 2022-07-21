import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import itertools
import warnings
import random
from math import sqrt, ceil
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import tifffile as tif

import skimage.io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
from sklearn.utils import shuffle

from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Input,Average,Conv2DTranspose,SeparableConv2D,dot,UpSampling2D,Add, Flatten,Concatenate,Multiply,Conv2D, MaxPooling2D,Activation,AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D,multiply,DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D,BatchNormalization,LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate ,Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *

from loss import *
from utils import *
from model import *


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()


smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

np.random.seed(42)

train_path = "/datasets/cells_dataset_noaugm/train/"
valid_path = "/datasets/cells_dataset_noaugm/val/"

## Training
train_x = sorted(glob(os.path.join(train_path, "images", "*")))
train_y = sorted(glob(os.path.join(train_path, "masks", "*")))

    ## Validation
valid_x = sorted(glob(os.path.join(valid_path, "images", "*")))
valid_y = sorted(glob(os.path.join(valid_path, "masks", "*")))


print("final training set length",len(train_x),len(train_y))
print("final valid set length",len(valid_x),len(valid_y))

SIZE_1 = 256
SIZE_2 = 256

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

X_tot_val = [get_image(sample_file,SIZE_1,SIZE_2) for sample_file in valid_x]
X_val = []
print(len(X_tot_val))

for i in range(0,len(valid_x)):
    X_val.append(X_tot_val[i])
X_val = np.array(X_val).astype(np.float32)

Y_tot_val = [get_image(sample_file,SIZE_1,SIZE_2,gray=True) for sample_file in valid_y]
Y_val = []
print(len(X_tot_val))

for i in range(0,len(valid_y)):
    Y_val.append(Y_tot_val[i])
Y_val = np.array(Y_val).astype(np.float32)
#Y_val  =  np.expand_dims(Y_val,axis=3)

def train(epochs, batch_size, model_save_dir):
    
    batch_count = int(len(train_x) / batch_size)
    max_val_dice= -1
    arch =  ResUnet(input_size=[SIZE_1,SIZE_2,3])
    G = arch.build_model()
    G.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss, metrics = ['accuracy'])
    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' %e ,' out of ',epochs, '-'*15)
        #sp startpoint
        for sp in range(0,batch_count,1):
            if (sp+1)*batch_size>len(train_x):
                batch_end = len(train_x)
            else:
                batch_end = (sp+1)*batch_size
            X_batch_list = train_x[(sp*batch_size):batch_end]
            Y_batch_list = train_y[(sp*batch_size):batch_end]
            X_tot = [get_image(sample_file,SIZE_1,SIZE_2) for sample_file in X_batch_list]
            X_batch= []
            for i in range(0,batch_size):
                X_batch.append(X_tot[i])
            X_batch = np.array(X_batch).astype(np.float32)
            Y_tot = [get_image(sample_file,SIZE_1,SIZE_2, gray=True) for sample_file in Y_batch_list]
            Y_batch = []
            for i in range(0,batch_size):
                Y_batch.append(Y_tot[i])
            Y_batch = np.array(Y_batch).astype(np.float32)
            G.train_on_batch(X_batch,Y_batch)
        y_pred = G.predict(X_val,batch_size=5)[:,:,:,0]
        y_pred = (y_pred >=0.5).astype(int)
        res = mean_dice_coef(Y_val,y_pred)
        if(res > max_val_dice):
            max_val_dice = res
            G.save(model_save_dir + 'resunet_cells_noaugm_retry')
            G.save_weights(model_save_dir + 'resunet_cells_noaugm_retry_weights')
            print('New Val_Dice HighScore',res)            
            
model_save_dir = '/resunet/resunet_cells_noaugm_retry_model/'
train(200,8,model_save_dir)
