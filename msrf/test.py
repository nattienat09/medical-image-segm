import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import itertools
import warnings
import cv2
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from math import sqrt, ceil
import tifffile as tif

import skimage.io
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Input,Average,Conv2DTranspose,SeparableConv2D,dot,UpSampling2D,Add, Flatten,Concatenate,Multiply,Conv2D, MaxPooling2D,Activation,AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D,multiply,DepthwiseConv2D,ZeroPadding2D,BatchNormalization,LeakyReLU
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

np.random.seed(123)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()

test_img_list = sorted(glob("/datasets/dsb18/test/images/*"))
test_mask_list = sorted(glob("/datasets/dsb18/test/masks/*"))


SIZE_1 = 256
SIZE_2 = 256

G = msrf(input_size=(SIZE_1,SIZE_2,3),input_size_2=(SIZE_1,SIZE_2,1))
G.load_weights('/msrf/segpc_noaugm_model/segpc_noaugm_weights')
print("loaded the model")
optimizer = get_optimizer()
G.compile(optimizer = optimizer, loss = {'x':seg_loss,'edge_out':'binary_crossentropy','pred4':seg_loss,'pred2':seg_loss},loss_weights={'x':1.,'edge_out':1.,'pred4':1. , 'pred2':1.})

X_tot_test = [get_image(sample_file,SIZE_1,SIZE_2) for sample_file in test_img_list]
X_test,edge_x_test = [],[]
for i in range(0,len(test_img_list)):
    X_test.append(X_tot_test[i][0])
    edge_x_test.append(X_tot_test[i][1])
X_test = np.array(X_test).astype(np.float32)
edge_x_test = np.array(edge_x_test).astype(np.float32)
edge_x_test  =  np.expand_dims(edge_x_test,axis=3)

Y_tot_test = [get_image(sample_file,SIZE_1,SIZE_2,gray=True) for sample_file in test_mask_list]
Y_test,edge_y_test = [],[]
for i in range(0,len(test_img_list)):
    Y_test.append(Y_tot_test[i][0])
Y_test = np.array(Y_test).astype(np.float32)
           
Y_test  =  np.expand_dims(Y_test,axis=3)


y_pred,_,_,_ = G.predict([X_test,edge_x_test],batch_size=5)
y_pred = (y_pred >=0.5).astype(int)

def compute_iou(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_true * y_pred).sum()

    #intersection = np.sum(intersection)   
    union = y_true.sum() + y_pred.sum() - intersection
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (intersection + 1e-15) / (union + 1e-15),tp/(tp+fp),tp/(tp+fn)

res = mean_dice_coef(Y_test,y_pred)
print("dice coef on test set",res)


res = compute_iou(y_pred,Y_test)
print('iou on test set is ',res[0]," precision is ",res[1]," recall is ",res[2])

