import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import itertools
import warnings
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

test_img_list = glob("/datasets/kvasir/test/images/*")
test_mask_list = glob("/datasets/kvasir/test/masks/*")


print("test images ",len(test_img_list))
print("test masks ",len(test_mask_list))


SIZE_1 = 256
SIZE_2 = 256

arch =  ResUnet(input_size=[SIZE_1,SIZE_2,3])
G = arch.build_model()
G.load_weights('/resunet/resunet_cvc_augm_model/resunet_cvc_augm_weights')
print("loaded the model")
G.compile(optimizer = Adam(lr = 1e-4), loss = dice_loss, metrics = ['accuracy'])

X_tot_test = [get_image(sample_file,SIZE_1,SIZE_2) for sample_file in test_img_list]
X_test = []
for i in range(0,len(test_img_list)):
    X_test.append(X_tot_test[i])
X_test = np.array(X_test).astype(np.float32)

Y_tot_test = [get_image(sample_file,SIZE_1,SIZE_2,gray=True) for sample_file in test_mask_list]
Y_test = []
for i in range(0,len(test_img_list)):
    Y_test.append(Y_tot_test[i])
Y_test = np.array(Y_test).astype(np.float32)
           
y_pred = G.predict(X_test,batch_size=5)[:,:,:,0]
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

