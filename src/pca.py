from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pickle, sys, os, time
from sklearn.preprocessing import StandardScaler
from netCDF4 import Dataset
#from keras import Model
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
#from keras.utils import multi_gpu_model
# own modile
from Preprocessing import *
from config import ModelMGPU

if __name__ == '__main__':
    features = 9
    X_train, X_test, y_train, y_test, pca, X_ori, y_ori = load_alldata(features)
    #X_train, X_test, y_train, y_test = load_alldata(dirx, diry, features)
    X_train, X_test, y_train, y_test = Preprocessing_CNN3D(X_train, X_ori, y_train, y_ori, features,7)
   
    pre = np.load('../predict/reset_data/3DCNN_pca_all.npy')
    pre = pca.inverse_transform(pre)
    print(pre.shape)
    #np.save('./ffffffffffffffffff.npy', pre)

    wh = Dataset('../input/d16_ans_wh.nc')
    y = wh['wh'][:]
    mean_y = np.mean(np.mean(np.mean(y, axis=0), axis=-1), axis=-1)
    std_y = np.std(np.std(np.std(y, axis=0), axis=-1), axis=-1)
    pre = (pre * std_y) + mean_y
    
    #y_test = y_test.reshape(133,32,32,33)
    pre = pre.reshape(666,32,32,33)
    #pre.tofile("../dat/CNN_1217_9features_no_nor.dat")
    new = np.zeros((666,33,32,32), dtype='float32')
    for i in range(666):
        for j in range(32):
            for k in range(32):
                new[i,:,j,k] = pre[i,j,k,:]
    np.save('../predict/reset_data/all_3DCNN.npy', new)
    #new.tofile("../dat/CNN_1217_9features_no_nor.dat")
