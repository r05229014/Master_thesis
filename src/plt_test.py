import numpy as np
import sys
import random
import pickle
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pre = np.load('../predict/CNN_9features_1216_2/testing.npy')

scaled_wh = np.load('./scaled_wh.npy')
scaled_wh = np.swapaxes(scaled_wh, 1,2)
scaled_wh = np.swapaxes(scaled_wh, 2,3)
scaled_wh = scaled_wh.reshape(-1,33)
pca = PCA(n_components=0.95,svd_solver = 'full')
y = pca.fit_transform(scaled_wh)
y = y.reshape(666,32,32,5)
#y = np.swapaxes(y, 2,3)
#y = np.swapaxes(y, 1,2)

# shuffle
indices = np.arange(y.shape[0])
nb_test_samples = int(0.2 * y.shape[0])
random.seed(777)
random.shuffle(indices)
y = y[indices]
y_train = y[nb_test_samples:]
y_test = y[0:nb_test_samples]
y_test = y_test.reshape(-1,5)


p = y_test - pre
print(pre.shape)
print(y_test.shape)
print(p.shape)
z = [0,1,2,3,4]
for i in range(100):
    plt.figure(i)
    plt.plot(y_test[i],z, label='y_test')
    plt.plot(pre[i],z, label='pre')
    plt.savefig('./img2/%s.png' %i)
    plt.close()
