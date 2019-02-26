import numpy as np

a = np.load('../predict/DNN_1219_6f/testing.npy')
b = np.load('../predict/DNN_1219_9f/testing.npy')
a = a.astype('float32')
b = b.astype('float32')
c = np.load('../predict/Linear_1214_6f/testing_6f.npy')
d = np.load('../predict/Linear_1214_9f/testing_9f.npy')
c = c.astype('float32')
d = d.astype('float32')

e = np.load('../predict/CNN_6features_1219/testing.npy')
f = np.load('../predict/CNN_9features_1216_2/testing.npy')

a.tofile("../dat_1221/DNN_6f.dat")
b.tofile("../dat_1221/DNN_9f.dat")
c.tofile("../dat_1221/Linear_6f.dat")
d.tofile("../dat_1221/Linear_9f.dat")
e.tofile("../dat_1221/CNN_6f.dat")
f.tofile("../dat_1221/CNN_9f.dat")
