import numpy as np
import matplotlib.pyplot as plt
import sys

plot = sys.argv[1]

## true
true = './True/y_test.npy'
t = np.load(true)

## DNN
#dnn_6f = './DNN_1219_6f/testing.npy'
#dnn_9f = './DNN_1219_9f/testing.npy'
#t = np.load(dnn_6f)
#
### CNN need to reconstruct and rescale !!!!!!
#cnn_6f = './CNN_6features_1219/testing.npy'
#cnn_9f = './CNN_9features_1216_2/testing.npy'
#t = np.load(cnn_9f)
#
### Linear
#linear_6f = './Linear_1214_6f/testing_6f.npy'
#linear_9f = './Linear_1214_9f/testing_9f.npy'
#t = np.load(linear_6f)*2.5*10**6
#print(t.shape)


###################
z = np.loadtxt('../z.txt')[1::]
y = t/1004
y = np.swapaxes(y,0,1)
y = y.reshape(33,-1)
percent_99 = np.percentile(y, 99, axis=1)
percent_999 = np.percentile(y, 99.9, axis=1)
percent_9995 = np.percentile(y, 99.95, axis=1)
yy = y[2]
pp = percent_99[2]
print(yy[yy>pp].shape)
dir = 'Ans/'
case = dir[:-1]


if plot == 'go':
    plt.figure(figsize=(8,9))
    plt.title("%s amplitude of $\overline{w'h'}$ in different percentile"%case, fontsize=14)
    plt.plot(percent_99, z, label='99%', linewidth=2)
    plt.plot(percent_999, z, label='99.9%', linewidth=2)
    plt.plot(percent_9995, z, label='99.95%', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.xlim(-0.2,7)
    plt.ylabel('Height [m]', fontsize=12)
    plt.xlabel(r"$\overline{w'h'}$ [K m/s]", fontsize=12)
    plt.savefig('./amp_img/%s.png'%case)
