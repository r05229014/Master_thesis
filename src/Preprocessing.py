import numpy as np
import sys
import random
import pickle
import os
from sklearn.preprocessing import StandardScaler
from netCDF4 import Dataset
from sklearn.decomposition import PCA

def load_alldata(path_x, path_y, features):
    TEST_SPLIT = 0.2

    u = Dataset('../input/d16_ans_u.nc')
    u = u['u'][:]
    u = u.reshape(666,33,32,32,1)
    #print(u.shape)

    v = Dataset('../input/d16_ans_v.nc')
    v = v['v'][:]
    v = v.reshape(666,33,32,32,1)
    #print(v.shape)

    w = Dataset('../input/d16_ans_w.nc')
    w = w['w'][:]
    w = w.reshape(666,33,32,32,1)
    #print(w.shape)

    qv = Dataset('../input/d16_ans_qv.nc')
    qv = qv['qv'][:]
    qv = qv.reshape(666,33,32,32,1)
    #print(qv.shape)

    th = Dataset('../input/d16_ans_th.nc')
    th = th['th'][:]
    th = th.reshape(666,33,32,32,1)
    #print(th.shape)

    mse = Dataset('../input/d16_ans_mse.nc')
    mse = mse['mse'][:]
    mse = mse.reshape(666,33,32,32,1)
    #print(mse.shape)

    cape = Dataset('../input/d16_cape.nc')
    cape = cape['cape'][:666]
    print(cape.shape)

    tmp = np.zeros((666,34,32,32))
    for t in range(666):
        for z in range(34):
            tmp[t,z,:,:] = cape[t,:,:,:]
    tmp = tmp.reshape(666,34,32,32,1)
    tmp = tmp[:,1:,:,:,:]

    mfc = Dataset('../input/d16_mfc.nc')
    mfc = mfc['mfc'][:666]

    tmp2 = np.zeros((666,34,32,32))
    for t in range(666):
        for z in range(34):
            tmp2[t,z,:,:] = mfc[t,:,:,:]
    tmp2 = tmp2.reshape(666,34,32,32,1)
    tmp2 = tmp2[:,1:,:,:,:]
    
    z = np.loadtxt('../input/z.txt')
    tmp3 = np.zeros((666,34,32,32))
    for t in range(666):
        for x in range(32):
            for y_ in range(32):
                tmp3[t,:,x,y_] = z[:]
    tmp3 = tmp3.reshape(666,34,32,32,1)
    tmp3 = tmp3[:,1:,:,:,:]
    
    if features == 6:
        X = np.concatenate((u,v,w,qv,th,mse), axis=-1)
    elif features == 9:
        X = np.concatenate((u,v,w,qv,th,mse, tmp, tmp2, tmp3), axis=-1)

    
    #wh = Dataset('../input/d16_ans_wh.nc')
    #y = wh['wh'][:]/2.5/10**6
    # load y
    scaled_wh = np.load('../input/scaled_wh.npy')
    scaled_wh = np.swapaxes(scaled_wh, 1,2)
    scaled_wh = np.swapaxes(scaled_wh, 2,3)
    #print(scaled_wh.shape, 'scsccccc')
    scaled_wh = scaled_wh.reshape(-1,33)
    #print(scaled_wh.shape, 'scsccccc')
    pca = PCA(n_components=0.95,svd_solver = 'full')
    y = pca.fit_transform(scaled_wh)
    #print(y.shape, 'yyyy')
    y = y.reshape(666,32,32,5)
    y = np.swapaxes(y, 2,3)
    y = np.swapaxes(y, 1,2)

    # shuffle
    indices = np.arange(X.shape[0])
    nb_test_samples = int(TEST_SPLIT * X.shape[0])
    random.seed(777)
    random.shuffle(indices)
    X = X[indices]
    X_train = X[nb_test_samples:]
    X_test = X[0:nb_test_samples]
    y = y[indices]
    y_train = y[nb_test_samples:]
    y_test = y[0:nb_test_samples]
    #print(indices[0:nb_test_samples][59])

    print('X_train shape is : ', X_train.shape)
    print('y_train shape is : ', y_train.shape)
    print('X_test shape is : ', X_test.shape)
    print('y_test shape is : ', y_test.shape)

    return X_train, X_test, y_train, y_test#, pca



def Preprocessing_Linear(X_train, X_test, y_train, y_test, features):
    X_train = X_train.reshape(-1,features)
    X_test = X_test.reshape(-1,features)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    return X_train, X_test, y_train, y_test 


def Preprocessing_DNN(X_train, X_test, y_train, y_test, features):
    X_train = X_train.reshape(-1,features)
    X_test = X_test.reshape(-1,features)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    sc = StandardScaler()

    # normalize
    for feature in range(features):
        X_train[:,feature:feature+1] = sc.fit_transform(X_train[:, feature:feature+1])
        X_test[:,feature:feature+1] = sc.fit_transform(X_test[:, feature:feature+1])
    
    return X_train, X_test, y_train, y_test 


def Preprocessing_RNN_vir(X_train, X_test, y_train, y_test, features):

    X_train = X_train.reshape(-1,features)
    X_test = X_test.reshape(-1,features)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    sc = StandardScaler()

    # normalize
    for feature in range(features):
        X_train[:,feature:feature+1] = sc.fit_transform(X_train[:, feature:feature+1])
        X_test[:,feature:feature+1] = sc.fit_transform(X_test[:, feature:feature+1])

    X_train = X_train.reshape(-1,33,32,32,features)
    X_test = X_test.reshape(-1,33,32,32,features)
    y_train = y_train.reshape(-1,33,32,32,1)
    y_test = y_test.reshape(-1,33,32,32,1)

    X_train = np.swapaxes(X_train, 1,3)
    X_test = np.swapaxes(X_test, 1,3)
    y_train = np.swapaxes(y_train, 1,3)
    y_test = np.swapaxes(y_test, 1,3)
    
    X_train = X_train.reshape(-1,33,features)
    X_test = X_test.reshape(-1,33,features)
    y_train = y_train.reshape(-1,33,1)
    y_test = y_test.reshape(-1,33,1)

    print('\nThis is for RNN\'s input! If we assume there are some relationship in vertical!')
    print('X_train shape is : ', X_train.shape)
    print('X_test shape is : ', X_test.shape)
    print('y_train shape is : ', y_train.shape)
    print('y_test shape is : ', y_test.shape)
    return X_train, X_test, y_train, y_test 


def pool_reflect(arr, size):
    # size is your arr's size 3*3(size=3), 5*5(size=5)...etc
    new = np.zeros((arr.shape[0], arr.shape[1]+(size-1), arr.shape[2]+(size-1), arr.shape[3]))
    for sample in range(arr.shape[0]):
        for feature in range(arr.shape[3]):
            tmp = arr[sample,:,:,feature]
            tmp_ = np.pad(tmp, int((size-1)/2), 'wrap')
            new[sample,:,:,feature] = tmp_
    return new


def cnn_type_x(arr,size):
    # size is your arr's size 3*3(size=3), 5*5(size=5)...etc
    out = np.zeros((arr.shape[0]*(arr.shape[1]-(size-1))*(arr.shape[2]-(size-1)), size, size, arr.shape[3]))

    count = 0
    for s in range(arr.shape[0]):
        for x in range(0, arr.shape[1]-(size-1)):
            for y in range(0, arr.shape[2]-(size-1)):
                out[count] = arr[s, x:x+size,  y:y+size, :]

                count  +=1  
    print("X shape : ", out.shape)
    return out


def cnn_type_y(arr):
    out = np.zeros((arr.shape[0]*arr.shape[1]*arr.shape[2], 1, 1, arr.shape[3]))
    
    count = 0
    for s in range(arr.shape[0]):
        for x in range(0 ,arr.shape[1]):
            for y in range(0, arr.shape[2]):
                out[count] = arr[s, x, y, :]  

                count += 1  
    out = np.squeeze(out)
    out = out.reshape(out.shape[0], 1)
    print('y shape : ', out.shape)
    return out


def Preprocessing_CNN(X_train, X_test, y_train ,y_test, features):
    print('\nCNN Preprocessing~~~~~~')
    X_train = X_train.reshape(-1,features)
    X_test = X_test.reshape(-1,features)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    sc = StandardScaler()

    # normalize
    for feature in range(features):
        X_train[:,feature:feature+1] = sc.fit_transform(X_train[:, feature:feature+1])
        X_test[:,feature:feature+1] = sc.fit_transform(X_test[:, feature:feature+1])

    X_train = X_train.reshape(-1,32,32,features)
    X_test = X_test.reshape(-1,32,32,features)
    y_train = y_train.reshape(-1,32,32,1)
    y_test = y_test.reshape(-1,32,32,1)

    X_train = pool_reflect(X_train, 7)
    X_test = pool_reflect(X_test, 7)
    X_train = cnn_type_x(X_train, 7)
    X_test = cnn_type_x(X_test, 7)

    print(y_train.shape, '!!!!!!!!!!')
    y_train = cnn_type_y(y_train)
    print(y_train.shape, '!!!!!!!!!!')
    y_test = cnn_type_y(y_test)

    return X_train, X_test, y_train, y_test


def CNN3D_type_x(arr,size):
    # size is your arr's size 3*3(size=3), 5*5(size=5)...etc
    out = np.zeros((arr.shape[0]*(arr.shape[2]-(size-1))*(arr.shape[3]-(size-1)), arr.shape[1], size, size, arr.shape[4]), dtype='float16')

    count = 0
    for t in range(arr.shape[0]):
        for x in range(0, arr.shape[2]-(size-1)):
            for y in range(0, arr.shape[3]-(size-1)):
                 out[count] = arr[t, :, x:x+size,  y:y+size, :]

                count  +=1  
    print("X shape : ", out.shape)
    return out


def CNN3D_type_y(arr):
    out = np.zeros((arr.shape[0]*arr.shape[2]*arr.shape[3], arr.shape[1], 1, 1, arr.shape[4]), dtype='float16')
    
    count = 0
    for t in range(arr.shape[0]):
        for x in range(0 ,arr.shape[2]):
            for y in range(0, arr.shape[3]):
                out[count] = arr[t, :, x:x+1, y:y+1, :]  

                 count += 1 
    #out = np.squeeze(out)
    #out = out.reshape(out.shape[0], 5, 1) # LRCN
    out = out.reshape(out.shape[0], 5) # 3D CNN

    print('y shape : ', out.shape)
    return out


def pool_CNN3D(arr, size):
    # size is your arr's size 3*3(size=3), 5*5(size=5)...etc
    print(arr.shape)
    new = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]+(size-1), arr.shape[3]+(size-1), arr.shape[4]), dtype = 'float16')
    for sample in range(arr.shape[0]):
        for z in range(arr.shape[1]):
            for feature in range(arr.shape[4]):
                 tmp = arr[sample,z,: , :,feature]
                tmp_ = np.pad(tmp, int((size-1)/2), 'wrap')
                new[sample,z,:,: ,feature] = tmp_
    return new


def Preprocessing_CNN3D(X_train, X_test, y_train ,y_test, features, kernel_size):
    size = kernel_size
    print('\n3D-CNN Preprocessing~~~~~~')
    X_train = X_train.reshape(-1,features)
    X_test = X_test.reshape(-1,features)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    sc = StandardScaler()

    # normalize
    for feature in range(features):
        X_train[:,feature:feature+1] = sc.fit_transform(X_train[:, feature:feature+1])
        X_test[:,feature:feature+1] = sc.fit_transform(X_test[:, feature:feature+1])

    X_train = X_train.reshape(-1,33,32,32,features)
    X_test = X_test.reshape(-1,33,32,32,features)
    y_train = y_train.reshape(-1,5,32,32,1)
    y_test = y_test.reshape(-1,5,32,32,1)

    X_train = pool_CNN3D(X_train, size)
    X_test = pool_CNN3D(X_test, size)
    #print(X_test.shape)
    X_train = CNN3D_type_x(X_train, size)
    X_test = CNN3D_type_x(X_test, size)
    #print(X_test.shape)
    y_train = CNN3D_type_y(y_train)
    y_test = CNN3D_type_y(y_test)
    
    #a = X_train[0,:,0,0,:]
    #for i in range(9):
    #    print(a[:,i])
    return X_train, X_test, y_train, y_test

#dirx = '../feature/'
#diry = '../target/'
#X_train , X_test, y_train, y_test = load_alldata(dirx, diry, 9)
#print(y_test.shape, '!!!!!!!!!!')
#t = np.array(y_test)
#print(t.shape, '!!!!!!!!!!')
#np.save('../predict/True/y_test.npy', t)
#X_train , X_test, y_train, y_test = Preprocessing_LRCN(X_train, X_test, y_train, y_test, 9, 7)
