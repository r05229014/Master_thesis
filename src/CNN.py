import numpy as np
import pickle
import sys
import os 
import time
from sklearn.preprocessing import StandardScaler
#from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Dense, Dropout, Convolution3D, MaxPooling3D
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
#from keras.utils import multi_gpu_model
# own modile
from Preprocessing import *
from config import ModelMGPU


def CNN(features):
    print("Build CNN model!!")
    model = Sequential()
    model.add(Convolution3D(32, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu', input_shape=(33,7,7,features)))
    model.add(Convolution3D(64, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution3D(128, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Convolution3D(32, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution3D(64, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution3D(128, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Flatten())
    #model.add(Dense(1024, activation='relu'))
    #model.add(Dense(512, activation='relu'))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='linear'))

    #return model

    #model.add(Dense(256, activation = 'relu', kernel_initializer='random_uniform',bias_initializer='zeros', input_shape=(5,)))
    #for i in range(10):
    #    model.add(Dense(512, activation = 'relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
    #    #model.add(LeakyReLU(alpha=0.1))
    #    #model.add(BatchNormalization())
    #model.add(Dense(1, activation = 'linear',kernel_initializer='random_uniform',bias_initializer='zeros'))
    return model


if __name__ == '__main__':
    tStart = time.time()
    features = 6
    act = sys.argv[1]
    model_name = 'CNN_6features_1219'
    
    dirx = '../feature/'
    diry = '../target/'
    X_train, X_test, y_train, y_test = load_alldata(dirx, diry, 6)
    X_train, X_test, y_train, y_test = Preprocessing_LRCN(X_train, X_test, y_train, y_test, 6, 7)
    
    
    if act == 'train':
        # model
        model = CNN(features)
        print(model.summary())
        parallel_model = ModelMGPU(model, 3)
        adam = optimizers.Adam(lr=0.001/3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        parallel_model.compile(optimizer = 'adam', loss='mean_squared_error')
        print(model.summary())

        # model save path
        dirpath = "../model/%s/" %model_name
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        
        # path and callbacks
        filepath= dirpath + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                                    save_best_only=True, period=1)
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        
        # training
        history = parallel_model.fit(X_train , y_train, validation_split=0.1, batch_size=1024, epochs=150, shuffle=True, callbacks = [checkpoint, earlystopper])
        
        # save history
        history_path = '../history/%s/' %model_name
        if not os.path.exists(history_path):
            os.mkdir(history_path)
        with open(history_path + 'CNN.pkl', 'wb') as f:
            pickle.dump(history.history, f)

    elif act == 'test':
        model_path = sys.argv[2]
        model = load_model(model_path)
        y_pre = model.predict(X_test, batch_size=1024)
        pre_dir = '../predict/%s/'%model_name
        if not os.path.exists(pre_dir):
            os.mkdir(pre_dir)
        np.save(pre_dir + 'testing_pca.npy', y_pre)
    
    else:
        print('Please type the action you want...')

    
    tEnd = time.time()
    print("It cost %f sec" %(tEnd - tStart))
