import numpy as np
import pickle, sys, os, time
from sklearn.preprocessing import StandardScaler
#from keras import Model
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
#from keras.utils import multi_gpu_model
# own modile
from Preprocessing import load_alldata, Preprocessing_DNN
from config import ModelMGPU


def DNN(features):
    print("Build model!!")
    model = Sequential()
    
    model.add(Dense(256, activation = 'relu', kernel_initializer='random_uniform',bias_initializer='zeros', input_shape=(features,)))
    for i in range(10):
        #model.add(BatchNormalization())
        model.add(Dense(512, activation = 'relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
        #model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1, activation = 'linear',kernel_initializer='random_uniform',bias_initializer='zeros'))
    return model


if __name__ == '__main__':
    tStart = time.time()
    act = sys.argv[1]
    
    features = 9
    dirx = '../feature/'
    diry = '../target/'
    X_train, X_test, y_train, y_test = load_alldata(dirx, diry, features)
    X_train, X_test, y_train, y_test = Preprocessing_DNN(X_train, X_test, y_train, y_test, features)
    
    
    if act == 'train':
        # model
        model = DNN(features)
        parallel_model = ModelMGPU(model, 3)
        #opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
        parallel_model.compile(optimizer = 'adam', loss='mean_squared_error')
        print(model.summary())

        # model save path
        dirpath = "../model/DNN_1219_9f/"
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        
        # path and callbacks
        filepath= dirpath + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                                    save_best_only=True, period=1)
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        
        # training
        history = parallel_model.fit(X_train , y_train, validation_split=0.1, batch_size=1024, epochs=150, shuffle=True, callbacks = [checkpoint, earlystopper])
        
        # save history<
        history_path = '../history/DNN_1219_6f/'
        if not os.path.exists(history_path):
            os.mkdir(history_path)
        with open(history_path + 'DNN.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        y_pre = model.predict(X_test, batch_size=1024)
        #pre_dir = '../predict/DNN_1213_6f/'
        #if not os.path.exists(pre_dir):
        #    os.mkdir(pre_dir)
        #np.save(pre_dir + 'testing.npy', y_pre)

    elif act == 'test':
        model_path = sys.argv[2]
        model = load_model(model_path)
        y_pre = model.predict(X_test, batch_size=4096)
        print(y_pre.shape)
        y_pre = y_pre.reshape(-1,33,32,32)*2.5*10**6
        print(y_pre.shape)
        pre_dir = '../predict/DNN_1219_9f/'
        if not os.path.exists(pre_dir):
            os.mkdir(pre_dir)
        np.save('../predict/DNN_1219_9f/testing.npy', y_pre)
    
    else:
        print('Please type the action you want...')

    
    tEnd = time.time()
    print("It cost %f sec" %(tEnd - tStart))
