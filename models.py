import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution3D, MaxPooling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def CNN3D(features):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(33,7,7,features)))
    model.add(Convolution3D(32, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution3D(64, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution3D(128, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    # model.add(BatchNormalization())
    model.add(Convolution3D(32, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution3D(64, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(Convolution3D(128, (2,2,2), use_bias=True, padding='SAME', strides=1, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    # model.add(BatchNormalization())
    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='linear'))
    model.compile(optimizer='Nadam', loss='mean_squared_error')
    return model
