import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from data_generator import DataGenerator
from models import CNN3D
from tensorflow.keras.models import load_model


def _parse_data_function(example_proto):
    data_feature_description = {
        'feature' : tf.io.FixedLenFeature([], tf.string),
        'label' : tf.io.FixedLenFeature([], tf.string)
    }

    # Parse the input tf.train.Example proto using the dictionary above.
    features = tf.io.parse_single_example(example_proto, data_feature_description)
    data = tf.io.parse_tensor(features['feature'], "float") 
    label = tf.io.parse_tensor(features['label'], "double")
    data.set_shape([33,7,7,9])
    label.set_shape([5,])
    return data, label

def get_dataset(dataset):
    # dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE = 1024
    scaler_path = 'model_save/test1/pickles/scaler.pkl'
    pca_path = 'model_save/test1/pickles/pca.pkl'

	# get tf dataset
    raw_dataset = tf.data.TFRecordDataset('dataset/test_dataset.tfrecords')
    data_feature_description = {
            'feature' : tf.io.FixedLenFeature([], tf.string),
            'label' : tf.io.FixedLenFeature([], tf.string)
    }
    full_dataset = raw_dataset.map(_parse_data_function, num_parallel_calls=AUTOTUNE)
    full_dataset = get_dataset(full_dataset)

    # get model
    model = CNN3D(9)
    model.load_weights('models_save/test1/CNN3D_027-0.00154894.hdf5')
    pca = pickle.load(open(pca_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))

    # predict
    # Predict, Y = np.zeros((262, 32, 32, 33)), np.zeros((262, 32, 32, 33))
    X = np.zeros((262, 32, 32, 33))
    for i, (X, y) in enumerate(full_dataset):
        print(X.shape)
        # predict = model.predict(X, batch_size=1024)
        # predict, y = pca.inverse_transform(predict), pca.inverse_transform(y)
        # predict, y = scaler.inverse_transform(predict), scaler.inverse_transform(y)
        # Predict[i], Y[i] = predict.reshape(32, 32, -1), y.reshape(32, 32, -1)
    #np.save('test1_predict.npy', Predict)
    #np.save('y.npy', Y)
    #np.save()
