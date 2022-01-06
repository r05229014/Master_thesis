import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from data_generator import DataGenerator
from models import CNN3D
from tensorflow.keras.models import load_model


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    root_path = 'data/vars/w'
    var_names = os.listdir('data/vars/')
    pca_path = 'models/pca.pkl'
    IDs = [name.rstrip('.npy').lstrip('w_')
           for name in os.listdir(root_path)]

    # Parameters
    params = {'dim': (34, 32, 32),
              'batch_size': 1,
              'n_channels': 9,
              'kernel_size': 7,
              'shuffle': False,
              'train': False,
              'var_names': var_names,
              'pca_path': pca_path}

    # Datasets
    partition = {'train': IDs[:int(len(IDs)*0.8)],
                 'validation': IDs[int(len(IDs)*0.8):]}

    # get generator
    val_generator = DataGenerator(partition['validation'], **params)

    # get model
    # model = CNN3D(params['n_channels'])
    # model.load_weights('models_save/test1/CNN3D_013-0.00000017.hdf5')
    model = load_model('models_save/test1/CNN3D_013-0.00000017.hdf5')
    #print(model.summary())
    assert 6==5
    pca = pickle.load(open(pca_path, 'rb'))

    # predict
    for i, (X, y) in enumerate(val_generator):
        predict = model.predict(X, batch_size=1024)
        predict = pca.inverse_transform(predict)

        fig, ax = plt.subplots(1, 2)
        for j in range(1024):
            print(predict[j], j)
            ax[0].plot(predict[j], np.arange(0, 34))
            ax[1].plot(y[j], np.arange(0, 34))
        plt.savefig(f'img/20200105/{i}.png')
        plt.close()
        # predict, y = predict.reshape(32, 32, -1), y.reshape(32, 32, -1)
        # print(predict.shape, y.shape)
