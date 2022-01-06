import numpy as np
import pickle
from tensorflow import keras
from PIL import Image
from utils import pad_boundary, sliding_window_x


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,
                 var_names, pca_path,
                 kernel_size=7,
                 batch_size=256, dim=(34, 32, 32),
                 n_channels=9, 
                 shuffle=True, train=True):
        'Initialization'
        self.dim = dim
        self.vars = var_names
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.pca = pickle.load(open(pca_path, 'rb'))
        self.shuffle = shuffle
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def preprocessing(self, X, y):
        X = pad_boundary(X, self.kernel_size)
        X = sliding_window_x(X,
                             z=34,
                             y=self.kernel_size,
                             x=self.kernel_size,
                             f=9,
                             kernel_size=self.kernel_size)

        y = np.moveaxis(y, 1, -1).reshape(-1, 34)

        if self.train:
            y = self.pca.transform(y)
        return X, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 9))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            for v, name in enumerate(self.vars):
                var = np.load(f'data/vars/{name}/{name}_{ID}.npy')
                X[i, ..., v] = var

            # Store class
            y[i] = np.load(f'data/target/wh/wh_{ID}.npy')/2.5/10**6
        X, y = self.preprocessing(X, y)
        return X, y
