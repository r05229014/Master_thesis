import os
import numpy as np
import tensorflow as tf
from PIL import Image
from data_generator import DataGenerator
from models import CNN3D


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
              'shuffle': True,
              'train': True,
              'var_names': var_names,
              'pca_path': pca_path}

    # Datasets
    partition = {'train': IDs[:int(len(IDs)*0.8)],
                 'validation': IDs[int(len(IDs)*0.8):]}

    # get generator
    train_generator = DataGenerator(partition['train'], **params)
    val_generator = DataGenerator(partition['validation'], **params)

    # get model
    model = CNN3D(params['n_channels'])

    # callback
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/test2/',
                                                 update_freq=1)
    checkpoint_filepath = 'models_save/test2/CNN3D_{epoch:03d}-{val_loss:.8f}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=checkpoint_filepath,
                                            save_weights_only=True,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=False)

    # Train model on dataset
    model.fit(train_generator,
              epochs=1000,
              validation_data=val_generator,
              callbacks=[model_checkpoint_callback, tensorboard])