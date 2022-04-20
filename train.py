import os
import numpy as np
import tensorflow as tf
from PIL import Image
from models import CNN3D
from utils import _parse_data_function, get_dataset

# def _parse_data_function(example_proto):
#     data_feature_description = {
#         'feature' : tf.io.FixedLenFeature([], tf.string),
#         'label' : tf.io.FixedLenFeature([], tf.string)
#     }

#     # Parse the input tf.train.Example proto using the dictionary above.
#     features = tf.io.parse_single_example(example_proto, data_feature_description)
#     data = tf.io.parse_tensor(features['feature'], "float") 
#     label = tf.io.parse_tensor(features['label'], "float")
#     data.set_shape([33,7,7,9])
#     label.set_shape([5,])
#     return data, label


# def get_dataset(dataset, BATCH_SIZE):
#     dataset = dataset.shuffle(2048)
#     dataset = dataset.prefetch(buffer_size=AUTOTUNE)
#     dataset = dataset.batch(BATCH_SIZE)
#     return dataset


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	AUTOTUNE = tf.data.AUTOTUNE
	BATCH_SIZE = 1024
	# root_path = 'input/x'
	# var_names = os.listdir('data/vars/')
	# pca_path = 'models/pca.pkl'
	# IDs = [name.rstrip('.npy').lstrip('x_') for name in os.listdir(root_path)]

	# Parameters
	# params = {'dim': (33, 7, 7),
	# 		  'batch_size': 512,
	# 		  'n_channels': 9,
	# 		  'kernel_size': 7,
	# 		  'shuffle': True,
	# 		  'train': True,
	# 		  'var_names': var_names,
	# 		  'pca_path': pca_path}

	# Datasets
	# partition = {'train': IDs[:int(len(IDs)*0.8)],
	# 			 'validation': IDs[int(len(IDs)*0.8):]}

	# get tf dataset
	train_dataset = tf.data.TFRecordDataset('dataset/train_dataset.tfrecords')
	val_dataset = tf.data.TFRecordDataset('dataset/val_dataset.tfrecords')

    # Create a dictionary describing the features.
	train_dataset = train_dataset.map(_parse_data_function, num_parallel_calls=AUTOTUNE)
	val_dataset = val_dataset.map(_parse_data_function, num_parallel_calls=AUTOTUNE)
	train_dataset = get_dataset(train_dataset, BATCH_SIZE)
	val_dataset = get_dataset(val_dataset, BATCH_SIZE)

	# get model
	model = CNN3D(9)

	# callback
	checkpoint_filepath = 'models_save/test1/CNN3D_{epoch:03d}-{val_loss:.8f}.hdf5'
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
											filepath=checkpoint_filepath,
											save_weights_only=True,
											monitor='val_loss',
											mode='min',
											save_best_only=True,
											save_format='tf')

	# Train model on dataset
	model.fit(train_dataset,
			  validation_data=val_dataset,
			  epochs=1000,
			  batch_size=BATCH_SIZE,
			  callbacks=[model_checkpoint_callback])
