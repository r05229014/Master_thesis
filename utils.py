import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def pad_boundary(arr, kernel_size):
    '''
    arr : input array
    kernel_size : training sample size
    '''

    pad_size = int((kernel_size - 1) / 2)

    if len(arr.shape) == 5:
        pad_arr = np.pad(arr, pad_size, 'wrap')
        pad_arr = pad_arr[pad_size:-pad_size, ..., pad_size:-pad_size, pad_size:-pad_size]
    else:
        raise ValueError('len(arr.shape) should be 5')
    return pad_arr


def sliding_window_x(arr, z=33, y=7, x=7, f=9, kernel_size=7):
    arr = np.lib.stride_tricks.sliding_window_view(arr,
                                                   (kernel_size, kernel_size),
                                                   axis=(2, 3))
    arr = np.moveaxis(arr, 4, -1)
    arr = np.moveaxis(arr, 1, 3)
    arr = arr.reshape(-1, z, y, x, f)
    return arr


def load_X(dir_path, feature_range=(0, 1)):
    Vars = []
    names = [p.name for p in Path(dir_path).glob('*/')]
    for name in names:
        path = Path(dir_path).joinpath(name)
        data = np.stack([np.load(p) for p in path.glob('*.npy')])
        if name == 'cape':
            data = data[:, np.newaxis, ..., np.newaxis]
            data = np.concatenate([data]*34, axis=1)
        else:
            data = data[..., np.newaxis]
        Vars.append(data)
    X = np.concatenate(Vars, axis=-1)[:, 1:, ...]
    X = np.moveaxis(X, 1, -2)
    shape = X.shape

    # stdandardlization
    scaler = MinMaxScaler(feature_range=feature_range)
    X = X.reshape(-1, 9)
    X = scaler.fit_transform(X)
    X = X.reshape(shape)
    
    return X.astype(np.float32), scaler


def X_ETL(path, idx, nb_test_samples, mode='train', feature_range=(0,1), pickle_save_dir='models_save/test1/pickles'):
    # load_X and select train/val/test set due to lack of memory
    X, scaler = load_X(path, feature_range)
    nb_test_samples = int(0.2 * X.shape[0])
    
    # shuffle
    X = X[idx]
    
    # split
    if mode is 'train':
        X = X[2*nb_test_samples:]
    elif mode is 'val':
        X = X[0:nb_test_samples]
    else:
        X = X[nb_test_samples:2*nb_test_samples]
    
    # CNN input style
    X = pad_boundary(X, 7)
    X = np.lib.stride_tricks.sliding_window_view(X, (7, 7), axis=(1, 2))
    X = np.moveaxis(X, 4, -1)
    
    # save scaler and pca as pickles
    if mode is 'train':
        Path(pickle_save_dir).mkdir(parents=True, exist_ok=True)
        pickle.dump(scaler, open(f'{pickle_save_dir}/scaler_X.pkl', 'wb'))
    return X.reshape(-1, 33, 7, 7, 9)


def y_ETL(dir_path, idx, nb_test_samples, mode='train', feature_range=(0, 1), pickle_save_dir='models_save/test1/pickles'):
    # load from folder
    whs = list(Path(dir_path).glob('*.npy'))
    whs = np.stack([np.load(path) for path in whs])
    print(whs.shape)
    nb_test_samples = int(0.2 * whs.shape[0])
    whs = whs[idx]
    if mode is 'train':
        whs = whs[2*nb_test_samples:]
    elif mode is 'val':
        whs = whs[0:nb_test_samples]
    else:
        whs = whs[nb_test_samples:2*nb_test_samples]    
    
    # move sample axis to the last, reshape, and remove the first layer
    whs = np.moveaxis(whs, 1, 3).reshape(-1, 34)[:, 1:]
    
    # Min Max normalization
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_whs = scaler.fit_transform(whs)
    
    # PCA
    pca = PCA(n_components=0.96, svd_solver='full')
    out = pca.fit_transform(scaled_whs)
    
    # save scaler and pca as pickles
    if mode is 'train':
        Path(pickle_save_dir).mkdir(parents=True, exist_ok=True)
        pickle.dump(scaler, open(f'{pickle_save_dir}/scaler.pkl', 'wb'))
        pickle.dump(pca, open(f'{pickle_save_dir}/pca.pkl', 'wb'))
    return out.astype(np.float32), pca

# save tf_record data
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def parse_single_data(feature, label):
    #define the dictionary -- the structure -- of our single example
    data = {
        'feature' : _bytes_feature(tf.io.serialize_tensor(feature).numpy()),
        'label' : _bytes_feature(tf.io.serialize_tensor(label).numpy())
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out


def write_data_to_tfr_short(datas, labels, filename:str="data"):
    filename= filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0

    for index in range(len(datas)):

        #get the data we want to write
        current_data = datas[index]
        current_label = labels[index]

        out = parse_single_data(feature=current_data, label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


# parse tf_record data
def _parse_data_function(example_proto):
    data_feature_description = {
        'feature' : tf.io.FixedLenFeature([], tf.string),
        'label' : tf.io.FixedLenFeature([], tf.string)
    }

    # Parse the input tf.train.Example proto using the dictionary above.
    features = tf.io.parse_single_example(example_proto, data_feature_description)
    data = tf.io.parse_tensor(features['feature'], "float") 
    label = tf.io.parse_tensor(features['label'], "float")
    data.set_shape([33,7,7,9])
    label.set_shape([5,])
    return data, label


def get_dataset(dataset, BATCH_SIZE):
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset
