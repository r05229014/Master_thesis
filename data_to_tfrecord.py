import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

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

def _parse_data_function(example_proto, data_feature_description):
    # Parse the input tf.train.Example proto using the dictionary above.
    features = tf.io.parse_single_example(example_proto, data_feature_description)
    features['feature'] = tf.io.parse_tensor(features['feature'], "float32")
    features['label'] = tf.io.parse_tensor(features['label'], "float32")
    return features


if __name__ == '__main__':
    # load npy data
    x = np.load('input/data_x.npy')
    y = np.load('input/data_y.npy')

    # write data to TFRecord
    write_data_to_tfr_short(x, y, filename='dataset_v1')

    # read TFRecord
    raw_dataset = tf.data.TFRecordDataset('dataset_v1.tfrecords')

    # Create a dictionary describing the features.
    data_feature_description = {
            'feature' : tf.io.FixedLenFeature([], tf.string),
            'label' : tf.io.FixedLenFeature([], tf.string)
    }

    parsed_dataset = raw_dataset.map(_parse_data_function, data_feature_description)
