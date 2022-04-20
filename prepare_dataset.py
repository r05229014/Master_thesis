import os
import pickle
import random
import numpy as np
from pathlib import Path
import tensorflow as tf
from utils import X_ETL, y_ETL
from utils import parse_single_data, write_data_to_tfr_short


if __name__ == '__main__':
    # Settings
    mode = 'test'
    save_name_suffix = '0421'
    X_path = 'data/vars/'
    y_path = 'data/target/wh/'
    pickle_save_dir='models_save/test2/pickles'
    
    # set seed for reproduce
    seed = 777
    idx = np.arange(1314)
    random.seed(seed)
    random.shuffle(idx)
    nb_test_samples = int(0.2 * idx.shape[0])

    # X pre-process
    X = X_ETL(X_path, idx, nb_test_samples, mode=mode, pickle_save_dir=pickle_save_dir)

    # y pre-process
    y, pca = y_ETL(y_path, idx, nb_test_samples, mode=mode, pickle_save_dir=pickle_save_dir)

    # save to tf_record data
    write_data_to_tfr_short(X, y, filename=f'{mode}_dataset_{save_name_suffix}')