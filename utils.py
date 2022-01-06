import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pad_boundary(arr, kernel_size):
    '''
    arr : input array
    kernel_size : training sample size
    '''

    pad_size = int((kernel_size - 1) / 2)

    if len(arr.shape) == 5:
        pad_arr = np.pad(arr, pad_size, 'wrap')
        pad_arr = pad_arr[pad_size:-pad_size, pad_size:-pad_size, ..., pad_size:-pad_size]
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


# def sliding_window_y(arr, z=5, kernel_size=1):
#     print(arr.shape, '!!!')
#     arr = np.lib.stride_tricks.sliding_window_view(arr,
#                                                    (kernel_size, kernel_size),
#                                                    axis=(2, 3))
#     print(arr.shape, '!!!')
#     arr = np.moveaxis(arr, 2, -1)
#     arr = arr.reshape(-1, z)
#     return arr
