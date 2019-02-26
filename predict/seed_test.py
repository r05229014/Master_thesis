import numpy as np
import random 

def shuffle_forward(l):
    order = list(range(len(l)))
    random.shuffle(order)
    return list(np.array(l)[order]), order

def shuffle_backward(l, order):
    l_out = [0] * len(l)
    for i,j in enumerate(order):
        l_out[j] = l[i]
    return l_out

def CNN3D_type_y(arr):
    out = np.zeros((arr.shape[0]*arr.shape[2]*arr.shape[3], arr.shape[1], 1, 1, arr.shape[4]), dtype='float16')
    
    count = 0
    for t in range(arr.shape[0]):
        for x in range(0 ,arr.shape[2]):
            for y in range(0, arr.shape[3]):
                out[count] = arr[t, :, x:x+1, y:y+1, :]  

                count += 1 
    out = out.reshape(out.shape[0], 5) # 3D CNN

    print('y shape : ', out.shape)
    return out

pre = np.load('./reset_data/3DCNN_pca_all.npy')

print(pre.shape)
