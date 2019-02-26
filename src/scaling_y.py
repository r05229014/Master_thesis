import numpy as np
from sklearn.preprocessing import StandardScaler
from netCDF4 import Dataset
from sklearn.decomposition import PCA
import time 
import matplotlib.pyplot as plt

start = time.time()
# scaled wh
wh = Dataset('../input/d16_ans_wh.nc')
y = wh['wh'][:]
print(y.shape)

mean_y = np.mean(np.mean(np.mean(y, axis=0), axis=-1), axis=-1)
print(mean_y.shape)
std_y = np.std(np.std(np.std(y, axis=0), axis=-1), axis=-1)
print(std_y.shape)
#print(mean_y, mean_y.shape, 'mean')
#print(std_y, std_y.shape, 'std')

scaled_y = np.zeros((666,33,32,32))
for t in range(y.shape[0]):
    for yy in range(y.shape[2]):
        for xx in range(y.shape[3]):
            scaled_y[t,:,yy,xx] = (y[t,:,yy,xx] - mean_y)/std_y
print(scaled_y.shape, "scaled_y's shape")

np.save('../input/scaled_wh.npy', scaled_y)
#np.save('mean_wh.npy', mean_y)
#np.save('std_wh.npy', std_y)
#wh = Dataset('../input/d16_ans_wh.nc')
#scaled_wh= wh['wh'][:]

#scaled_wh = np.load('./scaled_wh.npy')
#scaled_wh = np.swapaxes(scaled_wh, 1,2)
#scaled_wh = np.swapaxes(scaled_wh, 2,3)
#scaled_wh = scaled_wh.reshape(-1,33)
#print(scaled_wh.shape)
#
#pca = PCA(n_components=1)
#pca.fit(scaled_wh)
#print(pca.explained_variance_ratio_)
#print(sum(pca.explained_variance_ratio_))
#
#eigen_wh = pca.fit_transform(scaled_wh)
#print(eigen_wh)
##print(eigen_wh.shape, 'eigen')
##print(pca.components_[0,:].shape)
##print(pca.components_[0,-1])
###wh_1 = pca.inverse_transform(eigen_wh)
###print(wh_1.shape, 'first')
##
##print('\n It cost ',time.time()-start, 'seconds')
##
##
##z = np.loadtxt('../z.txt')[1::]
##plt.figure(figsize=(4,6))
##plt.plot(pca.components_[0,:], z )
##plt.legend()
##plt.savefig('./pca.png')
