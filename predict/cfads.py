import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
from numba import jit
label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

plot = sys.argv[1]
ha = sys.argv[2]
## true
true = './True/y_test.npy'
t = np.load(true)
#tt = t
## DNN
dnn_6f = './DNN_1219_6f/testing.npy'
dnn_9f = './DNN_1219_9f/testing.npy'
d6 = np.load(dnn_6f)
d9 = np.load(dnn_9f)

## CNN need to reconstruct and rescale !!!!!!
cnn_6f = './CNN_6features_1219/testing.npy'
cnn_9f = './CNN_9features_1216_2/testing.npy'
#tt = np.load(cnn_6f)

c6 = np.load(cnn_6f)
c9 = np.load(cnn_9f)

## Linear
linear_6f = './Linear_1214_6f/testing_6f.npy'
linear_9f = './Linear_1214_9f/testing_9f.npy'
l6 = np.load(linear_6f)*2.5*10**6
l9 = np.load(linear_6f)*2.5*10**6 
#print(t.shape)

if ha == 'd9':
    tt = d9
    case = 'DNN9f'
elif ha == 'c9':
    tt = c9
    case = 'CNN9f'
elif ha == 'l9':
    tt = l9
    case = 'Linear9f'
elif ha == 'Ans':
    tt = t
    case = 'Ans'
    
###################
#case = 'Ans'
z = np.loadtxt('../z.txt')[1::]

y = t/1004
print(y.shape)
y = np.swapaxes(y,1,2)
y = np.swapaxes(y,2,3)
y = y.reshape(-1,33)
max_y = np.max(y)
min_y = np.min(y)

y = tt/1004
y = np.swapaxes(y,1,2)
y = np.swapaxes(y,2,3)
y = y.reshape(-1,33)
inter = np.arange(min_y, max_y, (max_y-min_y)/100)
inter = np.append(inter, max_y)
#temp = np.zeros((33,100))
@jit(nopython=True)
def filttt(arr1, arr2):
    temp = np.zeros((33,100))
    for p in range(arr1.shape[0]-1):
        print(p)
        for zz in range(33):
            for s in range(arr2.shape[0]):
                if arr2[s,zz] >=  arr1[p] and arr2[s,zz] <= arr1[p+1]:
                     temp[zz,p]+=1
    return temp
temp = filttt(inter, y)
#for p in range(inter.shape[0]-1):
#    print(p)
#    for zz in range(33):
#        for s in range(y.shape[0]):
#            if y[s,zz]>= inter[p] and y[s,zz] <= inter[p+1]:
#                temp[zz,p] += 1

#np.save('yyy.npy', temp)
#temp = np.load('yyy.npy')/136192
temp = temp/136192
temp[temp==0] = np.nan





y2 = t/1004
y2 = np.swapaxes(y2,1,2)
y2 = np.swapaxes(y2,2,3)
y2 = y2.reshape(-1,33)
inter2 = np.arange(min_y, max_y, (max_y-min_y)/100)
inter2 = np.append(inter2, max_y)
temp2 = filttt(inter2, y2)
temp2 = temp2/136192
temp2[temp2<0.00001] = 0


cmap = plt.get_cmap('jet_r')

XX,yy = np.meshgrid(np.arange(inter.min(),inter.max(),(inter.max()-inter.min())/100), z)
#levels = MaxNLocator(nbins=100).tick_values(0,1)
#levels = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.75 ,0.8, 0.85,0.9,0.95,1]
levels = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.5, 0.7 ,0.8,0.9,0.95,1]
levels22 = levels
temp2[XX <0.00001] = 0
norm = BoundaryNorm(levels, ncolors=256, clip=True)
if plot == 'go':
    plt.figure(figsize=(8,8))
    plt.title("%s_CFADs"%case, fontsize=20)
    c = plt.pcolor(XX,yy/1000,temp, norm=norm,cmap=cmap)
    #c2 = plt.contour(XX,yy/1000, temp2, colors='black', levels=levels22,
    #        linewidths=3, linestyles='solid')
    #c = plt.contour(XX,yy,temp, levels=levels,cmap=cmap)
    #c = plt.contourf(XX,yy,temp, levels=levels, cmap=cmap)
    plt.xlabel(r"$\overline{w'h'}$[K m/s]", fontsize=20)
    plt.ylabel('Height [km]', fontsize=20)
    cbar = plt.colorbar(c)
    cbar.set_label('Frequency [%]', fontsize=20)
    cbar.set_ticks(np.array(levels))
    cbar.set_ticklabels(np.array(levels)*100)
    #plt.grid(True)
    plt.savefig('./CFADs_img/%s.png'%case)
