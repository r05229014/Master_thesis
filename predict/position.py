import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os 
import sys
import matplotlib.colors as colors
import matplotlib as mpl

def truncate_colormap(cmap, minval=0.0, maxval=1.0, ncolor=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval)), ncolor)
    return new_cmap

label_size = 13
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

plot = sys.argv[1]
ha = sys.argv[2]
## true
true = './True/y_test.npy'
ans = np.load(true)

## DNN
dnn_6f = './DNN_1219_6f/testing.npy'
dnn_9f = './DNN_1219_9f/testing.npy'
d6 = np.load(dnn_6f)
d9 = np.load(dnn_9f)

## CNN need to reconstruct and rescale !!!!!!
cnn_6f = './CNN_6features_1219/testing.npy'
cnn_9f = './CNN_9features_1216_2/testing.npy'
c6 = np.load(cnn_6f)
c9 = np.load(cnn_9f)

## Linear
linear_6f = './Linear_1214_6f/testing_6f.npy'
linear_9f = './Linear_1214_9f/testing_9f.npy'
l6 = np.load(linear_6f)*2.5*10**6
l9 = np.load(linear_9f)*2.5*10**6
#print(t.shape)
if ha == 'd9':
    t = d9
    dir = 'DNN9f/'
elif ha == 'c9':
    t = c9
    dir = 'CNN9f/'
elif ha == 'l9':
    t = l9
    dir = 'Linear9f/'
elif ha == 'Ans':
    t = ans
    dir = 'Ans/'
###################
y = t[:,12,:,:]
#y = y.reshape(-1)
#for i in range(y.shape[0]):
#    if y[i] >-0.1 and y[i]<0.1:
#        y[i] = np.nan
#y = y.reshape(-1,32,32)
y2 = ans[:,12,:,:]
print(y.shape)
#dir = 'Ans/'
case = dir[:-1]

if plot == 'go':

    # plot set
    x = np.arange(32)
    XX, YY = np.meshgrid(x,x)
    levels = MaxNLocator(nbins=9).tick_values(-5, 5)
    levels = [-6,-5,-4, -3, -2,-1, -0.01, 0.01, 1, 2, 3, 4,5,6]
    levels2 = [ 0.01,2]
    #z = np.loadtxt('../z.txt')
    #print(z[12])
    # plot position through time
    t_list = [116, 12, 3, 4]
    ################################
    #for t in t_list:
    #    plt.figure(t, figsize=(6,5))
    #    cmap = plt.get_cmap('seismic')
    #    #cmap2 = plt.get_cmap('seismic_r')
    #    c = plt.contourf((XX+1)*16, (YY+1)*16, y[t,:,:]/1004, levels=levels, cmap=cmap)
    #    #cc = plt.contour((XX+1)*16, (YY+1)*16, y2[t,:,:]/1004, levels=levels2)
    #    plt.xlabel('[km]', fontsize=13)
    #    plt.ylabel('[km]', fontsize=13)
    #    #plt.title(case  + '  '+ r"$\overline{w'h'}$" + ' with Time = %s \n z = 3km' %t, fontsize=14)
    #    plt.title(case + ' ' + r"$\overline{w'h'}$", fontsize=14)
    #    cbar = plt.colorbar(c, ticks=[-6,-5,-4,-3,-2,-1,-0.01, 0.01,1,2,3,4,5,6])
    #    #cbar = plt.colorbar(cc, ticks=levels2)
    #    cbar.set_label('[K m/s]', fontsize=13)
    #    plt.grid(True)
    #    plt.savefig('./position_img/' + dir + '%s'%t, dpi=300)
    #    plt.close()
    #    ###############################
    plt.figure(figsize=(6,5))
    cmap = plt.get_cmap('seismic')
    fig, axes = plt.subplots(nrows=2, ncols=2)
    
    time=0
    for ax in axes.flat:
        print(ax)
        im = ax.contourf((XX+1)*16, (YY+1)*16, y[t_list[0+time],:,:]/1004, levels=levels, cmap=cmap)
        im2 = ax.contour((XX+1)*16, (YY+1)*16, y2[t_list[0+time],:,:]/1004, levels=levels2)
        #ax.set_xlabel('[km]', fontsize=13)
        #ax.set_ylabel('[km]', fontsize=13)
        ax.grid(True)
        time+=1

    axes[0,0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False)
    axes[0,1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False, left=False)
    axes[1,0].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=True)
    axes[1,1].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=True, labelleft=False, left=False)
    axes[0,0].set_ylabel('[km]')
    axes[1,0].set_ylabel('[km]')
    axes[1,0].set_xlabel('[km]')
    axes[1,1].set_xlabel('[km]')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, ticks=[-6,-5,-4,-3,-2,-1,-0.01, 0.01,1,2,3,4,5,6], cax=cbar_ax)
    cbar.set_label('[K m/s]', fontsize=13)

    fig.suptitle(case + ' ' + r"$\overline{w'h'}$", fontsize=14)
    plt.savefig('./position_img/' + dir + 'all_plot', dpi=300)
    plt.close()
