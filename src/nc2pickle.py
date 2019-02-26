from netCDF4 import Dataset
import os
import pickle

'''
This is a file to turn .nc file to .npy file
Just check the path of .nc file and where you wanna save the .npy file

and maybe you should check the variables' name
'''

def get_nc(path, save_path):
    name_list = os.listdir(path)
    var_list = ['u', 'v', 'w', 'th', 'qv']
    for nc in name_list:
        ncfile = Dataset(path + nc)
        u = ncfile.variables['u'][:]
        v = ncfile.variables['v'][:]
        w = ncfile.variables['w'][:]
        th = ncfile.variables['th'][:]
        qv = ncfile.variables['qv'][:]

        dict = {'u': u, 'v': v, 'w': w, 'th': th, 'qv':qv }
        with open (save_path + nc[0:3] + '.pkl', 'wb') as f:
            pickle.dump(dict, f)
path = '../ncdata/'
save_path = '../data/'
get_nc(path, save_path)
