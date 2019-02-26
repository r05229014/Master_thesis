import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile

ans_wh_all = np.fromfile('./CWV_dat/ans_wh_cond_cwv.dat', dtype=float).reshape(33,31)
print(ans_wh_all/1004)

