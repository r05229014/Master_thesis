# nc2pickle.py
This file is use to transfer .nc file to pickle.

# generate_x.py
The original data is high resolution data(output by VVM), we want to imitate the Cumulus Parameterization in GCM.
So we do some mean operation to mimic the low resolution grid in GCM.

# generate_y.py
Calculate the $\bar{w'qv'}$ and $\bar{w'th'}$

# LinearRegression.py
Use the classic Linear Rgression method to solve the quesion.
We will take this result as a baseline.

# DNN.py
Use DNN to approch our question.

## todo
RNN
biRNN
RNN with time
CNN + RNN
