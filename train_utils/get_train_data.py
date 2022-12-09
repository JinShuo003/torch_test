import numpy as np

x_value = [i for i in range(11)]
x_train = np.array(x_value, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_value = [2*i+1 for i in range(11)]
y_train = np.array(y_value, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


