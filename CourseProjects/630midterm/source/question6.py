import numpy as np

h = np.array([.2, .3, .5]).T
mu = np.array([1.1, 1.2, 1.3]).T
Q = np.array([[3, 0.2, 0.1], [0.2, 7, 0.4], [0.1, 0.4, 4]]) * 0.01

# 6.1
re = np.matmul(h.T, mu)
print("expected return=", re, "%")

# 6.2
var = np.matmul(np.matmul(h.T, Q), h)
std = np.sqrt(var)
print("variance=", var)
print("std=", std)
