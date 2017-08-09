import numpy as np
import matplotlib.pyplot as plt

# 7.1
M = np.array([[.01, 0, 0, 1], [0, .05, 0, 1], [0, 0, .07, 1], [1, 1, 1, 0]])
b = np.array([[0], [0], [0], [1]])
hlambda = np.matmul(np.linalg.inv(M), b)
h = hlambda[0:-1, :]
print("h=", h)
Q = np.array([[.01, 0, 0], [0, .05, 0], [0, 0, .07]])
var = np.matrix(h.T) * np.matrix(Q) * np.matrix(h)
print("variance=", var[0, 0])
std = np.sqrt(var)
print("std=", std[0, 0])
mu = [1.1, 1.2, 1.3]
re = np.matmul(h.T, mu)
print("expected return = ", re[0], "%")

# 7.2
M = M
mu = [.011, .012, .013]
Q = Q
M_inv = np.linalg.inv(M)


def b(coef):
    mu = [.011, .012, .013]
    b = [i * coef for i in mu]
    b.append(1)
    return np.asarray(b).T


x_coef = [i * 0.01 for i in range(101)]
h_list = []
r_list = []
std_list = []
for coef in x_coef:
    hlambda = np.matmul(M_inv, b(coef=coef))
    h = hlambda[0:-1]
    h_list.append(h)
    re = np.matmul(h, mu)
    r_list.append(re)
    var = np.matrix(h) * np.matrix(Q) * np.matrix(h).T
    std_list.append(np.sqrt(var[0, 0]))

plt.subplot(121)
plt.title("expected return")
plt.plot(x_coef, r_list)

plt.subplot(122)
plt.title("std")
plt.plot(x_coef, std_list)

plt.show()
