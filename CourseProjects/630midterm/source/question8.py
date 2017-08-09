import numpy as np

mu = np.array([1.1, 1.2, 1.3]).T * 0.01
Q = np.array([[3, 0.2, 0.1], [0.2, 7, 0.4], [0.1, 0.4, 4]]) * 0.01
l = np.array([1, 1, 1]).T
A = np.matmul(np.matmul(mu.T, np.linalg.inv(Q)), mu)
B = np.matmul(np.matmul(mu.T, np.linalg.inv(Q)), l)
C = np.matmul(np.matmul(l.T, np.linalg.inv(Q)), l)
D = A * C - B * B
rf = 0.005


def efficientFrontier(re, sigma, A, B, C, D):
    return sigma * sigma / (1 / C) - pow((re - B / C), 2) / (D / C) - 1


def sharpeRatio(re, rf, sigma):
    return (re - rf) / sigma


# 8.1
from scipy.optimize import minimize

SR = lambda x: -sharpeRatio(x[0], rf, x[1])
cons = ({'type': 'eq',
         'fun': lambda x: np.array([efficientFrontier(x[0], x[1], A, B, C, D)]),
         'jac': lambda x: np.array([-2 * (x[0] - B / C) / (D / C), 2 * C * x[1]])},
        {'type': 'ineq',
         'fun': lambda x: np.array([x[0] - rf]),
         'jac': lambda x: np.array([1, 0])},
        {'type': 'ineq',
         'fun': lambda x: np.array([x[1]]),
         'jac': lambda x: np.array([0, 1])}
        )
res = minimize(SR, [0, 0.5], constraints=cons, options={'disp': True})
print(res)

optimized_sharpe_ratio = -res.fun
optimized_return = res.x[0]
optimized_sigma = res.x[1]
print("expected return of tangency portfolio:", optimized_return)
print("standard deviation of tangency portfolio:", optimized_sigma)

# 8.2
import pandas as pd

mu_pair = np.array([rf, optimized_return]).T


def expectedReturn(h, mu):
    return np.matmul(h.T, mu)


def standardDeviation(h, sigma):
    return sigma * h[1]


df = pd.DataFrame(columns=['h1', 'h2', 'expected_return', 'standard_deviation'])
h1_list = [.8, .5, .2, -.2]
for i in range(np.size(h1_list)):
    h = np.array([h1_list[i], 1 - h1_list[i]]).T
    df.loc[i] = [h[0], h[1], expectedReturn(h, mu_pair), standardDeviation(h, optimized_sigma)]

print(df)
