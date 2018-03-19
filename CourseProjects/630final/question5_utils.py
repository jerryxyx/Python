import numpy as np
import pandas as pd

alpha_array = np.array([10, 9, 11, 5, 8, 7, 5, 4, 8, 3]) * 0.01
beta_array = np.array([1.2, 1.1, 1.3, 0.8, 0.9, 0.7, 0.6, 0.6, 0.8, 0.5])
omega_array = np.array([30, 25, 27, 20, 22, 20, 20, 29, 24, 20]) * 0.01
b_array = np.array([10, 12, 12, 6, 21, 8, 6, 8, 6, 11]) * 0.01
sigmaM = 0.2  # market standard deviation
muM = 0.07  # expected market return
rf = 0.03  # risk-free rate
re_array = alpha_array + beta_array * muM  # expected securities return array
rb = np.sum(re_array * b_array)  # benchmark return
n_features = len(beta_array)  # number of securities
eta = np.ones(n_features)

df = pd.DataFrame(columns=['alpha', 'beta', 'omega', 'b', 'mu'])
df.alpha = alpha_array
df.beta = beta_array
df.omega = omega_array
df.mu = df.alpha + df.beta * muM
mu_array = df.mu
df.b = b_array
print("data:")
print(df)
print("market's standard deviation: " + str(sigmaM))
print("risk-free interest rate: " + str(rf))
print("expected market's return: " + str(muM))
print("expected benchmark's return: " + str(rb))


def compute_expected_alpha(h, alpha):
    return np.dot(h, alpha)


def compute_expected_return(h, alpha, beta, muM):
    return np.dot(h, alpha + beta * muM)


def compute_standard_deviation(h, cov):
    var = np.dot(np.dot(h.T, cov), h)
    std = np.sqrt(var)
    return std


def compute_covariance(beta, omega, sigmaM):
    n_features = len(omega)
    cov = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            cov[i, j] = beta[i] * beta[j] * sigmaM ** 2
            if i == j:
                cov[i, j] += omega[i] ** 2
    return cov


cov = compute_covariance(beta_array, omega_array, sigmaM)


def compute_portfolio_variance(h, Q):
    return np.dot(np.dot(h, Q), h)

# print(compute_portfolio_variance(np.array([1,2]),np.array([[1,2],[3,4]])))
