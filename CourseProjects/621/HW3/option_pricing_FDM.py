import math
import numpy as np
import pandas as pd
import xlwings as xw
import scipy as sp
from scipy import stats
import matplotlib as plt


def bsm_option_value(S0, E, T, R, SIGMA):
    S0 = float(S0)
    d1 = (math.log(S0 / E) + (R + (0.5 * SIGMA ** 2)) * T) / (SIGMA * math.sqrt(T))
    d2 = d1 - (SIGMA * math.sqrt(T))
    call_value = S0 * stats.norm.cdf(d1, 0, 1) - E * math.exp(-R * T) * stats.norm.cdf(d2, 0, 1)
    delta_call = stats.norm.cdf(d1, 0, 1)
    gamma_call = stats.norm.pdf(d1, 0, 1) / (S0 * SIGMA * math.sqrt(T))
    theta_call = -(R * E * math.exp(-R * T) * stats.norm.cdf(d2, 0, 1)) - (
                SIGMA * S0 * stats.norm.pdf(d1, 0, 1) / (2 * math.sqrt(T)))
    rho_call = T * E * math.exp(-R * T) * stats.norm.cdf(d2, 0, 1)
    vega_call = math.sqrt(T) * S0 * stats.norm.pdf(d1, 0, 1)

    put_value = E * math.exp(-R * T) * stats.norm.cdf(-d2, 0, 1) - (S0 * stats.norm.cdf(-d1, 0, 1))
    delta_put = -stats.norm.cdf(-d1, 0, 1)
    gamma_put = stats.norm.pdf(d1, 0, 1) / (S0 * SIGMA * math.sqrt(T))
    theta_put = (R * E * math.exp(-R * T) * stats.norm.cdf(-d2, 0, 1)) - (
                SIGMA * S0 * stats.norm.pdf(d1, 0, 1) / (2 * math.sqrt(T)))
    rho_put = -T * E * math.exp(-R * T) * stats.norm.cdf(-d2, 0, 1)
    vega_put = math.sqrt(T) * S0 * stats.norm.pdf(d1, 0, 1)

    return call_value, delta_call, gamma_call, theta_call, rho_call, vega_call, put_value, delta_put, gamma_put, theta_put, rho_put, vega_put


def optionPrice_FDM(E=100, T=1, r=.06, sigma=0.25, isAmerican=False, isCall=True, NAS=100, NTS=1000,method="explicit"):

    SIGMA = sigma  # Volatility
    Type = isCall  # Type of Option True=Call False=Put
    Ex = isAmerican  # Early Exercise True=Yes  False=No

    ds = 2 * E / NAS  # Asset Value Step Size
    dt = T / NTS  # Time Step Size

    #     NAS = 200  #Number of Asset Steps - Higher is more accurate, but more time consuming
    #     ds = 2 * E / NAS  #Asset Value Step Size
    #     dt = (0.9/NAS/NAS/SIGMA/SIGMA)  #Time Step Size
    #     NTS = int(T / dt) + 1  #Number of Time Steps
    #     dt = T / NTS #Time Step Size

    print("Asset Step Size %.2f Time Step Size %.2f Number of Time Steps %.2f Number of Asset Steps %.2f" % (
    ds, dt, NTS, NAS))

    # Setup Empty numpy Arrays
    value_matrix = np.zeros((int(NAS + 1), int(NTS)))
    asset_price = np.arange(NAS * ds, 0 - 1e-6, -ds)
    # print(asset_price.shape)
    if method == "explicit":

        # Evaluate Terminal Value for Calls or Puts
        if Type == True:
            value_matrix[:, -1] = np.maximum(asset_price - E, 0)
        else:
            value_matrix[:, -1] = np.maximum(E - asset_price, 0)

        # Set Lower Boundry in Grid
        for x in range(1, NTS):
            value_matrix[-1, -x - 1] = value_matrix[-1, -x] * math.exp(-r * dt)

        # Set Mid and Ceiling Values in Grid
        for x in range(1, int(NTS)):

            for y in range(1, int(NAS)):
                # Evaluate Option Greeks
                Delta = (value_matrix[y - 1, -x] - value_matrix[y + 1, -x]) / 2 / ds
                Gamma = (value_matrix[y - 1, -x] - (2 * value_matrix[y, -x]) + value_matrix[y + 1, -x]) / ds / ds
                Theta = (-.5 * SIGMA ** 2 * asset_price[y] ** 2 * Gamma) - (r * asset_price[y] * Delta) + (
                            r * value_matrix[y, -x])

                # Set Mid Values
                value_matrix[y, -x - 1] = value_matrix[y, -x] - Theta * dt
                if Ex == True:
                    value_matrix[y, -x - 1] = np.maximum(value_matrix[y, -x - 1], value_matrix[y, -1])

                # Set Ceiling Value
                value_matrix[0, -x - 1] = 2 * value_matrix[1, -x - 1] - value_matrix[2, -x - 1]
    elif method=="implicit":
        # Evaluate Terminal Value for Calls or Puts
        if Type == True:
            value_matrix[:, -1] = np.maximum(asset_price - E, 0)
            # Set Lower Boundry in Grid
            for x in range(1, NTS):
                value_matrix[-1, -x - 1] = 0
                value_matrix[0, -x - 1] = asset_price[0] - E * math.exp(-r * dt * x)
        else:
            value_matrix[:, -1] = np.maximum(E - asset_price, 0)
            # Set Lower Boundry in Grid
            for x in range(1, NTS):
                value_matrix[-1, -x - 1] = value_matrix[-1, -x] * math.exp(-r * dt)
                value_matrix[0, -x - 1] = 0
        a = lambda x: (r * x / 2 / ds - (SIGMA * x) ** 2 / 2 / ds ** 2) * dt
        b = lambda x: 1 + r * dt + (SIGMA * x) ** 2 * dt / ds ** 2
        c = lambda x: -(r * x / 2 / ds + (SIGMA * x) ** 2 / 2 / ds ** 2) * dt
        a_entries = a(asset_price[1:-2])
        b_entries = b(asset_price[1:-1])
        c_entries = c(asset_price[2:-1])
        A = np.diag(a_entries, 1) + np.diag(b_entries) + np.diag(c_entries, -1)
        # Set Mid and Ceiling Values in Grid
        for x in range(1, int(NTS)):
            d = value_matrix[1:-1, -x]
            d[0] -= c(asset_price[1]) * value_matrix[0, -x - 1]
            d[-1] -= a(asset_price[-2]) * value_matrix[-1, -x - 1]
            value_matrix[1:-1, -x - 1] = np.linalg.solve(A, d)
    elif method == "crank-nicolson":
        # Crank-Nicolson Scheme
        if Type == True:
            value_matrix[:, -1] = np.maximum(asset_price - E, 0)
            # Set Lower Boundry in Grid
            for x in range(1, NTS):
                value_matrix[-1, -x - 1] = 0
                value_matrix[0, -x - 1] = asset_price[0]
        else:
            value_matrix[:, -1] = np.maximum(E - asset_price, 0)
            # Set Lower Boundry in Grid
            for x in range(1, NTS):
                value_matrix[-1, -x - 1] = value_matrix[-1, -x] * math.exp(-r * dt)
                value_matrix[0, -x - 1] = 0
        #     xw.view(value_matrix)
        a1 = lambda x: (-r * x / 4 / ds + (SIGMA * x) ** 2 / 4 / ds ** 2)
        b1 = lambda x: -(1 / dt + r / 2 + (SIGMA * x) ** 2 / 2 / ds ** 2)
        c1 = lambda x: (r * x / 4 / ds + (SIGMA * x) ** 2 / 4 / ds ** 2)
        a2 = a1
        b2 = lambda x: -(-1 / dt + r / 2 + (SIGMA * x) ** 2 / 2 / ds ** 2)
        c2 = c1

        a1_entries = a1(asset_price[1:-2])
        b1_entries = b1(asset_price[1:-1])
        c1_entries = c1(asset_price[2:-1])

        A = np.diag(a1_entries, 1) + np.diag(b1_entries) + np.diag(c1_entries, -1)
        # Set Mid and Ceiling Values in Grid
        for x in range(1, int(NTS)):
            d = -(value_matrix[0:-2, -x] * a2(asset_price[1:-1]) + value_matrix[1:-1, -x] * b2(asset_price[1:-1])
                  + value_matrix[2:, -x] * c2(asset_price[1:-1]))
            d[0] -= c1(asset_price[1]) * value_matrix[0, -x - 1]
            d[-1] -= a1(asset_price[-2]) * value_matrix[-1, -x - 1]
            value_matrix[1:-1, -x - 1] = np.linalg.solve(A, d)

    # Option Valuation Profile in pandas - Index is Strike Price, column 0 is the option price
    value_df = pd.DataFrame(value_matrix)
    value_df = value_df.set_index(asset_price)

    # Export Value Grid to Excel via xlWings
    # xw.view(value_df)

    # Run BSM Calculation for values and greeks
    S0 = 100  # Current Value
    R = r
    # BSM function call and output assignment
    call_value, delta_call, gamma_call, theta_call, rho_call, vega_call, put_value, delta_put, gamma_put, theta_put, rho_put, vega_put = bsm_option_value(
        S0, E, T, R, SIGMA)

    if Type == False:
        BSM_val = put_value
    if Type == True:
        BSM_val = call_value
    print("Black-Scholes value: {}".format(BSM_val))

    # Finite Differences Method Value at S0
    fd_value = value_df.ix[S0, 1]
    print("finite difference value: {}".format(fd_value))

    # Difference
    diff = BSM_val - fd_value
    print("Nominal Difference is %.4f" % diff)

    pct_diff = abs(diff / BSM_val)
    print("Percent Difference is {percent:.4%}".format(percent=pct_diff))
    return fd_value