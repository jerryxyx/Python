import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize

initial_value = 500
value_vector = np.array([520, 560, 600, 540, 480, 400, 700, 710, 705, 715, 640, 670])
inflow_vector = np.array([50, 100, 80, 0, 0, 0, 250, 0, 0, 10, 20, 15])
outflow_vector = np.array([0, 0, 20, 0, 10, 20, 50, 10, 0, 0, 60, 15])
rm_vector = np.array([10, 8, 5, 12, -3, -1, 10, 4, 5, -5, -8, 7]) * 0.01
n_months = 12

df = pd.DataFrame()
df['value'] = value_vector
df['inflow'] = inflow_vector
df['outflow'] = outflow_vector
df['rm'] = rm_vector
df.index.name = "date"
print(df)


def compute_time_weighted_return(initial_value, value_vector, inflow_vector,
                                 outflow_vector, n_months):
    time_weighted_return = \
        (value_vector[0] - inflow_vector[0] + outflow_vector[0]) \
        / initial_value
    for i in range(1, n_months):
        time_weighted_return *= \
            (value_vector[i] - inflow_vector[i] + outflow_vector[i]) \
            / value_vector[i - 1]
    time_weighted_return -= 1
    return time_weighted_return


def compute_modified_monthly_return(initial_value, value_vector,
                                    inflow_vector, outflow_vector,
                                    n_months):
    modified_monthly_return = []
    modified_monthly_return.append(
        (value_vector[0] - inflow_vector[0] + outflow_vector[0])
        / initial_value)
    for i in range(1, n_months):
        modified_monthly_return.append(
            (value_vector[i] - inflow_vector[i] + outflow_vector[i])
            / value_vector[i - 1])
    modified_monthly_return = np.asarray(modified_monthly_return)
    modified_monthly_return -= 1
    return modified_monthly_return


def compute_simple_dietz_return(initial_value, end_value, inflow_vector,
                                outflow_vector):
    C_net = np.sum(inflow_vector - outflow_vector)
    return (end_value - initial_value - C_net) / (initial_value + C_net / 2)


def compute_modified_dietz_return(initial_value, end_value, inflow_vector,
                                  outflow_vector, n_months):
    C_net = np.sum(inflow_vector - outflow_vector)
    C = inflow_vector - outflow_vector
    C_avg = 0
    for i in range(n_months):
        C_avg += (n_months - i - 1) / n_months * C[i]
    return (end_value - initial_value - C_net) / (initial_value + C_avg)


time_weighted_return = \
    compute_time_weighted_return(initial_value, value_vector, inflow_vector,
                                 outflow_vector, n_months)
simple_dietz_return = \
    compute_simple_dietz_return(initial_value, value_vector[-1],
                                inflow_vector, outflow_vector)
modified_dietz_return = \
    compute_modified_dietz_return(initial_value, value_vector[-1], inflow_vector,
                                  outflow_vector, n_months)
print("time weighted return: " + str(time_weighted_return))
print("simple dietz return: " + str(simple_dietz_return))
print("modified dietz return: " + str(modified_dietz_return))


# 7.2
def compute_future_value(interest_rate, initial_value, inflow_vector,
                         outflow_vector, n_months=12):
    cash_flow = inflow_vector - outflow_vector
    future_value = (1 + interest_rate) * initial_value
    future_value = float(future_value)
    for i in range(n_months):
        future_value += cash_flow[i] * (1 + interest_rate) ** (1 - (i + 1) / n_months)
        # print(future_value)
    return future_value


sol = sp.optimize.root(lambda r:
                       compute_future_value(r, initial_value, inflow_vector,
                                            outflow_vector)
                       - value_vector[-1],
                       x0=0)
irr = float(sol.x)
print('IRR: ' + str(irr))
print("check future value: "
      + str(compute_future_value(irr, initial_value, inflow_vector, outflow_vector)))

# 7.3
import statsmodels.api as sm

modified_monthly_return = \
    compute_modified_monthly_return(initial_value, value_vector, inflow_vector,
                                    outflow_vector, n_months)
df['monthly_return'] = modified_monthly_return
print(df)

results = sm.OLS(df.monthly_return, sm.add_constant(df.rm)).fit()
# print(results.summary())
rsquared = results.rsquared
rsquared_adj = results.rsquared_adj
alpha = results.params[0]
beta = results.params[1]
omega = np.std(results.resid)
mp = np.mean(df.monthly_return)
mm = np.mean(df.rm)
std_m = np.std(df.rm)
std_p = np.std(df.monthly_return)
rF = 0
Sharpe_ratio = (mp - rF) / std_p
Treynor_ratio = (mp - rF) / beta
IR = (mp - beta * mm) / omega
msquared = mp * std_m / std_p - mm

print("alpha: " + str(alpha))
print("beta: " + str(beta))
print("R squared: " + str(rsquared))
print("adjusted R squared: " + str(rsquared_adj))
print("Sharpe Ratio: " + str(Sharpe_ratio))
print("Treynor Ratio: " + str(Treynor_ratio))
print("IR: " + str(IR))
print("M squared: " + str(msquared))
