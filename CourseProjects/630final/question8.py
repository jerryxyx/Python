import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize

liabilities = np.array([10, 15, 20, 25, 20, 15])
df = pd.DataFrame()
df['maturity'] = [12, 12, 12, 24, 24, 24, 36, 36, 36]
df['coupon'] = np.dot([0, 1, 1.2, 0, 1.3, 1.4, 0, 1.3, 1.4], 0.01)
price_vector = np.array([997, 1000, 1050, 992, 1100, 1150, 990, 1200, 1280])
df['price'] = price_vector
df.index.name = 'bond'
print("###########################################################")
print(df)
print("###########################################################")
n_bonds = len(df.index)
n_coupon_dates = len(liabilities)


def compute_cash_flow_matrix(df, period=6, length=36):
    if (length % period == 0):
        cash_flow_matrix = np.zeros((length // period, len(df.index)))
        for i in range(len(df.index)):
            data = df.ix[i, :]
            n_coupon_date = int(data.maturity // period)
            cash_flow_matrix[:n_coupon_date, i] = data.coupon * 1000
            cash_flow_matrix[n_coupon_date - 1, i] += 1000
        return cash_flow_matrix

    else:
        print("check your period")
        return


def compute_portfolio_cash_flow(cash_flow_matrix, n_vector):
    return np.dot(cash_flow_matrix, n_vector)


cash_flow_matrix = compute_cash_flow_matrix(df)
print("###########################################################")
print("cash flow matrix:")
print(cash_flow_matrix)
print("###########################################################")
##########################################################################
# scheme 1
cons_1 = ({'type': 'ineq',
           'fun': lambda n: np.dot(cash_flow_matrix, n) - liabilities,
           'jac': lambda n: cash_flow_matrix},
          {'type': 'ineq',
           'fun': lambda n: n - 1e-6,
           'jac': lambda n: np.eye(n_bonds)})
res_1 = sp.optimize.minimize(lambda n: np.dot(price_vector, n),
                             x0=np.ones(n_bonds) / n_bonds,
                             constraints=cons_1,
                             options={'disp': True})
n_vector = res_1.x
print("###########################################################")
print("scheme 1: match with cash flow")
print("n vector: " + str(n_vector))
print("number of each bond to be bought: " + str((n_vector * 1000000).astype(int)))
print("portfolio cash flow: " + str(np.dot(cash_flow_matrix, n_vector)))
print("portfolio cost: " + "{:.4e} million dollars".format(
    np.dot(n_vector, price_vector)))
print("portfolio profit: " + "{:.4e} million dollars".format(
    np.sum(np.dot(cash_flow_matrix, n_vector))))
print("portfolio net earning: " + "{:.4e} million dollars".format(
    np.sum(np.dot(cash_flow_matrix, n_vector)) - np.dot(n_vector, price_vector)))
print("###########################################################")


##########################################################################
# scheme 2
def compute_gross_reinvestment_rates_matrix(n_dates, interest_rates_vector=None):
    if (interest_rates_vector):
        if (len(interest_rates_vector) == n_dates):
            return np.diag(interest_rates_vector)
        else:
            print('check n_dates and len(interest_rates_vector) should be equal')
            return
    else:
        return np.eye(n_dates)


R_matrix = compute_gross_reinvestment_rates_matrix(n_dates=n_coupon_dates)

J_matrix = np.zeros((n_coupon_dates, n_coupon_dates))
for i in range(n_coupon_dates - 1):
    J_matrix[i + 1, i] = 1

Y_matrix = np.dot(J_matrix, R_matrix) - np.eye(n_coupon_dates)
pi_vector = np.hstack((price_vector, np.zeros(n_coupon_dates)))
init_theta_vector = np.ones(n_bonds + n_coupon_dates) / n_bonds
constraint_matrix = np.hstack((cash_flow_matrix, Y_matrix))
# print("######################################")
# print(constraint_matrix)
# print(pi_vector)
# print(init_theta_vector)
# print(liabilities)

cons_2 = ({
              'type': 'eq',
              'fun': lambda theta:
              np.dot(constraint_matrix, theta) - liabilities,
              'jac': lambda n: constraint_matrix
          },
          {
              'type': 'ineq',
              'fun': lambda theta: theta,
              'jac': lambda theta: np.eye(n_bonds + n_coupon_dates)
          })
res_2 = sp.optimize.minimize(lambda theta: np.dot(theta, pi_vector),
                             x0=init_theta_vector,
                             constraints=cons_2,
                             options={'disp': True}
                             )
theta_vector = res_2.x
# print("theta:"+str(theta_vector))
# print("init theta:"+str(init_theta_vector))
n_vector2 = theta_vector[:n_bonds]
print("###########################################################")
print("scheme 2: match with cash carry-forword")
print("n vector: " + str(n_vector2))
print("number of each bond to be bought: "
      + str((n_vector2 * 1000000).astype(int)))
print("portfolio cash flow: " + str(np.dot(cash_flow_matrix, n_vector2)))
print("portfolio cost: " + "{:.4e} million dollars".format(
    np.dot(n_vector2, price_vector)))
print("portfolio profit: " + "{:.4e} million dollars".format(
    np.sum(np.dot(cash_flow_matrix, n_vector2))))
print("portfolio net earning: " + "{:.4e} million dollars".format(
    np.sum(np.dot(cash_flow_matrix, n_vector2))
    - np.dot(n_vector2, price_vector)))
print("###########################################################")


#############################################################################
# 8.3
def compute_bond_duration(bond_price, cash_flow_vector, n_coupons_per_year,
                          annual_interest_rate=0):
    bond_duration = 0
    for k in range(len(cash_flow_vector)):
        bond_duration += k * cash_flow_vector[k] \
                         / (1 + annual_interest_rate / n_coupons_per_year) ** k
    bond_duration /= n_coupons_per_year * bond_price
    return bond_duration


def compute_bond_portfolio_duration(n_vector, price_vector, duration_vector,
                                    portfolio_price):
    bond_portfolio_duration = \
        np.sum(n_vector * price_vector * duration_vector) / portfolio_price
    return bond_portfolio_duration


annual_interest_rate = 0
df['duration'] = np.zeros(n_bonds)
bond_duration_vector = []
for i in range(n_bonds):
    cash_flow_vector = cash_flow_matrix[:, i]
    bond_price = df.price[i]
    duration = compute_bond_duration(bond_price, cash_flow_vector,
                                     2, annual_interest_rate)
    bond_duration_vector.append(duration)

df.duration = bond_duration_vector
print(df)

#############################################################################
# 8.4
portfolio_price1 = np.dot(n_vector, price_vector)
portfolio_price2 = np.dot(n_vector2, price_vector)
portfolio_cash_flow1 = np.dot(cash_flow_matrix, n_vector)
portfolio_cash_flow2 = np.dot(cash_flow_matrix, n_vector2)

# scheme 1: using portfolio cash flow
print("scheme 1")
portfolio_duration1 = \
    compute_bond_duration(portfolio_price1, portfolio_cash_flow1,
                          2, annual_interest_rate)
portfolio_duration2 = \
    compute_bond_duration(portfolio_price2, portfolio_cash_flow2,
                          2, annual_interest_rate)
print("portfolio1's duration: " + str(portfolio_duration1))
print("portfolio2's duration: " + str(portfolio_duration2))

# scheme 2: using each bonds cash flows
print("scheme 2")
portfolio_duration11 = \
    compute_bond_portfolio_duration(n_vector, price_vector,
                                    bond_duration_vector,
                                    portfolio_price1)
portfolio_duration22 = \
    compute_bond_portfolio_duration(n_vector2, price_vector,
                                    bond_duration_vector,
                                    portfolio_price2)
print("portfolio1's duration: " + str(portfolio_duration11))
print("portfolio2's duration: " + str(portfolio_duration22))
