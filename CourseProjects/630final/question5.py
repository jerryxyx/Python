import numpy as np
import scipy as sp
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt

# mu = np.array([4,5,6])*0.01
# beta = np.array([0.9,1,1.1])
# omega = np.array([0.18,0.20,0.22])
alpha_array = np.array([10,9,11,5,8,7,5,4,8,3])*0.01
beta_array = np.array([1.2,1.1,1.3,0.8,0.9,0.7,0.6,0.6,0.8,0.5])
omega_array = np.array([30,25,27,20,22,20,20,29,24,20])*0.01
b_array = np.array([10,12,12,6,21,8,6,8,6,11])*0.01
sigmaM = 0.2    # market standard deviation
muM = 0.07  # expected market return
rf = 0.03   # risk-free rate
re_array = alpha_array+beta_array*muM   # expected securities return array
rb = np.sum(re_array*b_array)   # benchmark return
n_features = len(beta_array)    # number of securities
eta = np.ones(n_features)

df = pd.DataFrame(columns=['alpha','beta','omega','b'])
df.alpha = alpha_array
df.beta = beta_array
df.omega = omega_array
df.b = b_array
print(df)
def compute_expected_alpha(h,alpha):
    return np.dot(h,alpha)

def compute_expected_return(h,alpha,beta,muM):
    return np.dot(h,alpha+beta*muM)

def compute_standard_deviation(h,cov):
    var = np.dot(np.dot(h.T, cov), h)
    std = np.sqrt(var)
    return std

def compute_covariance(beta,omega,sigmaM):
    n_features = len(omega)
    cov = np.zeros((n_features,n_features))
    for i in range(n_features):
        for j in range(n_features):
            cov[i,j] = beta[i]*beta[j]*sigmaM**2
            if i==j:
                cov[i,j] += omega[i]**2
    return cov

cov = compute_covariance(beta_array,omega_array,sigmaM)

def compute_portfolio_variance(h,Q):
    return np.dot(np.dot(h.T,Q),h)

# print(compute_portfolio_variance(np.array([1,2]),np.array([[1,2],[3,4]])))
cons = ({'type':'eq',
         'fun' : lambda h: np.dot(h,eta)-1,
         'jac' : lambda h: eta})

varP = lambda h: compute_portfolio_variance(h,Q=cov)
negativeAdjReturn = lambda h,tau: -((np.dot(h,alpha_array) + np.dot(h,beta_array)*muM)*tau - 1/2 * varP(h))

########################################################################################################
# 5.1
print("Q5.1, minium variance portfolio:")
res = sp.optimize.minimize(negativeAdjReturn, np.zeros(n_features), args=(0,),constraints=cons, options={'disp': True})
h_op = res.x
print("numerical result:")
print("h:")
print(h_op)
print("expected return:"+str(np.sum(h_op*(alpha_array+beta_array*muM))))
var1_1 = compute_portfolio_variance(h_op,cov)
print("var:"+str(var1_1))
print("std:"+str(np.sqrt(var1_1)))
plt.scatter(np.sqrt(var1_1),np.sum(h_op*(alpha_array+beta_array*muM)),marker='o',c='g')

# h_class = np.array([0.1135,0.3628,0.5237])
# print(compute_portfolio_variance(h_class,cov))
# muE = mu - eta*muM

# h_T = np.dot(np.matrix(cov).I,muE)/np.dot(np.dot(eta,np.matrix(cov).I),muE)
# h_T = np.array(h_T)[0,:]
# print(h_T)
# print(compute_portfolio_variance(h_T,cov))

h_v = np.dot(np.matrix(cov).I,eta)/np.dot(np.dot(eta,np.matrix(cov).I),eta)
h_v = np.array(h_v)[0,:]
print("using matrix formula")
print("h:")
print(h_v)
print("expected return:"+str(np.sum(h_v*(alpha_array+beta_array*muM))))
var1_2 = compute_portfolio_variance(h_v,cov)
print("var:"+str(var1_2))
print("std:"+str(np.sqrt(var1_2)))

########################################################################################################
# 5.2
print("Q5.2, EGP with Sharpe's single index model, long-only:")
def compute_traynor_ratio(mu,beta,rf):
    return (mu-rf)/beta

def compute_cumulative_threshold(mu_list, beta_list, omega_list, sigmaM, rf):
    mu_array = np.array(mu_list)
    beta_array = np.array(beta_list)
    omega_array = np.array(omega_list)
    numerator = sigmaM**2 * np.sum((mu_array-rf)/omega_array**2 * beta_array)
    denominator = 1 + sigmaM**2 * np.sum(beta_array**2/omega_array**2)
    return numerator/denominator

c_array = compute_traynor_ratio(alpha_array, beta_array, rf)
# print(c_array)
cstar_array = np.zeros(n_features)
for i in range(n_features):
    cstar_array[i] = compute_cumulative_threshold(alpha_array[:i+1],beta_array[:i+1],omega_array[:i+1],sigmaM,rf)

# print(cstar_array)
df['c'] = c_array
df['cstar'] = cstar_array
inportfolio = c_array>cstar_array
thecstar = np.max(cstar_array)
z_array = (alpha_array-rf)/omega_array**2 -beta_array*thecstar/omega_array**2
df['z'] = z_array
x_array = z_array[inportfolio]/np.sum(z_array[inportfolio])
df['x']=np.zeros(n_features)
df.x[inportfolio] = x_array
print(df.sort_values(by='c',ascending=False))
r2 = np.sum((df.alpha+df.beta*muM)*df.x)
print("expected return: "+str(r2))
var2 = compute_portfolio_variance(df.x,compute_covariance(df.beta,df.omega,sigmaM))
print("variance:" + str(var2))
print("std:"+str(np.sqrt(var2)))
plt.scatter(np.sqrt(var2),r2,marker='^',c='r')
########################################################################################################
# 5.3
cstarN = compute_cumulative_threshold(alpha_array,beta_array,omega_array,sigmaM,rf)
df2 = df.copy()
x_array2 = z_array/np.sum(np.abs(z_array))
df2.x = x_array2
print(df2.sort_values(by='c',ascending=False))
r3 = np.sum((df2.alpha+df2.beta*muM)*df2.x)
print("expected return: "+str(r3))
var3 = compute_portfolio_variance(df2.x,compute_covariance(df2.beta,df2.omega,sigmaM))
print("variance:" + str(var3))
print("std:"+str(np.sqrt(var3)))
plt.scatter(np.sqrt(var3),r3,marker='s',c='b')
########################################################################################################
# 5.4
cov = compute_covariance(beta_array,omega_array,sigmaM)
print('rb:'+str(rb))
cons4 = ({'type':'eq',
         'fun' : lambda h: np.dot(h,eta)-1,
         'jac' : lambda h: eta},
         {'type':'eq',
          'fun' : lambda h: np.dot(h,beta_array)-1,
          'jac' : lambda h: beta_array},
         {'type':'ineq',
          'fun' : lambda h: h})

re_list = []
std_list = []
h_list = []
for i in range(100):
    taui = 0.001+i*(1-0.001)/99
    res = sp.optimize.minimize(negativeAdjReturn, np.zeros(n_features), args=(taui,),constraints=cons4, options={'disp': False})
    h_i = res.x
    # print("h:"+str(h_i))
    re_i =np.sum(h_i * (alpha_array + beta_array * muM))
    # print("expected return:" +str(re_i))
    var_i = compute_portfolio_variance(h_i, cov)
    std_i = np.sqrt(var_i)
    # print("std:" + str(std_i))
    re_list.append(re_i)
    std_list.append(std_i)
    h_list.append(h_i)
print(np.array(h_list)>0.2)
plt.plot(std_list,re_list)
# plt.show()

########################################################################################################
# 5.5
cov = compute_covariance(beta_array,omega_array,sigmaM)
print('rb:'+str(rb))
cons4 = ({'type':'eq',
         'fun' : lambda h: np.dot(h,eta)-1,
         'jac' : lambda h: eta},
         {'type':'eq',
          'fun' : lambda h: np.dot(h,beta_array)-1,
          'jac' : lambda h: beta_array},
         {'type':'ineq',
          'fun' : lambda h: h},
         {'type':'ineq',
          'fun' : lambda h: np.ones(n_features)*0.2-1e-5 - h})

re_list2 = []
std_list2 = []
h_list2 = []
for i in range(100):
    taui = 0.001+i*(1-0.001)/99
    res = sp.optimize.minimize(negativeAdjReturn, np.zeros(n_features),
                               args=(taui,),constraints=cons4, options={'disp': False})
    h_i = res.x
    # print("h:"+str(h_i))
    re_i =np.sum(h_i * (alpha_array + beta_array * muM))
    # print("expected return:" +str(re_i))
    var_i = compute_portfolio_variance(h_i, cov)
    std_i = np.sqrt(var_i)
    # print("std:" + str(std_i))
    re_list2.append(re_i)
    std_list2.append(std_i)
    h_list2.append(h_i)

print(np.array(h_list2)>0.2)
plt.plot(std_list2,re_list2)
# plt.show()

########################################################################################################
# 6.1

mub = compute_expected_return(b_array,alpha_array,beta_array,muM)   # expected return of benchmark
cov = compute_covariance(beta_array,omega_array,sigmaM) # covariance of securities

def compute_expected_active_return(h,b,alpha_array,beta_array,muM):
    return compute_expected_return(h-b,alpha_array,beta_array,muM)
    # return np.dot(h-b,alpha_array + beta_array * muM)

def compute_tracking_error(h_array,b_array,cov):
    active_weight_vector = h_array-b_array
    tracking_var = np.dot(np.dot(active_weight_vector,cov),active_weight_vector)
    tracking_error = np.sqrt(tracking_var)
    return tracking_error

def compute_tracking_var(h_array,b_array,cov):
    active_weight_vector = h_array-b_array
    tracking_var = np.dot(np.dot(active_weight_vector,cov),active_weight_vector)
    return tracking_var

def compute_expected_residual_return(h,b,alpha_array,beta_array,muM):   # I assume it as the expected alpha of the portfolio
    mu_p = compute_expected_return(h,alpha_array,beta_array,muM)
    mu_b = compute_expected_return(b,alpha_array,beta_array,muM)
    beta_p = np.dot(h,beta_array)
    residual_return = mu_p - beta_p * mu_b
    return residual_return


cons6_1 = ({'type': 'eq', # budget constraint
          'fun': lambda h: np.dot(h, eta) - 1,
          'jac' : lambda h: eta
          },
         {'type': 'eq', # tracking error constraint
          'fun': lambda h: compute_tracking_error(h,b_array,cov) - 0.03, # tracking error = 3%
          'jac' : lambda h:  np.dot(cov,h-b_array)/(compute_tracking_error(h,b_array,cov)+1e-10)
            },
         # {'type' : 'eq',
         #  'fun' : lambda h: np.dot(h-b_array,beta_array)}
         # {'type' : 'ineq',  # long-only constraint
         #  'fun' : lambda h: h,
         #  'jac' : lambda h: np.eye(n_features)},
         # {'type': 'ineq',   # boundary constraint
         #  'fun' : lambda h: eta-h,
         #  'jac' : lambda h: -np.eye(n_features)}
         )

res6_1 = sp.optimize.minimize(lambda h: -compute_expected_active_return(h,b_array,alpha_array,beta_array,muM),
                              x0=eta/n_features, constraints=cons6_1, tol=1e-8, options={'disp':False})
h_6_1 = res6_1.x
print("#######################################################################")
print("portfolio that maximize the portfolio's expected active return under only budget constraint "
      "and tracking error constraint:")
print("portfolio h:" + str(h_6_1))
print("portfolio total weight:" + str(np.dot(h_6_1,eta)))
print("portfolio expected active return:" + str(compute_expected_active_return(h_6_1,b_array,alpha_array,beta_array,muM)))
print("portfolio expected active alpha:" + str(compute_expected_alpha(h_6_1-b_array,alpha_array)))
print("portfolio expected return:" + str(compute_expected_return(h_6_1,alpha_array,beta_array,muM)))
print("benchmark expected return:" + str(mub))
print("portfolio standard deviation:" + str(compute_standard_deviation(h_6_1,cov)))
print("benchmark standard deviation:" + str(compute_standard_deviation(b_array,cov)))
print("portfolio tracking error:" + str(compute_tracking_error(h_6_1,b_array,cov)))
print("portfolio beta:" + str(np.dot(h_6_1,beta_array)))
print("portfolio expected residual return:" +str(compute_expected_residual_return(h_6_1,b_array,alpha_array,beta_array,muM)))
print("portfolio information ratio:"
      + str(compute_expected_alpha(h_6_1-b_array,alpha_array)/compute_tracking_error(h_6_1,b_array,cov)))
print("#######################################################################")


cons6_2 = ({'type': 'eq', # budget constraint
          'fun': lambda h: np.dot(h, eta) - 1,
          'jac' : lambda h: eta
          },
         {'type': 'eq', # tracking error constraint
          'fun': lambda h: compute_tracking_error(h,b_array,cov) - 0.03, # tracking error = 3%
          'jac' : lambda h:  np.dot(cov,h-b_array)/(compute_tracking_error(h,b_array,cov)+1e-10)
            },
         {'type' : 'eq',
          'fun' : lambda h: np.dot(h-b_array,beta_array)}
         # {'type' : 'ineq',  # long-only constraint
         #  'fun' : lambda h: h,
         #  'jac' : lambda h: np.eye(n_features)},
         # {'type': 'ineq',   # boundary constraint
         #  'fun' : lambda h: eta-h,
         #  'jac' : lambda h: -np.eye(n_features)}
         )
res6_2 = sp.optimize.minimize(lambda h: -compute_expected_alpha(h-b_array,alpha_array),
                              x0=eta/n_features, constraints=cons6_2, options={'disp':False})
h_6_2 = res6_2.x
print("#######################################################################")
print("portfolio that maximize the portfolio's expected market alpha under budget constraint, tracking error constraint "
      "and active market beta constraint(active_weights * beta == 0)")
print("portfolio h:" + str(h_6_2))
print("portfolio total weight:" + str(np.dot(h_6_2,eta)))
print("portfolio expected active return:" + str(compute_expected_active_return(h_6_2,b_array,alpha_array,beta_array,muM)))
print("portfolio expected active alpha:" + str(compute_expected_alpha(h_6_2-b_array,alpha_array)))
print("portfolio expected residual return:" +str(compute_expected_residual_return(h_6_2,b_array,alpha_array,beta_array,muM)))
print("portfolio expected return:" + str(compute_expected_return(h_6_2,alpha_array,beta_array,muM)))
print("benchmark expected return:" + str(mub))
print("portfolio standard deviation:" + str(compute_standard_deviation(h_6_2,cov)))
print("benchmark standard deviation:" + str(compute_standard_deviation(b_array,cov)))
print("portfolio tracking error:" + str(compute_tracking_error(h_6_2,b_array,cov)))
print("portfolio beta:" + str(np.dot(h_6_2,beta_array)))
print("portfolio information ratio:"
      + str(compute_expected_alpha(h_6_2-b_array,alpha_array)/compute_tracking_error(h_6_2,b_array,cov)))
print("#######################################################################")