from question5 import *

# expected return of benchmark
mub = compute_expected_return(b_array, alpha_array, beta_array, muM)
# covariance of securities
cov = compute_covariance(beta_array, omega_array, sigmaM)


def compute_expected_active_return(h, b, alpha_array, beta_array, muM):
    return compute_expected_return(h - b, alpha_array, beta_array, muM)
    # return np.dot(h-b,alpha_array + beta_array * muM)


def compute_tracking_error(h_array, b_array, cov):
    active_weight_vector = h_array - b_array
    tracking_var = \
        np.dot(np.dot(active_weight_vector, cov), active_weight_vector)
    tracking_error = np.sqrt(tracking_var)
    return tracking_error


def compute_tracking_var(h_array, b_array, cov):
    active_weight_vector = h_array - b_array
    tracking_var = \
        np.dot(np.dot(active_weight_vector, cov), active_weight_vector)
    return tracking_var


# I thought it was the supposed definition of expected alpha of the portfolio
def compute_expected_residual_return(h, b, alpha_array, beta_array, muM):
    mu_p = compute_expected_return(h, alpha_array, beta_array, muM)
    mu_b = compute_expected_return(b, alpha_array, beta_array, muM)
    beta_p = np.dot(h, beta_array)
    residual_return = mu_p - beta_p * mu_b
    return residual_return


cons6_1 = ({'type': 'eq',  # budget constraint
            'fun': lambda h: np.dot(h, eta) - 1,
            'jac': lambda h: eta
            },
           {'type': 'eq',  # tracking error constraint
            'fun':
                lambda h: compute_tracking_error(h, b_array, cov) - 0.03,

            'jac':
                lambda h: np.dot(cov, h - b_array)
                          / (compute_tracking_error(h, b_array, cov) + 1e-10)
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
res6_1_0 = sp.optimize.minimize(
    lambda h: -np.dot(h, alpha_array),
    x0=eta / n_features, constraints=cons6_1,
    tol=1e-8, options={'disp': False})
h_6_1_0 = res6_1_0.x
print("#######################################################################")
print("portfolio that maximize the portfolio's "
      "expected alpha under only budget constraint\n"
      "and tracking error constraint:")
print("portfolio h:" + str(h_6_1_0))
print("portfolio total weight:" + str(np.dot(h_6_1_0, eta)))
print("portfolio expected active return:"
      + str(compute_expected_active_return(h_6_1_0, b_array, alpha_array, beta_array, muM)))
print("portfolio expected active alpha:"
      + str(compute_expected_alpha(h_6_1_0 - b_array, alpha_array)))
print("portfolio expected return:"
      + str(compute_expected_return(h_6_1_0, alpha_array, beta_array, muM)))
print("benchmark expected return:"
      + str(mub))
print("portfolio standard deviation:"
      + str(compute_standard_deviation(h_6_1_0, cov)))
print("benchmark standard deviation:"
      + str(compute_standard_deviation(b_array, cov)))
print("portfolio tracking error:"
      + str(compute_tracking_error(h_6_1_0, b_array, cov)))
print("portfolio beta:"
      + str(np.dot(h_6_1_0, beta_array)))
print("portfolio expected residual return:"
      + str(compute_expected_residual_return(h_6_1_0, b_array, alpha_array, beta_array, muM)))
print("portfolio information ratio:"
      + str(compute_expected_alpha(h_6_1_0 - b_array, alpha_array)
            / compute_tracking_error(h_6_1_0, b_array, cov)))
print("#######################################################################")

res6_1 = sp.optimize.minimize(
    lambda h: -compute_expected_active_return(h, b_array, alpha_array, beta_array, muM),
    x0=eta / n_features, constraints=cons6_1, tol=1e-8, options={'disp': False})
h_6_1 = res6_1.x
print("#######################################################################")
print("portfolio that maximize the portfolio's expected active return "
      "under only budget constraint \n"
      "and tracking error constraint:")
print("portfolio h:" + str(h_6_1))
print("portfolio total weight:" + str(np.dot(h_6_1, eta)))
print("portfolio expected active return:"
      + str(compute_expected_active_return(h_6_1, b_array, alpha_array, beta_array, muM)))
print("portfolio expected active alpha:"
      + str(compute_expected_alpha(h_6_1 - b_array, alpha_array)))
print("portfolio expected return:"
      + str(compute_expected_return(h_6_1, alpha_array, beta_array, muM)))
print("benchmark expected return:" + str(mub))
print("portfolio standard deviation:" + str(compute_standard_deviation(h_6_1, cov)))
print("benchmark standard deviation:" + str(compute_standard_deviation(b_array, cov)))
print("portfolio tracking error:" + str(compute_tracking_error(h_6_1, b_array, cov)))
print("portfolio beta:" + str(np.dot(h_6_1, beta_array)))
print("portfolio expected residual return:"
      + str(compute_expected_residual_return(h_6_1, b_array, alpha_array, beta_array, muM)))
print("portfolio information ratio:"
      + str(compute_expected_alpha(h_6_1 - b_array, alpha_array)
            / compute_tracking_error(h_6_1, b_array, cov)))
print("#######################################################################")

cons6_2 = ({'type': 'eq',  # budget constraint
            'fun': lambda h: np.dot(h, eta) - 1,
            'jac': lambda h: eta
            },
           {'type': 'eq',  # tracking error constraint
            'fun': lambda h: compute_tracking_error(h, b_array, cov) - 0.03,
            'jac':
                lambda h: np.dot(cov, h - b_array)
                          / (compute_tracking_error(h, b_array, cov) + 1e-10)
            },
           {'type': 'eq',
            'fun': lambda h: np.dot(h - b_array, beta_array)}
           )

res6_2_0 = sp.optimize.minimize(lambda h: -np.dot(h, alpha_array),
                                x0=eta / n_features,
                                constraints=cons6_2,
                                options={'disp': False})
h_6_2_0 = res6_2_0.x
print("#######################################################################")
print("portfolio that maximize the portfolio's expected alpha"
      " under budget constraint,"
      " \ntracking error constraint "
      "and active market beta constraint(active_weights * beta == 0)")
print("portfolio h:" + str(h_6_2_0))
print("portfolio total weight:" + str(np.dot(h_6_2_0, eta)))
print("portfolio expected active return:"
      + str(compute_expected_active_return(h_6_2_0, b_array, alpha_array, beta_array, muM)))
print("portfolio expected active alpha:"
      + str(compute_expected_alpha(h_6_2_0 - b_array, alpha_array)))
print("portfolio expected residual return:"
      + str(compute_expected_residual_return(h_6_2_0, b_array, alpha_array, beta_array, muM)))
print("portfolio expected return:"
      + str(compute_expected_return(h_6_2_0, alpha_array, beta_array, muM)))
print("benchmark expected return:" + str(mub))
print("portfolio standard deviation:" + str(compute_standard_deviation(h_6_2_0, cov)))
print("benchmark standard deviation:" + str(compute_standard_deviation(b_array, cov)))
print("portfolio tracking error:" + str(compute_tracking_error(h_6_2_0, b_array, cov)))
print("portfolio beta:" + str(np.dot(h_6_2_0, beta_array)))
print("portfolio information ratio:"
      + str(compute_expected_alpha(h_6_2_0 - b_array, alpha_array)
            / compute_tracking_error(h_6_2_0, b_array, cov)))
print("#######################################################################")

res6_2 = sp.optimize.minimize(lambda h: -np.dot(h - b_array, mu_array),
                              x0=eta / n_features,
                              constraints=cons6_2,
                              options={'disp': False})
h_6_2 = res6_2.x
print("#######################################################################")
print("portfolio that maximize the portfolio's expected active return"
      " under budget constraint, \ntracking error constraint "
      "and active market beta constraint(active_weights * beta == 0)")
print("portfolio h:" + str(h_6_2))
print("portfolio total weight:" + str(np.dot(h_6_2, eta)))
print("portfolio expected active return:"
      + str(compute_expected_active_return(h_6_2, b_array, alpha_array, beta_array, muM)))
print("portfolio expected active alpha:"
      + str(compute_expected_alpha(h_6_2 - b_array, alpha_array)))
print("portfolio expected residual return:"
      + str(compute_expected_residual_return(h_6_2, b_array, alpha_array, beta_array, muM)))
print("portfolio expected return:"
      + str(compute_expected_return(h_6_2, alpha_array, beta_array, muM)))
print("benchmark expected return:" + str(mub))
print("portfolio standard deviation:" + str(compute_standard_deviation(h_6_2, cov)))
print("benchmark standard deviation:" + str(compute_standard_deviation(b_array, cov)))
print("portfolio tracking error:" + str(compute_tracking_error(h_6_2, b_array, cov)))
print("portfolio beta:" + str(np.dot(h_6_2, beta_array)))
print("portfolio information ratio:"
      + str(compute_expected_alpha(h_6_2 - b_array, alpha_array)
            / compute_tracking_error(h_6_2, b_array, cov)))
print("#######################################################################")
