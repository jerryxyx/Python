import numpy as np
from scipy.stats import norm

def generateTruncatedInterval_empirical(S0,strike,T,r,q,sigmaBSM,model="BSM"):
    # S0 and strike can be a integer or an array.
    # Example:
    # print(generateTruncatedInterval(50, 50, 0.1, 0.01, 0, 0.25, "BSM"))
    # print(generateTruncatedInterval(50, 50, 0.1, 0.01, 0, 0.25, "Heston"))
    initialValue = np.log(S0/strike)
    meanValue = initialValue + T*(r-q-0.5*sigmaBSM**2)
    gaussianVariance = T*sigmaBSM**2
    # How many std should we truncate
    if T >= 2:
        L1 = 14
        L2 = 14
    elif T >= .1:
        # L1 = 18
        # L2 = 20
        L1 = 12
        L2 = 20

    else:
        # L1 = 25
        # L2 = 28
        L1 = 28
        L2 = 28

    if model == "Heston":
        a = meanValue-L2*np.sqrt(gaussianVariance)
        b = meanValue+L2*np.sqrt(gaussianVariance)
    elif model == "BSM":
        a = meanValue - L1*np.sqrt(gaussianVariance)
        b = meanValue + L1*np.sqrt(gaussianVariance)

    return (a,b)


def calculateToleranceInterval(S0,strike,T,r,q,sigmaBSM,quantile):
    mean = np.log(S0/strike)+(r-q-.5*sigmaBSM**2)*T
    variance = sigmaBSM**2*T
    std = np.sqrt(variance)
    a = mean - quantile*std
    b = mean + quantile*std
    return (a,b)

def calculateNumGrid(T,sigmaBSM,quantile):
    numGrid = int(10*quantile*sigmaBSM*np.sqrt(T))
    return numGrid

def calculateConstantTerm(S0,strike,T,r,q,a):
    return np.log(S0/strike) + (r-q)*T - a
# todo: estimate
def calculateErrorUpperBound(S0,strike,r,q,T,sigmaBSM,N,quantile,showDetails=False):
    mean = (r-q-sigmaBSM**2/2)*T + np.log(S0/strike)
    (a,b) = calculateToleranceInterval(S0,strike,T,r,q,sigmaBSM,quantile)
    # error introduced by integral truncation
    error1 = strike*max(1-np.exp(a),0)*norm.cdf(-quantile)

    # error introduced by series truncation
    C = 1
    error2 = np.exp(-r*T)*(strike*0.5)*C/N**2

    # error introduced by approximate Ak by Fk
    error3 = N*np.exp(-r*T)/(quantile*sigmaBSM*np.sqrt(T)) * norm.cdf(-quantile)

    if(showDetails==True):
        print("error caused by integral truncation:",error1)
        print("error caused by series truncation:",error2)
        print("error caused by approximating Ak by Fk:",error3)
    errorBound = error1+error2+error3

    return errorBound