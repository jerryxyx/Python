import numpy as np
from scipy.stats import norm
import math
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

    std_approxy = 0.4 * np.sqrt(T)
    a_approxy = mean - quantile*std_approxy
    b_approxy = mean + quantile*std_approxy
    # a=-5
    # b=5
    return (a,b)
def calculateToleranceInterval_v2(S0,strike,T,r,q,sigmaBSM,quantile):
    if sigmaBSM<0:
        sigmaBSM==0.1
    mean = np.log(S0/strike)+(r-q-.5*sigmaBSM**2)*T
    variance = sigmaBSM**2*T
    std = np.sqrt(variance)
    a = mean - quantile*std
    b = mean + quantile*std

    std_approxy = 0.2 * np.sqrt(T)
    a_approxy = mean - quantile*std_approxy
    b_approxy = mean + quantile*std_approxy
    # a=-5
    # b=5
    return (a,b)
# def calculateToleranceIntervalWithoutSigma(S0,strike,T,r,q,quantile):
#     a = np.log(S0/strike)+(r-q-.5)*T - quantile
#     b = np.log(S0/strike)+(r-q)*T + quantile
#     return (a,b)

def calculateNumGrid(T,sigmaBSM,quantile):
    numGrid = int(10*quantile*sigmaBSM*np.sqrt(T))
    return numGrid

def calculateConstantTerm(S0,strike,T,r,q,a):
    return np.log(S0/strike) + (r-q)*T - a

def calculateNumGrid2(numGrid1,T,sigma,a,b):
    N1 = numGrid1
    ck = 2*math.pi*(N1-1)/(b-a)
    z = ck**2/2*sigma**2*T
    N2 = math.e*z*math.exp(1/math.e)
    v1=math.pow(math.exp(z),1/N2)
    v2=math.pow(2*math.pi,1/(2*N2))
    v3=math.pow(N2,1/(2*N2))
    # print("calculate N2:",N2,v1,v2,v3)
    N2 = int(N2+1)
    return N2

# todo: estimate
def calculateErrorUpperBound(S0,strike,r,q,T,sigmaBSM,N,quantile,showDetails=False):
    mean = (r-q-sigmaBSM**2/2)*T + np.log(S0/strike)
    (a,b) = calculateToleranceInterval(S0,strike,T,r,q,sigmaBSM,quantile)
    # (a, b) = calculateToleranceIntervalWithoutSigma(S0, strike, T, r, q, quantile)
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