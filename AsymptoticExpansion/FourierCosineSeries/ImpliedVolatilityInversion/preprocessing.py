import numpy as np

def generateTruncatedInterval(S0,strike,T,r,q,sigmaBSM,model):
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

def calculateConstantTerm(S0,strike,T,r,q,a):
    return np.log(S0/strike) + (r-q)*T -a