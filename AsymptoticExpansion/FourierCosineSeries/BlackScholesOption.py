import numpy as np
# from scipy.stats import norm
from math import erf, sqrt
from decimal import Decimal
import time

def d(S0,strike,T,r,q,sigmaBSM):
    # F0 = S0*np.exp((r-q)*T)
    d1 = (np.log(S0/strike)+(r-q+sigmaBSM**2/2)*T)/(sigmaBSM*sqrt(T))
    d2 = d1 - sigmaBSM*np.sqrt(T)
    # print(d1,d2)
    return (d1,d2)

def norm_cdf(x):
    cdf = 0.5+0.5*erf(x/sqrt(2))
    return cdf

def putOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,showDuration=False):
    tick = time.time()
    F0 = S0 * np.exp((r - q) * T)
    (d1,d2) = d(S0,strike,T,r,q,sigmaBSM)
    # P = strike*np.exp(-r*T)*norm.cdf(-d2) - F0*np.exp(-r*T)*norm.cdf(-d1)
    P = strike*np.exp(-r*T)*norm_cdf(-d2) - S0*np.exp(-q*T)*norm_cdf(-d1)
    tack = time.time()
    if(showDuration==True):
        print("consuming time for put option using BSM:",tack-tick)
    return P

def callOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,showDuration=False):
    tick = time.time()
    F0 = S0 * np.exp((r - q) * T)
    (d1,d2) = d(S0,strike,T,r,q,sigmaBSM)
    # C = -strike*np.exp(-r*T)*norm.cdf(d2) + F0*np.exp(-r*T)*norm.cdf(d1)
    C = -strike * np.exp(-r * T) * norm_cdf(d2) + F0 * np.exp(-r * T) * norm_cdf(d1)
    tack = time.time()
    if (showDuration == True):
        print("consuming time for call option using BSM:", tack - tick)
    return C