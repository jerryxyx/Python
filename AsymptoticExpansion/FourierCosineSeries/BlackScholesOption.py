import numpy as np
from scipy.stats import norm
import time

def d(S0,strike,T,r,q,sigmaBSM):
    # F0 = S0*np.exp((r-q)*T)
    d1 = (np.log(S0/strike)+(r-q+sigmaBSM**2*T/2))/sigmaBSM*np.sqrt(T)
    d2 = d1 - sigmaBSM*np.sqrt(T)
    # print(d1,d2)
    return (d1,d2)

def putOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,showDuration=False):
    tick = time.time()
    F0 = S0 * np.exp((r - q) * T)
    (d1,d2) = d(S0,strike,T,r,q,sigmaBSM)
    P = strike*np.exp(-r*T)*norm.cdf(-d2) - F0*np.exp(-r*T)*norm.cdf(-d1)
    tack = time.time()
    if(showDuration==True):
        print("consuming time for put option using BSM:",tack-tick)
    return P

def callOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,showDuration=False):
    tick = time.time()
    F0 = S0 * np.exp((r - q) * T)
    (d1,d2) = d(S0,strike,T,r,q,sigmaBSM)
    C = -strike*np.exp(-r*T)*norm.cdf(d2) + F0*np.exp(-r*T)*norm.cdf(d1)
    tack = time.time()
    if (showDuration == True):
        print("consuming time for call option using BSM:", tack - tick)
    return C