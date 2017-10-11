from Vk_utils import calculateVkPut
import numpy as np
# from preprocessing import generateTruncatedInterval, calculateConstantTerm
import preprocessing
import time

def chfBSM(x,S0,strike,r,q,T,sigmaBSM):
    chfBSM = np.exp( complex( -0.5 * x**2 * sigmaBSM**2 *T, x*(np.log(S0/strike)+(r-q)*T)) )
    return chfBSM

def calculateRk(S0,strike,T,r,q,sigmaBSM,a,b,numGrid):
    ckList = np.array([k*np.pi/(b-a) for k in range(numGrid)])
    m = preprocessing.calculateConstantTerm(S0, strike, T, r, q, a)
    Rk = np.exp(-ckList**2*T*sigmaBSM**2/2)*np.cos(ckList*(m-T*sigmaBSM**2/2))
    return Rk

def calculateRk_chf(S0,strike,T,r,q,sigmaBSM,a,b,numGrid):
    ckList = np.array([k*np.pi/(b-a) for k in range(numGrid)])
    # ckList = np.linspace(0,numGrid-1,numGrid)*np.pi/(b-a)
    # m = preprocessing.calculateConstantTerm(S0,strike,T,r,q,a)
    realPart = -ckList**2*sigmaBSM**2/2*T
    imagePart = ckList*(np.log(S0/strike)-a+(r-q-sigmaBSM**2/2)*T)
    c = [complex(rr,ii) for rr,ii in zip(realPart,imagePart)]
    Rk = np.exp(c).real
    # Rk = np.exp(-ckList**2*T*sigmaBSM**2/2)*np.cos(ckList*(m-T*sigmaBSM**2/2))
    return Rk

def putOptionPriceCOS(S0,strike,T,r,q,sigmaBSM,quantile,numGrid,showDuration=False):
    tick = time.time()
    (a,b) = preprocessing.calculateToleranceInterval(S0,strike,T,r,q,sigmaBSM,quantile)
    m = preprocessing.calculateConstantTerm(S0,strike,T,r,q,a)
    Vk = calculateVkPut(strike,a,b,numGrid)
    # Rk = calculateRk(m,T,sigmaBSM,a,b,numGrid)
    Rk = calculateRk_chf(S0,strike,T,r,q,sigmaBSM,a,b,numGrid)
    putPrice = np.exp(-r * T) * np.dot(Rk,Vk)
    tack = time.time()
    if (showDuration == True):
        print("consuming time for put option using COS:", tack - tick)
    return putPrice

def callOptionPriceCOS(S0,strike,T,r,q,sigmaBSM,quantile,numGrid,showDuration=False):
    tick = time.time()
    putPrice = putOptionPriceCOS(S0,strike,T,r,q,sigmaBSM,quantile,numGrid,showDuration=False)
    callPrice = putPrice + S0*np.exp(-q*T) - strike*np.exp(-r*T)
    tack = time.time()
    if (showDuration == True):
        print("consuming time for call option using COS:", tack - tick)
    return callPrice

# S0 = 50
# strike = 55
# # a = -0.9508
# # b = 0.9466
# a = -1.0461
# b = 0.8512
# numGrid = 6
#
# print(chfBSM(np.pi/(b-a),S0,strike,0.01,0,0.1,0.25))
# print(putOptionPriceCOS(50,55,0.1,0.01,0,0.25,7,a))
