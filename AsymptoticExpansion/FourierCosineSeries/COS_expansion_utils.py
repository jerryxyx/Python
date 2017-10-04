from Vk_utils import calculateVkPut
import numpy as np
from preprocessing import generateTruncatedInterval, calculateConstantTerm

def chfBSM(x,S0,strike,r,q,T,sigmaBSM):
    chfBSM = np.exp( complex( -0.5 * x**2 * sigmaBSM**2 *T, x*(np.log(S0/strike)+(r-q)*T)) )
    return chfBSM

def calculateRk(m,T,sigmaBSM,a,b,numGrid):
    ckList = np.array([k*np.pi/(b-a) for k in range(numGrid)])
    Rk = np.exp(-ckList**2*T*sigmaBSM**2/2)*np.cos(ckList*(m-T*sigmaBSM**2/2))
    return Rk

def calculatePutOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,numGrid):
    (a,b) = generateTruncatedInterval(S0,strike,T,r,q,sigmaBSM,model="BSM")
    m = calculateConstantTerm(S0,strike,T,r,q,a)
    Vk = calculateVkPut(strike,a,b,numGrid)
    Rk = calculateRk(m,T,sigmaBSM,a,b,numGrid)
    putPrice = np.exp(-r * T) * np.dot(Rk,Vk)
    return putPrice

# S0 = 50
# strike = 55
# # a = -0.9508
# # b = 0.9466
# a = -1.0461
# b = 0.8512
# numGrid = 6
#
# print(chfBSM(np.pi/(b-a),S0,strike,0.01,0,0.1,0.25))
# print(calculatePutOptionPriceBSM(50,55,0.1,0.01,0,0.25,7,a))
