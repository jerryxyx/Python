import numpy as np
import preprocessing
import time

def calculate_Vk_put_CFS(a,b,N1,strike):
    k_array = np.linspace(0,N1-1,N1)
    xk = np.array([complex(0,2*np.pi*k/(b-a)) for k in k_array])
    chi = (1-np.exp(a*(1+xk)))/(1+xk)
    psi_0 = -a
    psi_after = (1-np.exp(xk[1:]*a))/xk[1:]
    psi = np.append(psi_0,psi_after)
    Vk_put = 2*strike/(b-a)*(psi-chi)
    Vk_put[0]/=2
    return Vk_put

def calculate_chfk_BSM_CFS(S0,strike,T,r,q,sigma,a,b,N1):
    k_array = np.linspace(0, N1 - 1, N1)
    xk = np.array([-2*k*np.pi/(b-a) for k in k_array])
    chfk = np.exp([complex(-x**2*sigma**2*T/2, x*(np.log(S0/strike)+(r-q-sigma**2/2)*T)) for x in xk])
    return chfk

def putOptionPriceCFS(S0,strike,T,r,q,sigma,quantile,numGrid,showDuration=False):
    (a,b) = preprocessing.calculateToleranceInterval(S0,strike,T,r,q,sigma,quantile)
    tick = time.time()
    chfk = calculate_chfk_BSM_CFS(S0,strike,T,r,q,sigma,a,b,numGrid)
    Vk = calculate_Vk_put_CFS(a,b,numGrid,strike)
    putOptionPrice = np.sum(np.exp(-r*T)*chfk*Vk).real
    tack = time.time()
    if(showDuration==True):
        print("consuming time for call option using CFS:", tack - tick)
    return putOptionPrice
