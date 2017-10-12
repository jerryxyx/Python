import numpy as np
import preprocessing
import time
from math import factorial,log
import series_reversion
import decimal

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

def calculateIVCoefficientArray(S0,strike,T,r,q,sigmaBSM,N1,N2,quantile):
    (a,b) = preprocessing.calculateToleranceInterval(S0,strike,T,r,q,sigmaBSM,quantile)
    m = (r-q)*T + np.log(S0/strike)
    ckArray = np.array([k*2*np.pi/(b-a) for k in range(N1)])
    complexCkArray = np.array([k*2.j*np.pi/(b-a) for k in range(N1)],dtype=np.complex128)
    VkArray = calculate_Vk_put_CFS(a,b,N1,strike)
    coeffArray = np.zeros(N2,dtype=np.float64)
    for l in range(N2):
        # coeff1 = np.exp(-complex(0,ckArray*m))
        coeff1 = np.exp(-complexCkArray*m)
        # print("coeff1[",l,"]",coeff1)
        coeff2 = ((complexCkArray+complexCkArray**2)/2)**l /factorial(l)
        # print("coeff2[", l, "]", coeff2)
        coeffArray[l] = np.sum(coeff1*coeff2*VkArray).real
        # coeff2 = np.array([(complex(-ck**2,ck)/2)**l / factorial(l) for ck in ckArray])
        # coeffArray[l] = np.sum(coeff1*coeff2*VkArray)
        # coeffArray[l] = np.sum([complex(-.5*ck**2,.5*ck)**l/factorial(l)*np.exp(complex(0,-m*ck))*Vk for ck,Vk in zip(ckArray,VkArray)])

    # print("coffs",coeffArray)
    return coeffArray

def calculateChfkIV_CFS(S0,strike,T,r,q,sigma,a,b,N1,N2):
    from math import factorial
    chfkIV_CFS = np.zeros(N1,dtype=np.complex64)
    m = (r - q) * T + np.log(S0 / strike)
    for k in range(N1):
        ck=2*np.pi*k/(b-a)
        w = sigma**2*T
        exponentialExpansion = np.sum([(complex(-ck,1)*ck/2)**l/factorial(l) * w**l for l in range(N2)])
        chfkIV_CFS[k] = np.exp(-1.j*ck*m)*exponentialExpansion

    return chfkIV_CFS

def testifyExchangeSumOrder(S0,strike,T,r,q,sigma,a,b,N1,N2,quantile):
    import BlackScholesOption
    chfk_IV = calculateChfkIV_CFS(S0,strike,T,r,q,sigma,a,b,N1,N2)
    chfk = calculate_chfk_BSM_CFS(S0,strike,T,r,q,sigma,a,b,N1)
    coeffs = calculateIVCoefficientArray(S0,strike,T,r,q,sigma,N1,N2,quantile)
    Vk = calculate_Vk_put_CFS(a,b,N1,strike)
    print("Vk",Vk)
    V0 = BlackScholesOption.putOptionPriceBSM(S0,strike,T,r,q,sigma)
    V1 = np.exp(-r*T)*np.dot(chfk,Vk)
    V2 = np.exp(-r*T)*np.dot(chfk_IV,Vk)
    wl = [(sigma**2*T)**l for l in range(N2)]
    V3 = np.exp(-r*T)*np.dot(coeffs,wl)
    print("Black Scholes Value:",V0)
    print("CFS value:",V1)
    print("CFS+IV value:",V2)
    print("CFS+IV+changeOrder:",V3)
    inverse_coeffs = series_reversion.inverseSeries(coeffs)
    w = sigma**2*T
    y = V0*np.exp(r*T)-coeffs[0]
    yl = [y**l for l in range(11)]
    print("compare",np.dot(inverse_coeffs,yl),w)


    return