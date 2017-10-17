import numpy as np
import HestonModel
import heston
import BlackScholesOption

def toleranceInterval(initialAssetPrice,strike,quantile,initialVar, T, r, q, kappa, longTermVar, volOfVar, rho):
    c1 = (r-q)*T + (1-np.exp(-kappa*T))*(longTermVar-initialVar)/(2*kappa) - 0.5*longTermVar*T
    c2 = 1/(8*kappa**3) * (longTermVar*T*kappa*np.exp(-kappa*T)*(initialVar-longTermVar)*(8*kappa*rho-4*volOfVar)
                           + kappa*rho*volOfVar*(1-np.exp(-kappa*T))*(16*longTermVar-8*initialVar)
                           + 2*longTermVar*kappa*T*(-4*rho*volOfVar + volOfVar**2 +4*kappa**2)
                           + volOfVar**2*((longTermVar-2*initialVar)*np.exp(-2*kappa*T) + longTermVar*(6*np.exp(-kappa*T)-7) + 2*initialVar)
                           + 8*kappa**2*(initialVar-longTermVar)*(1-np.exp(-kappa*T)))
    a = c1+np.log(initialAssetPrice)-quantile*np.sqrt(abs(c2))
    b = c1+np.log(initialAssetPrice)+quantile*np.sqrt(abs(c2))
    print("a,b",a,b)
    if(np.log(strike)>b or np.log(strike)<a):
        print("Caution: log(K) is not in (a,b)",np.log(strike))
    return (a,b)


def chi(nPower, N1, a, b, c, d):
    k_array = np.linspace(0, N1 - 1, N1)
    xk = np.array([2.j * np.pi * k / (b - a) for k in k_array])
    chi = 1 / (nPower - xk) * (np.exp((nPower - xk) * d) - np.exp((nPower - xk) * c))
    return chi


def psi(N1, a, b, c, d):
    # print("ab in psi",a,b)
    k_array = np.linspace(0, N1 - 1, N1)
    xk = np.array([2.j * np.pi * k / (b - a) for k in k_array])
    psi = np.zeros(N1,dtype=np.complex128)
    psi[0] = d-c
    psi[1:] = 1 / (-xk[1:]) * (np.exp(-xk[1:] * d) - np.exp(-xk[1:] * c))
    # print("psi",psi)
    return psi


def VkPut(S0, strike, nPower, N1, a, b):
    Vk = 2 / (b - a) * (strike * psi(N1, a, b, a, np.log(strike)/nPower-np.log(S0)) - S0**nPower*chi(nPower, N1, a, b, a, np.log(strike)/nPower-np.log(S0)))
    Vk[0]/=2
    return Vk

def VkCall(S0, strike, nPower, N1, a, b):
    print( "a,0,b",a,np.log(strike)/nPower-np.log(S0),b)
    Vk = 2 / (b - a) * ( S0**nPower*chi(nPower, N1, a, b, np.log(strike)/nPower-np.log(S0),b)-strike * psi(N1, a, b, np.log(strike)/nPower-np.log(S0),b))
    Vk[0]/=2
    return Vk


def HestonChfSeries(N1,a,b, T, r, q, kappa, longTermVar, volOfVar, rho, initialVar):

    V0 = initialVar
    k_array = np.linspace(0, N1 - 1, N1)
    xk = np.array([2.j * np.pi * k / (b - a) for k in k_array])
    param = heston.HestonParam(lm=kappa, mu=longTermVar, eta=volOfVar, rho=rho, sigma=V0)
    hestonInstance = heston.Heston(param=param,riskfree=r-q, maturity=T)
    chfSeries = hestonInstance.charfun(xk)
    # EQUIVALENT
    chfSeries2 = HestonModel.chf(xk, T, r, q, kappa, longTermVar, volOfVar, rho, initialVar)
    # print("chfSeries by hston2", chfSeries2)
    # print("chf true",chfSeries)
    return chfSeries2

def BSChfSeries(N1,a,b,T,r,q,sigma):
    k_array = np.linspace(0, N1 - 1, N1)
    xk = np.array([2 * np.pi * k / (b - a) for k in k_array],dtype=np.complex128)
    # print("xk",xk)
    BSChf = BlackScholesOption.chf(xk,r,q,T,sigma)
    return BSChf

def putOpitonPricer(S0,strike,V0, nPower,N1,a,b, T, r, q, kappa, longTermVar, volOfVar, rho):
    chfSeries = HestonChfSeries(N1,a,b, T, r, q, kappa, longTermVar, volOfVar, rho, V0)
    # param = heston.HestonParam(lm=kappa, mu=r-q, eta=volOfVar, rho=rho, sigma=V0)
    # hestonInstance = heston.Heston(param=param, riskfree=r, maturity=T)
    # hestonInstance.charfun()
    VkPutSeries = VkPut(S0,strike, nPower, N1, a, b)
    # print("chfSeries", chfSeries)
    # print("VkPut", VkPutSeries)
    putPrice = np.exp(-r*T) * np.dot(chfSeries,VkPutSeries)
    return putPrice

def callOptionPricer(S0,strike,V0, nPower,N1,a,b, T, r, q, kappa, longTermVar, volOfVar, rho, usingPutCallParity=False):
    if(usingPutCallParity == True):
        putPrice = putOpitonPricer(S0, strike, V0, nPower, N1, a, b, T, r, q, kappa, longTermVar, volOfVar, rho)
        callPrice = putPrice + np.exp(-r*T)*(S0*np.exp(r-q)*T)**nPower - np.exp(-r*T)*strike
    else:
        chfSeries = HestonChfSeries(N1,a,b, T, r, q, kappa, longTermVar, volOfVar, rho, V0)
        VkCallSeries = VkCall(S0,strike, nPower, N1, a, b)
        callPrice = np.exp(-r * T) * np.dot(chfSeries, VkCallSeries)
    return callPrice

def putOpitonPricerBSM(S0,strike,nPower,N1,a,b, T, r, q, sigma):
    # chfSeries = HestonChfSeries(N1,a,b, T, r, q, kappa, longTermVar, volOfVar, rho, S0, initialVar)
    chfSeries = BSChfSeries(N1,a,b,T,r,q,sigma)
    VkPutSeries = VkPut(S0,strike, nPower, N1, a, b)
    # print("chfSeries", chfSeries)
    # print("VkPut", VkPutSeries)
    putPrice = np.exp(-r*T) * np.dot(chfSeries,VkPutSeries)
    return putPrice

def callOpitonPricerBSM(S0,strike,nPower,N1,a,b, T, r, q, sigma,usingPutCallParity=False):
    if(usingPutCallParity==True):
        putPrice = putOpitonPricerBSM(S0,strike,nPower,N1,a,b, T, r, q, sigma)
        callPrice = putPrice + np.exp(-r*T)*(S0*np.exp(r-q)*T)**nPower - np.exp(-r*T)*strike
        # callPrice = putPrice + S0  ** nPower - np.exp(-r * T) * strike
    else:
        chfSeries = BSChfSeries(N1, a, b, T, r, q, sigma)
        VkCallSeries = VkCall(S0, strike, nPower, N1, a, b)
        # print("chfSeries", chfSeries)
        # print("VkPut", VkPutSeries)
        callPrice = np.exp(-r * T) * np.dot(chfSeries, VkCallSeries)
    return callPrice