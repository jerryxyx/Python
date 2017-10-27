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
    # (a, b) = preprocessing.calculateToleranceIntervalWithoutSigma(S0, strike, T, r, q, quantile)
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
    # (a, b) = preprocessing.calculateToleranceIntervalWithoutSigma(S0, strike, T, r, q, quantile)
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
<<<<<<< HEAD
        coeffArray[l] = np.sum(coeff1*coeff2*VkArray)
=======
        coeffArray[l] = np.sum(coeff1*coeff2*VkArray).real
>>>>>>> 9e83db0e790146c4eff11148dd77e3d2e13dcca4
        # coeff2 = np.array([(complex(-ck**2,ck)/2)**l / factorial(l) for ck in ckArray])
        # coeffArray[l] = np.sum(coeff1*coeff2*VkArray)
        # coeffArray[l] = np.sum([complex(-.5*ck**2,.5*ck)**l/factorial(l)*np.exp(complex(0,-m*ck))*Vk for ck,Vk in zip(ckArray,VkArray)])

    # print("coffs",coeffArray)
    return coeffArray
def calculateImpliedVarianceCoefficientArray(S0,strike,T,r,q,sigmaBSM,N1,N2,quantile,a,b):
    # (a,b) = preprocessing.calculateToleranceInterval(S0,strike,T,r,q,sigmaBSM,quantile)

    m = (r-q)*T + np.log(S0/strike)
    ckArray = np.array([k*2*np.pi/(b-a) for k in range(N1)])
    complexCkArray = np.array([k*2.j*np.pi/(b-a) for k in range(N1)],dtype=np.complex128)
    VkArray = calculate_Vk_put_CFS(a,b,N1,strike)
    coeffArray = np.zeros(N2,dtype=np.float64)
    for l in range(N2):
        # coeff1 = np.exp(-complex(0,ckArray*m))
        coeff1 = np.exp(-complexCkArray*m)
        # print("coeff1[",l,"]",coeff1)
        coeff2 = ((complexCkArray+complexCkArray**2)*T/2)**l /factorial(l)
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
    # coeffs_volT = calculateIVCoefficientArray(S0,strike,T,r,q,sigma,N1,N2,quantile)
    coeffs_var = calculateImpliedVarianceCoefficientArray(S0,strike,T,r,q,sigma,N1,N2,quantile,a,b)
    Vk = calculate_Vk_put_CFS(a,b,N1,strike)
    print("Vk",Vk)
    print("coeffs of var",coeffs_var)
    V0 = BlackScholesOption.putOptionPriceBSM(S0,strike,T,r,q,sigma)
    V1 = np.exp(-r*T)*np.dot(chfk,Vk)
    V2 = np.exp(-r*T)*np.dot(chfk_IV,Vk)
    varl = [(sigma**2)**l for l in range(N2)]
    V3 = np.exp(-r*T)*np.dot(coeffs_var,varl)
    print("Black Scholes Value:",V0)
    print("put price using original chf in CFS method:",V1)
    print("chf represented as a w power series:",V2)
    print("exchange summation order, and let V represented as a var power series:",V3)
    inverse_coeffs = series_reversion.inverseSeries(coeffs_var)
    print("inverse coeffs:",inverse_coeffs)
    var_true = sigma**2
    y = V0*np.exp(r*T)-coeffs_var[0]
    yl = [y**l for l in range(11)]
    var_est = np.dot(inverse_coeffs, yl)
    print("compare",var_est,var_true)
    return

def testifyVarSeries(S0,strike,T,r,q,sigma,N1,N2,quantile,a,b):
    import BlackScholesOption
    import matplotlib.pyplot as plt
    coeffs_var = calculateImpliedVarianceCoefficientArray(S0, strike, T, r, q, sigma, N1, N2, quantile,a,b)
    inverse_coeffs = series_reversion.inverseSeries(coeffs_var)
    # V0 = BlackScholesOption.putOptionPriceBSM(S0, strike, T, r, q, sigma)
    V_BSM_list =[]
    V_CFSIV_list = []
    for sigma_i in np.linspace(0.1,0.8,8):
        V_i = BlackScholesOption.putOptionPriceBSM(S0, strike, T, r, q, sigma_i)
        var_i = sigma_i**2
        var_list = [var_i**l for l in range(len(coeffs_var))]
        V_i_CFSIV = np.exp(-r*T)*np.dot(coeffs_var,var_list)
        V_BSM_list.append(V_i)
        V_CFSIV_list.append(V_i_CFSIV)
    plt.plot(V_BSM_list)
    plt.plot(V_CFSIV_list)
    print("sigma_i",np.linspace(0.1,0.8,8))
    print("V_BSM_LIST",V_BSM_list)
    print("V_CFSIV_LIST",V_CFSIV_list)
    plt.show()
    return

def testify_CFS_IV(S0,strike,T,r,q,fixVol,N1,N2,quantile):
    import BlackScholesOption
    import matplotlib.pyplot as plt
    numLoop = 10
    (a,b) = preprocessing.calculateToleranceInterval(S0,strike,T,r,q,fixVol,quantile)
    coeffs_var = calculateImpliedVarianceCoefficientArray(S0, strike, T, r, q, fixVol, N1, N2, quantile,a,b)
    inverse_coeffs = series_reversion.inverseSeries(coeffs_var)
    # V0 = BlackScholesOption.putOptionPriceBSM(S0, strike, T, r, q, sigma)
    targetSigma = np.array([(i+1)*0.1 for i in range(10)])
    target_vars = np.power(targetSigma,2)
    var_estimations = np.zeros(numLoop)
    for i in range(numLoop):
        target_sigma_i = targetSigma[i]
        V_i = BlackScholesOption.putOptionPriceBSM(S0, strike, T, r, q, target_sigma_i)
        var_i = target_sigma_i**2
        Vi_List = [(V_i-coeffs_var[0])**l for l in range(len(inverse_coeffs))]
        var_estimations[i] = np.exp(-r*T)*np.dot(inverse_coeffs,Vi_List)

    print("CFS: target var",target_vars)
    print("CFS: var estimation",var_estimations)

    return
