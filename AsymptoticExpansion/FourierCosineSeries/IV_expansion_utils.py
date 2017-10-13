import numpy as np
from scipy.misc import factorial
from numpy.polynomial.polynomial import polyval
from Vk_utils import calculateVkPut
from series_reversion import inverseSeries
# from preprocessing import generateTruncatedInterval, calculateConstantTerm
import preprocessing
import BlackScholesOption
import time

def calculateTrigonometricSeries(ck,m,truncationOrder):

    N = truncationOrder//2
    n = np.linspace(0,N-1,N)
    cosSeries = np.cos(ck*m)* (-1)**n * np.power((ck/2),2*n) / factorial(2*n)
    sinSeries = np.sin(ck*m)* (-1)**n * np.power((ck/2),2*n+1) / factorial(2*n+1)
    # print(cosSeries)
    # print(sinSeries)

    # djk from 0 to 2N-1
    trigSeries = np.zeros(2*N)
    trigSeries[::2] = cosSeries
    trigSeries[1::2] = sinSeries
    # print(trigSeries)
    return trigSeries

def calculateExponentialSeries(ck,truncationOrder):
    N = truncationOrder//2
    n = np.linspace(0,2*N-1,2*N)
    expSeries = np.power(-ck**2/2,n)/factorial(n)
    # print(expSeries)
    return expSeries

def calculateHybridSeries(ck,m,truncationOrder):
    N = truncationOrder//2
    trigSeries = calculateTrigonometricSeries(ck,m,truncationOrder)
    expSeries = calculateExponentialSeries(ck,truncationOrder)
    # only term[0] to term[2N-1] is correct
    hybridSeries = np.polymul(trigSeries,expSeries)[:2*N]
    return hybridSeries

def calculateHybridSeries2d(ckList,m,truncationOrder):
    Dkl = np.array([calculateHybridSeries(ck, m, truncationOrder=truncationOrder) for ck in ckList])
    return Dkl

def calculateCoefficientList(strike,m,a,b,numGrid,truncationOrder=None):
    if(truncationOrder==None):
        truncationOrder=5*numGrid**2
    ckList = [k*np.pi/(b-a) for k in range(numGrid)]
    Vk = calculateVkPut(strike,a,b,numGrid)
    Dkl = calculateHybridSeries2d(ckList,m,truncationOrder)
    coefficientList = np.dot(Vk,Dkl)
    return coefficientList

def putOptionPriceIV(S0,strike,T,r,q,sigmaBSM, quantile, numGrid,truncationOrder,showDuration=False):
    tick = time.time()
    (a,b) = preprocessing.calculateToleranceInterval(S0,strike,T,r,q,sigmaBSM,quantile)
    m = preprocessing.calculateConstantTerm(S0,strike,T,r,q,a)
    coeffs = calculateCoefficientList(strike, m, a, b, numGrid, truncationOrder)
    w = T*sigmaBSM**2
    wList = np.array([w**l for l in range(len(coeffs))])
    # putPrice = np.exp(-r * T) * polyval(T * sigmaBSM ** 2, coeffs)
    putPrice = np.exp(-r*T)*np.dot(wList,coeffs)
    tack = time.time()
    if (showDuration == True):
        print("consuming time for call option using IV:", tack - tick)
    return putPrice


def calculateImpliedVolatilityByPutOptionPrice(S0, strike, T, r, q, price, quantile, N1,N2,fixPoint=0.20,showDuration=False):
    (a,b) = preprocessing.calculateToleranceInterval(S0,strike,T,r,q,fixPoint,quantile)
    m = preprocessing.calculateConstantTerm(S0, strike, T, r, q, a)
    tick = time.time()


    # coeffs = calculateCoefficientList(strike, m, a, b, numGrid=N1, truncationOrder=N2)
    # inverseCoeffs = inverseSeries(coeffs)
    # y = price * np.exp(r * T) - coeffs[0]
    # omega = polyval(y, inverseCoeffs)
    # # print(omega,T)
    # volEstimation = np.sqrt(omega/T)
    coeffs = calculateCoefficientList(strike, m, a, b, numGrid=N1, truncationOrder=N2)
    # print("coeff for COS:", coeffs)
    # inverseCoeffs_old = inverseSeries_old(coeffs)
    inverseCoeffs = inverseSeries(coeffs)
    print("new", inverseCoeffs)
    # print("old", inverseCoeffs_old)

    # print(inverseCoeffs)
    y = price * np.exp(r * T) - coeffs[0]
    w = polyval(y, inverseCoeffs)
    print("w", w)
    # print("T*sigmaBSM**2", sigmaBSM ** 2 * T)
    # print("absolute error:", (w - sigmaBSM ** 2 * T))
    volEstimation = np.sqrt(w/T)

    tack = time.time()
    if(showDuration==True):
        print("consuming time for calculating implied volatility:",tack-tick)
    return volEstimation

# print(calculateTrigonometricSeries(10,1,10))
# print(calculateExponentialSeries(np.sqrt(2),10))
# print(calculateHybridSeries(1,1,10))
