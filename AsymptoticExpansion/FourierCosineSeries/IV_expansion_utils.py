import numpy as np
from scipy.misc import factorial
from numpy.polynomial.polynomial import polyval
from Vk_utils import calculateVkPut
from preprocessing import generateTruncatedInterval, calculateConstantTerm

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

def calculatePutOptionPriceBSM(S0,strike,T,r,q,sigmaBSM, numGrid,truncationOrder):
    (a,b) = generateTruncatedInterval(S0,strike,T,r,q,sigmaBSM,model="BSM")
    m = calculateConstantTerm(S0,strike,T,r,q,a)
    coeffs = calculateCoefficientList(strike, m, a, b, numGrid, truncationOrder)
    putPrice = np.exp(-r * T) * polyval(T * sigmaBSM ** 2, coeffs)
    return putPrice

# print(calculateTrigonometricSeries(10,1,10))
# print(calculateExponentialSeries(np.sqrt(2),10))
# print(calculateHybridSeries(1,1,10))
