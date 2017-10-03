import numpy as np
from scipy.misc import factorial

def calculateTrigonometricSeries(ck,m,truncatedOrder):

    N = truncatedOrder//2
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

def calculateExponentialSeries(ck,truncatedOrder):
    N = truncatedOrder//2
    n = np.linspace(0,2*N-1,2*N)
    expSeries = np.power(-ck**2/2,n)/factorial(n)
    # print(expSeries)
    return expSeries

def calculateHybridSeries(ck,m,truncatedOrder):
    N = truncatedOrder//2
    trigSeries = calculateTrigonometricSeries(ck,m,truncatedOrder)
    expSeries = calculateExponentialSeries(ck,truncatedOrder)
    # only term[0] to term[2N-1] is correct
    hybridSeries = np.polymul(trigSeries,expSeries)[:2*N]
    return hybridSeries




# calculateTrigonometricSeries(1,1,10)
# calculateExponentialSeries(1,10)
# print(calculateHybridSeries(1,1,10))
