import BlackScholesOption
import preprocessing
import Vk_utils
import IV_expansion_utils
import COS_expansion_utils
import numpy as np
import time
from numpy.polynomial.polynomial import polyval
import series_reversion
import CFS_expansion_utils

############################
# Hyperparameters:
S0 = 100
strike = 100
T = 0.1
r = 0.05
q = 0
sigmaBSM = 0.2
quantile = 10
N1 = 16
N2 = N1 #that is we want to obtain (convergent rapidly)
# numStrikes = 10
#############################
# Derivative parameters:
(a,b) = preprocessing.calculateToleranceInterval(S0,strike,T,r,q,sigmaBSM,quantile)
#############################
print("***************************************************************************")
print("Hyperparameters:")
print("S0:",S0,"r",r,"q",q,"sigmaBSM",sigmaBSM,"quantile",quantile)
print("N1:",N1)
print("N2", N2)
print("***************************************************************************")
print("Derivative parameters:")
print("a and b (truncated interval):",(a,b))
print("***************************************************************************")
print("Other parameters:")
m = preprocessing.calculateConstantTerm(S0,strike,T,r,q,a)
print("m:", m)
Vk = Vk_utils.calculateVkPut(strike,a,b,numGrid=N1)
print("Vk put:")
print(Vk)
print("***************************************************************************")
print("calculate put option price using Black-Scholes formula:")
tickPutBSM = time.time()
putPriceBSM = BlackScholesOption.putOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,showDuration=True)
tackPutBSM = time.time()
print("put price:", putPriceBSM)
print("calculate call option price using Black-Scholes formula:")
callPriceBSM = BlackScholesOption.callOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,showDuration=True)
print("call price:", callPriceBSM)
print("***************************************************************************")
print("calculate put option price using COS method:")
print("N1:",N1)
errorUpperBound = preprocessing.calculateErrorUpperBound(S0,strike,r,q,T,sigmaBSM,N=N1,quantile=quantile,showDetails=True)
print("priory error estimation:",errorUpperBound)
putPriceCOS = COS_expansion_utils.putOptionPriceCOS(S0,strike,T,r,q,sigmaBSM,quantile=quantile,numGrid=N1,showDuration=True)
print("put price:", putPriceCOS)
print("absolute error:", np.abs(putPriceCOS-putPriceBSM))
callPriceCOS = COS_expansion_utils.callOptionPriceCOS(S0,strike,T,r,q,sigmaBSM,quantile,numGrid=N1,showDuration=True)
print("call price:", callPriceCOS)
print("absolute error:", np.abs(callPriceCOS-callPriceBSM))
print("***************************************************************************")
print("calculate put option price using CFS method:")
print("N1:",N1)
putPriceCFS = CFS_expansion_utils.putOptionPriceCFS(S0,strike,T,r,q,sigmaBSM,quantile,numGrid=N1,showDuration=True)
print("put price:", putPriceCFS)
print("absolute error:", np.abs(putPriceCFS-putPriceBSM))
print("***************************************************************************")
print("calculate put option price using IV expansion method:")
print("N1:",N1,"N2:",N2)
putPriceIV=IV_expansion_utils.putOptionPriceIV(S0,strike,T,r,q,sigmaBSM,quantile,numGrid=N1,truncationOrder=N2,showDuration=True)
print("put price:", putPriceIV)
print("absolute error:", np.abs(putPriceIV-callPriceBSM))
print("***************************************************************************")
print("same volatility:")
coeffs = IV_expansion_utils.calculateCoefficientList(strike,m,a,b,numGrid=N1,truncationOrder=N2)
# print(coeffs)
inverseCoeffs = series_reversion.inverseSeries(coeffs)
# print(inverseCoeffs)
y = putPriceIV*np.exp(r*T) - coeffs[0]
w = polyval(y,inverseCoeffs)
print("w",w)
print("T*sigmaBSM**2",sigmaBSM**2*T)
print("absolute error:", (w-sigmaBSM**2*T)/(sigmaBSM**2*T))
print("***************************************************************************")
print("different volatility:")
targetVol = 0.3
print("fixed volatility for expansion:",sigmaBSM)
print("target(true) implied volatility:", targetVol)
putPriceBSM_2 = BlackScholesOption.putOptionPriceBSM(S0,strike,T,r,q,targetVol)
targetVolEstimation = IV_expansion_utils.calculateImpliedVolatilityByPutOptionPrice(S0,
                strike, T, r, q, putPriceBSM_2, quantile, N1,N2,fixPoint=0.20,showDuration=True)
print("target implied volatility estimation:", targetVolEstimation)
print("absolute error:", np.abs(targetVolEstimation-targetVol))

