import BlackScholesOption
import preprocessing
import Vk_utils
import IV_expansion_utils
import COS_expansion_utils
import numpy as np
import time
from numpy.polynomial.polynomial import polyval
import series_reversion

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
print("Vk:")
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
errorUpperBound = preprocessing.calculateErrorUpperBound(S0,strike,r,q,T,sigmaBSM,N=N1,quantile=quantile)
print("priory error estimation:",errorUpperBound)
putPriceCOS = COS_expansion_utils.putOptionPriceCOS(S0,strike,T,r,q,sigmaBSM,numGrid=N1,showDuration=True)
print("put price:", putPriceCOS)
print("absolute error:", np.abs(putPriceCOS-putPriceBSM))
callPriceCOS = COS_expansion_utils.callOptionPriceCOS(S0,strike,T,r,q,sigmaBSM,numGrid=N1,showDuration=True)
print("call price:", callPriceCOS)
print("absolute error:", np.abs(callPriceCOS-callPriceBSM))
print("***************************************************************************")
print("calculate put option price using IV expansion method:")
print("numGrid:",N1,"truncationOrder:",N2)
start_time = time.time()
putIV=IV_expansion_utils.putOptionPriceIV(S0,strike,T,r,q,sigmaBSM,numGrid=N1,truncationOrder=N2,showDuration=True)
end_time = time.time()
print(end_time-start_time)
print(putIV)
print("***************************************************************************")
print("Reverse the series:")
coeffs = IV_expansion_utils.calculateCoefficientList(strike,m,a,b,numGrid=N1,truncationOrder=N2)
# print(coeffs)
inverseCoeffs = series_reversion.inverseSeries(coeffs)
# print(inverseCoeffs)
y = putIV*np.exp(r*T) - coeffs[0]
w = polyval(y,inverseCoeffs)
print("w",w)
print("T*sigmaBSM**2",sigmaBSM**2*T)
print("error:", (w-sigmaBSM**2*T)/(sigmaBSM**2*T))
