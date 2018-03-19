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
strike = 80
T = 0.1
r = 0.05
q = 0
sigmaBSM = 0.2
<<<<<<< HEAD
quantile = 10
N1 = 32
N2 = 8 #that is we want to obtain (convergent rapidly)
=======

N1 = 16
# todo:quantile setting for convergency
quantile = 10
# quantile = (np.pi*(N1-1))**2/2+1
>>>>>>> 9e83db0e790146c4eff11148dd77e3d2e13dcca4
# numStrikes = 10
#############################
# Derivative parameters:
(a,b) = preprocessing.calculateToleranceInterval(S0,strike,T,r,q,sigmaBSM,quantile)

# (a,b) = preprocessing.calculateToleranceIntervalWithoutSigma(S0,strike,T,r,q,quantile)
# N2 = preprocessing.calculateNumGrid2(N1,T,sigmaBSM,a,b)# that is we want to obtain (convergent rapidly)
N2=32
# N2 = 32  #  >12 since we use the reversion =10

#############################
print("***************************************************************************")
print("Hyperparameters:")
print("S0:",S0,"strike",strike,"T",T,"r",r,"q",q,"sigmaBSM",sigmaBSM,"quantile",quantile)
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
putPriceBSM = float(putPriceBSM)
callPriceBSM = float(callPriceBSM)
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
coeffsCFS = CFS_expansion_utils.calculateIVCoefficientArray(S0,strike,T,r,q,sigmaBSM,N1,N2,quantile)
print("coeff for CFS:",coeffsCFS)
inverseCoeffsCFS = series_reversion.inverseSeries(coeffsCFS)
print("inverse coeffs for CFS:",inverseCoeffsCFS)
targetVolCFS = 0.25
putPriceBSM_2 = float(BlackScholesOption.putOptionPriceBSM(S0,strike,T,r,q,targetVolCFS))
yCFS = putPriceBSM_2*np.exp(r*T) - coeffsCFS[0]
wCFS = polyval(yCFS,inverseCoeffsCFS)
print("w",wCFS)
print("absolute error for implied volatility:",np.abs(wCFS-sigmaBSM**2*T))
print("***************************************************************************")
print("calculate put option price using IV expansion method:")
print("N1:",N1,"N2:",N2)
putPriceIV=IV_expansion_utils.putOptionPriceIV(S0,strike,T,r,q,sigmaBSM,quantile,numGrid=N1,truncationOrder=N2,showDuration=True)
print("put price:", putPriceIV)
print("absolute error:", np.abs(putPriceIV-putPriceBSM))
print("***************************************************************************")
print("same volatility:")
coeffs = IV_expansion_utils.calculateCoefficientList(strike,m,a,b,numGrid=N1,truncationOrder=N2)
print("coeff for COS:",coeffs)
# inverseCoeffs_old = series_reversion.inverseSeries_old(coeffs)
inverseCoeffs = series_reversion.inverseSeries(coeffs)
print("inverse coeff for COS",inverseCoeffs)
# print("old",inverseCoeffs_old)

# print(inverseCoeffs)
y = putPriceIV*np.exp(r*T) - coeffs[0]
w = polyval(y,inverseCoeffs)
print("w",w)
print("T*sigmaBSM**2",sigmaBSM**2*T)
print("absolute error:", (w-sigmaBSM**2*T))
print("***************************************************************************")
print("different volatility:")
targetVol = 0.4
print("fixed volatility for expansion:",sigmaBSM)
print("target(true) implied volatility:", targetVol)
putPriceBSM_2 = float(BlackScholesOption.putOptionPriceBSM(S0,strike,T,r,q,targetVol))
targetVolEstimation = IV_expansion_utils.calculateImpliedVolatilityByPutOptionPrice(S0,
                strike, T, r, q, putPriceBSM_2, quantile, N1,N2,fixPoint=0.20,showDuration=True)
print("target implied volatility estimation:", targetVolEstimation)
print("absolute error:", np.abs(targetVolEstimation-targetVol))
print("***************************************************************************")
# print("Testify chf:")
# chfk=CFS_expansion_utils.calculate_chfk_BSM_CFS(S0,strike,T,r,q,sigmaBSM,a,b,N1)
#
# chfkIV_CFS = CFS_expansion_utils.calculateChfkIV_CFS(S0,strike,T,r,q,sigmaBSM,a,b,N1,N2)
# print("original chfk:",chfk)
# print("IV CFS chfk:",chfkIV_CFS)
# for strike_i in np.linspace(80,120,3):
#     print("strike=",strike_i)
#     CFS_expansion_utils.testifyExchangeSumOrder(S0,strike_i,T,r,q,sigmaBSM,a,b,N1,N2,quantile)
# print([series_reversion.testifyExponentSeries(1./(i+1)) for i in range(20)])
# IV_expansion_utils.testify_IV(S0,strike,T,r,q,16,32,10,0.3)
