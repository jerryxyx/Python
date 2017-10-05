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
# numStrikes = 10
numGrid = 6
truncationOrder = 3*numGrid**2 #that is we want to obtain (convergent rapidly)
#############################
print("***************************************************************************")
print("Hyperparameters:")
print("S0:",S0,"r",r,"q",q,"sigmaBSM",sigmaBSM)
print("***************************************************************************")
print("a and b (truncated interval):")
(a,b) = preprocessing.generateTruncatedInterval(S0,strike,T,r,q,sigmaBSM,"BSM")
print(a,b)
m = preprocessing.calculateConstantTerm(S0,strike,T,r,q,a)
Vk = Vk_utils.calculateVkPut(strike,a,b,numGrid=numGrid)
print("Vk:")
print(Vk)
print("***************************************************************************")
print("calculate put option price using COS method:")
print("numGrid:",5)
print(COS_expansion_utils.calculatePutOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,numGrid=5))
print("numGrid:",32)
time1 = time.time()
put1=COS_expansion_utils.calculatePutOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,numGrid=32)
time2 = time.time()
print(time2-time1)
call1 = put1+S0*np.exp(-q*T) -strike*np.exp(-r*T)
print(call1)
print("***************************************************************************")
print("calculate put option price using IV expansion method:")
print("numGrid:",5,"truncationOrder:",5)
start_time = time.time()
put=IV_expansion_utils.calculatePutOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,numGrid=5,truncationOrder=5)
end_time = time.time()
print(end_time-start_time)
print(put)
print("numGrid:",20,"truncationOrder:",20)
start_timeIV10 = time.time()
putIV10 = IV_expansion_utils.calculatePutOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,numGrid=10,truncationOrder=10)
end_timeIV10 = time.time()
print("time consumed for N=10:",end_timeIV10-start_timeIV10)
print(putIV10)
print("***************************************************************************")
print("Reverse the series:")
coeffs = IV_expansion_utils.calculateCoefficientList(strike,m,a,b,numGrid=10,truncationOrder=10)
# print(coeffs)
inverseCoeffs = series_reversion.inverseSeries(coeffs)
# print(inverseCoeffs)
y = putIV10*np.exp(r*T) - coeffs[0]
w = polyval(y,inverseCoeffs)
print("w",w)
print("T*sigmaBSM**2",sigmaBSM**2*T)
print("error (N=10):", (w-sigmaBSM**2*T)/(sigmaBSM**2*T))
