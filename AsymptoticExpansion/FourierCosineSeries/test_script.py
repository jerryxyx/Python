import preprocessing
import Vk_utils
import IV_expansion_utils
import COS_expansion_utils
import numpy as np
from numpy.polynomial.polynomial import polyval

############################
# Hyperparameters:
S0 = 50
strike = 55
T = 0.1
r = 0.01
q = 0
sigmaBSM = 0.25
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
print("numGrid:",20)
print(COS_expansion_utils.calculatePutOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,numGrid=20))
print("***************************************************************************")
print("calculate put option price using IV expansion method:")
print("numGrid:",5,"truncationOrder:",5)
print(IV_expansion_utils.calculatePutOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,numGrid=5,truncationOrder=5))
print("numGrid:",20,"truncationOrder:",20)
print(IV_expansion_utils.calculatePutOptionPriceBSM(S0,strike,T,r,q,sigmaBSM,numGrid=20,truncationOrder=20))
