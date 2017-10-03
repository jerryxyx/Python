import preprocessing
import Vk_utils
import Dlk_utils
import numpy as np

############################
# Hyperparameters:
S0 = 50
strike = 50
T = 0.1
r = 0.01
q = 0
sigmaBSM = 0.25
# numStrikes = 10
numGrid = 2**6
truncatedOrder = 10 #that is we want to obtain (convergent rapidly)
#############################
# Derivative parameters
(a,b) = preprocessing.generateTruncatedInterval(S0,strike,T,r,q,sigmaBSM,"BSM")
print(a,b)
m = preprocessing.calculateConstantTerm(S0,strike,T,r,q,a)
#############################

Vk = Vk_utils.calculateVkPut(strike,a,b,numGrid)
print("Vk",Vk)
ckList = [k*np.pi/(b-a) for k in range(20)]
# Truncated order should satisfies j >> (k/2)^2. j = 5*j^2 is a proper value
# Dl3 = Dlk_utils.calculateHybridSeries(5*np.pi/(b-a),m,truncatedOrder=45)
Dl5 = Dlk_utils.calculateHybridSeries(5*np.pi/(b-a),m,truncatedOrder=100)
print("k=5,Dlk",Dl5)
# Dl10 = Dlk_utils.calculateHybridSeries(10*np.pi/(b-a),m,truncatedOrder=100)# here should be 550 to see convergence
# DlkList = [Dlk_utils.calculateHybridSeries(ck,m,truncatedOrder=10) for ck in ckList]
# print("k=0..19,Dlk",DlkList)