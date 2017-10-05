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
numGrid = 2**6  # 64should be a proper number
truncatedOrder = 10 # that is we want to obtain (convergent rapidly)
#############################
# Derivative parameters
(a,b) = preprocessing.generateTruncatedInterval(S0,strike,T,r,q,sigmaBSM,"BSM")
print(a,b)
m = preprocessing.calculateConstantTerm(S0,strike,T,r,q,a)
#############################

Vk = Vk_utils.calculateVkPut(strike,a,b,numGrid=5)
print(np.shape(Vk))
print("Vk",Vk)
ckList = [k*np.pi/(b-a) for k in range(5)]
# Truncated order should satisfies j >> (k/2)^2. j = 5*j^2 is a proper value
# Dl3 = Dlk_utils.calculateHybridSeries(5*np.pi/(b-a),m,truncatedOrder=45)
# Dl5 = Dlk_utils.calculateHybridSeries(5*np.pi/(b-a),m,truncatedOrder=100)
# print("k=5,Dlk",Dl5)
# Dl10 = Dlk_utils.calculateHybridSeries(10*np.pi/(b-a),m,truncatedOrder=100)# here should be 550 to see convergence
DlkList = [Dlk_utils.calculateHybridSeries(ck,m,truncatedOrder=64) for ck in ckList] # truncationOrder should be 64^2
print("k=0..5,Dlk",DlkList)
print(np.shape(DlkList))
