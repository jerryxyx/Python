import numpy as np

def calculateChi(a,b,c,d,numGrid):
#   k can be an integer: k1 = 0
#   k can be an array: k2 = np.float32([i for i in range(N)])
#   k can be a matrix: k3 = np.tile(k2,(numStrikes,1)).T
#   in this case, we restrict k to be an array
    k = np.float32([i for i in range(numGrid)])
    pi = np.pi
    var1 = 1/(1+np.power(k*pi/(b-a),2))
    var2 = np.cos(k*pi*(d-a)/(b-a)) * np.exp(d)
    var3 = np.cos(k*pi*(c-a)/(b-a)) * np.exp(c)
    var4 = k*pi/(b-a)*np.sin(k*pi*(d-a)/(b-a)) * np.exp(d)
    var5 = k*pi/(b-a)*np.sin(k*pi*(c-a)/(b-a)) * np.exp(c)
    chi = var1*(var2-var3+var4-var5)
    # print("chi",chi)
    return chi

def calculatePsi(a,b,c,d,numGrid):
    pi = np.pi
    k = np.float32([i for i in range(numGrid)])[1:]
    psi_0 = d - c
    var1 = (b-a)/(k*pi)
    var2 = np.sin(k*pi*(d-a)/(b-a))
    var3 = np.sin(k*pi*(c-a)/(b-a))
    psi_after = var1*(var2-var3)
    psi = np.append(psi_0,psi_after)
    # print("psi",psi)
    return psi

# Todo: can be accelerated by using function specialize the case c=a, d=0
def calculateVkPut(strike,a,b,numGrid):
    psi = calculatePsi(a,b,a,0,numGrid)
    chi = calculateChi(a,b,a,0,numGrid)
    VkPut = 2*strike/(b-a) * (psi-chi)
    VkPut[0] /= 2
    return VkPut

def calculateVkCall(strike,a,b,numGrid):
    psi = calculatePsi(a, b, 0, b, numGrid)
    chi = calculateChi(a, b, 0, b, numGrid)
    VkCall = 2 * strike / (b - a) * (chi - psi)
    VkCall[0] /= 2
    return VkCall

# S0 = 50
# strike = 55
# # a = -0.9508
# # b = 0.9466
# a = -1.0461
# b = 0.8512
# c = a
# d = 0
# # numStrikes = 10;
# numGrid = 2**8;
# chi = calculateChi(a,b,c,d,numGrid)
# psi = calculatePsi(a,b,c,d,numGrid)
# print(chi)
# print(psi)
# print(psi-chi)
# # print(calculateVkPut(strike,a,b,numGrid))
