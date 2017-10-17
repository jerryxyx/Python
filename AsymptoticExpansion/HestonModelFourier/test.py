import CFSMethod
import BSMMethod
import heston
import HestonModel
import numpy as np

S0=100
K=150
V0=0.010201
sigma = 0.3
kappa=6.21
theta=0.019
volOfVar=0.61
rho = -0.7
r = 0.0319
q = 0
# mu = r-q
T = 1
N1 = 16
param = heston.HestonParam(lm=kappa, mu=theta, eta=volOfVar, rho=rho, sigma=V0)
hestonInstance = heston.Heston(param=param, riskfree=r, maturity=T)
a,b = hestonInstance.cos_restriction(10)
# print(hestonInstance.charfun(3))
# print(HestonModel.chf(3,T,r,q,kappa,theta,volOfVar,rho,S0,V0))

a1,b1 = CFSMethod.toleranceInterval(initialAssetPrice=S0,strike=K,quantile=10,initialVar=V0, T=T, r=r,
                                  q=q, kappa=kappa, longTermVar=theta, volOfVar=volOfVar, rho=rho)
c1 = (r-q-1)*T
a2,b2 = c1-10*np.sqrt(sigma**2*T),c1+10*np.sqrt(sigma**2*T)

print("Heston a,b",a,b)
print("BlackScholes a2,b2",a2,b2)

print("*********************************************************")
putPriceCFS_Heston = CFSMethod.putOpitonPricer(S0=S0,strike=K,V0=V0, nPower=1,N1=N1,a=a ,b=b, T=T,
                                     r=r, q=q, kappa=kappa, longTermVar=theta, volOfVar=volOfVar, rho=rho)
callPriceCFS_Heston = CFSMethod.callOptionPricer(S0=S0,strike=K,V0=V0, nPower=1,N1=N1,a=a ,b=b, T=T,
                                     r=r, q=q, kappa=kappa, longTermVar=theta, volOfVar=volOfVar, rho=rho)
putPriceBSM = BSMMethod.putOptionPrice(S0=S0,strike=K,T=T,r=r,q=q,sigmaBSM=sigma,showDuration=False)
callPriceBSM = BSMMethod.callOptionPrice(S0=S0,strike=K,T=T,r=r,q=q,sigmaBSM=sigma,showDuration=False)
putPriceCFS_GBM = CFSMethod.putOpitonPricerBSM(S0=S0,strike=K,nPower=1,N1=N1,a=a2,b=b2, T=T, r=r, q=q, sigma=sigma)
callPriceCFS_GBM = CFSMethod.callOpitonPricerBSM(S0=S0,strike=K,nPower=1,N1=N1,a=a2,b=b2, T=T, r=r, q=q, sigma=sigma)
print("power==1")
print("put price using CFS Heston",putPriceCFS_Heston)
print("put price using CFS GBM",putPriceCFS_GBM)
print("put price using BSM",putPriceBSM)
print("call price using CFS Heston",callPriceCFS_Heston)
print("call price using CFS GBM",callPriceCFS_GBM)
print("call price using BSM",callPriceBSM)
print("*********************************************************")
print("power==2")
callPriceCFS_Heston1 = CFSMethod.callOptionPricer(S0=S0,strike=K,V0=V0, nPower=2,N1=N1,a=a ,b=b, T=T,
                                     r=r, q=q, kappa=kappa, longTermVar=theta, volOfVar=volOfVar, rho=rho)
callPriceCFS_Heston2 = CFSMethod.callOptionPricer(S0=S0,strike=K,V0=V0, nPower=2,N1=N1,a=a ,b=b, T=T,
                                     r=r, q=q, kappa=kappa, longTermVar=theta, volOfVar=volOfVar, rho=rho,usingPutCallParity=True)
callPriceCFS_GBM1 = CFSMethod.callOpitonPricerBSM(S0=S0,strike=K,nPower=2,N1=N1,a=a2,b=b2, T=T, r=r, q=q, sigma=sigma,usingPutCallParity=False)
callPriceCFS_GBM2 = CFSMethod.callOpitonPricerBSM(S0=S0,strike=K,nPower=2,N1=N1,a=a2,b=b2, T=T, r=r, q=q, sigma=sigma,usingPutCallParity=True)

print("call price using CFS Heston",callPriceCFS_Heston1)
print("call price using CFS Heston(put call parity)",callPriceCFS_Heston2)
print("call price using CFS GBM",callPriceCFS_GBM1)
print("call price using CFS GBM(put call parity)",callPriceCFS_GBM2)