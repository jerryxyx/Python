import pandas as pd
#import math
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

def SVI(strike,a,b,sigma,rho,m):
    return a+b*(rho*(strike-m)+np.sqrt((strike-m)**2+sigma*sigma))

CIV = pd.read_csv("cImpliedVolatility.csv")
PIV = pd.read_csv("pImpliedVolatility.csv")
cbsm = CIV.CBSM
cHasIV = cbsm.notnull()
pbsm = PIV.PBSM
pHasIV = pbsm.notnull()
strike1 = np.array(CIV.cStrike[cHasIV])
volatility1 = np.array(CIV.CBSM[cHasIV])
strike2 = np.array(PIV.pStrike[pHasIV])
volatility2 = np.array(PIV.PBSM[pHasIV])

parameter1,covariance1=scipy.optimize.curve_fit(SVI,np.array(CIV.cStrike[cHasIV]),np.array(CIV.CBSM[cHasIV]))
parameter2,covariance2=scipy.optimize.curve_fit(SVI,np.array(PIV.pStrike[pHasIV]),np.array(PIV.PBSM[pHasIV]))
SVI1 = lambda strike: SVI(strike,parameter1[0],parameter1[1],parameter1[2],parameter1[3],parameter1[4])
SVI2 = lambda strike: SVI(strike,parameter2[0],parameter2[1],parameter2[2],parameter2[3],parameter2[4])

figure,axarr = plt.subplots(2,sharex=True,sharey=True)

axarr[0].plot(strike1,volatility1)
axarr[0].plot(strike1,SVI1(strike1))
axarr[0].set_title("EU Call Option SVI")
axarr[1].plot(strike2,volatility2)
axarr[1].plot(strike2,SVI2(strike2))
axarr[1].set_title("US Put Option SVI")

plt.show()