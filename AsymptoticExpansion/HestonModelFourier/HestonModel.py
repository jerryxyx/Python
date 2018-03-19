import numpy as np

def chf2(u,t,r,q,kappa,longTermVar,volOfVar,rho,initialAssetPrice,initialVar):
    d = np.sqrt((rho*volOfVar*u*1.j-kappa)**2 + volOfVar**2*(1.j*u+u**2))
    g2 = (kappa-rho*volOfVar*u*1.j-d)/(kappa-rho*volOfVar*u*1.j+d)
    chf2 = np.exp(1.j*u*(np.log(initialAssetPrice)+(r-q)*t))
    chf2 *= np.exp(longTermVar*kappa/volOfVar**2 * ((kappa-rho*volOfVar*u*1.j-d)*t-2*np.log((1-g2*np.exp(-d*t))/(1-g2))))
    chf2 *= np.exp(initialVar/volOfVar**2 * (kappa-rho*volOfVar*u*1.j-d)*(1-np.exp(-d*t))/(1-g2*np.exp(-d*t)))
    return chf2

def chf(u,t,r,q,kappa,longTermVar,volOfVar,rho,initialVar):
    d = np.sqrt((rho*volOfVar*u*1.j-kappa)**2 + volOfVar**2*(1.j*u+u**2))
    g2 = (kappa-rho*volOfVar*u*1.j-d)/(kappa-rho*volOfVar*u*1.j+d)
    chf2 = np.exp(1.j*u*(+(r-q)*t))
    chf2 *= np.exp(longTermVar*kappa/volOfVar**2 * ((kappa-rho*volOfVar*u*1.j-d)*t-2*np.log((1-g2*np.exp(-d*t))/(1-g2))))
    chf2 *= np.exp(initialVar/volOfVar**2 * (kappa-rho*volOfVar*u*1.j-d)*(1-np.exp(-d*t))/(1-g2*np.exp(-d*t)))
    return chf2