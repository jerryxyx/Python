import IV_expansion_utils
import preprocessing


IV_expansion_utils.testify_IV(S0=100,strike=100,T=1,r=0.05,q=0,a=-10,b=10,N1=16,N2=32,quantile=10,fixVol=0.3)

print("Conclusion: Only when ln(S0/K)+(r-q-0.5*sigma**2)*T is sufficiently small can we choose a symmetric interval")