import IV_expansion_utils
import CFS_expansion_utils
import preprocessing

def sensitivityAnalysis(S0,T,strike,r,q,fixVol,N1,quantile):
    a,b = preprocessing.calculateToleranceInterval(S0,strike,T,r,q,fixVol,quantile)
    N2=preprocessing.calculateNumGrid2(N1,T,fixVol,a,b)
    print("Parameters setting:")
    print("S0",S0,"strike",strike,"T",T,"r",r,"q",q,"fixvol",fixVol,"N1",N1,"N2",N2,"a",a,"b",b,"quantile",quantile)
    IV_expansion_utils.testify_IV(S0,strike,T,r,q,a,b,N1,N2,quantile,fixVol)
    CFS_expansion_utils.testify_CFS_IV(S0=S0,strike=strike,T=T,r=r,q=q,
                                   fixVol=fixVol,N1=N1,N2=N2,quantile=quantile)
    return

print("***************************************************************************")
print("Sensitivity Analysis")
print("Basic setting: S0=100,r=0.05,q=0.0,N1=16,quantile=10")
print("The expansion coefficient is calculated based on a fixed volatility 0.2")
print("The experiment is to testify the sensitivity of implied volatility method with respect to different strikes and different maturities")
print("strikes: 80, 100, 120")
print("maturities: 0.1, 1, 3")
print("Below is the result for COS expansion and CFS expansion")
print("Only list the comparison of variances which is the square of volatilities")
i=0
S0=100
for strike in [80,100,120]:
    for T in [0.1,1,3]:
        i+=1
        print("*************************************")
        print("Trial ",i)
        print("variable triplet (S0,strike,T)=(",S0,',',strike,',',T,")")
        sensitivityAnalysis(S0 = S0, T=T, strike = strike, r = 0.05,
                    q = 0.0, fixVol = 0.3, N1 = 16,quantile = 10)
print("*******************************************************")
print("CONCLUSION:")
print("Only in the market case with COS method with relatively low volatility(vol<0.6 or var<0.36), the result doesn't go crazy. But still far from acceptible")