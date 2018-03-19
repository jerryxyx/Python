import numpy as np
import matplotlib.pyplot as plt

n_iters=100;
sz=(n_iters,)
xhat=np.zeros(sz)
xhatminus=np.zeros(sz)
P=np.zeros(sz)
Pminus=np.zeros(sz)
K=np.zeros(sz)
#xhat=xhatminus=P=Pminus=K=np.zeros(sz)

R=0.1
Q=0.001
P[0]=1
xhat[0]=0
actual_value=-.377
z=np.random.normal(actual_value,0.1,size=sz)
def kalmanfun(Q,R,n_iters):
    for i in range(1,n_iters):
        xhatminus[i]=xhat[i-1]
        Pminus[i]=P[i-1]+Q
        K[i]=Pminus[i]/(Pminus[i]+R)
        xhat[i]=xhatminus[i]+K[i]*(z[i]-xhatminus[i])
        P[i]=(1-K[i])*Pminus[i]
    plt.plot(z,'k+',label='noisy measurement')
    plt.axhline(actual_value,color='g',label='actual value')
    plt.plot(xhat,'b-',label='posteri estimate')
   # plt.legend()
    plt.show()
    return



kalmanfun(0.0001,0.01,100)
