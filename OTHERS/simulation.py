import numpy as np
p=[]
def call_simu(s0,x,T,r,sigma,n_steps):
    dt=T/n_steps
    sT=s0
    p.append(s0)
    for i in np.arange(n_steps):
        e=np.random.normal()
        sT*=np.exp((r-0.5*sigma*sigma)*dt+e*sigma*np.sqrt(dt))
        p.append(sT)
    return max(sT-x,0)*np.exp(-r*T)
