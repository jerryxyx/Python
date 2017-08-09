from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
z=0.325
def f(t):
    return stats.norm.pdf(t)
plt.ylim(0,0.45)
x=np.arange(-3,3,.1)
y1=f(x)
x2=np.arange(-4,z,1/40)
sum_=0
delta=.05
s=np.arange(-10,z,delta)
for i in s:
    sum_+=f(i)*delta
cdf=stats.norm.cdf(z)
print('sum');print(sum_)
print('ppf');print(cdf)
plt.plot(x,f(x))
plt.fill_between(x2,f(x2))
plt.annotate('z='+str(z),xy=(z,0))
plt.annotate('area is'+str(round(sum_,4))+'cdf is'+str(round(cdf,4)),xy=(-1,0.25),xytext=(-3.8,0.4),arrowprops=dict(facecolor='red',shrink=.01))


