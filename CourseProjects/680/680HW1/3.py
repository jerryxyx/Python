import pandas as pd
import numpy as np

data = {'maturity':[3/12,6/12,1,2,3,5,7,10,30],'yield':np.array([0.14,0.19,0.25,0.51,0.75,1.41,2.02,2.57,3.63])*0.01}
df = pd.DataFrame(data=data)
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
x = df['maturity']
y = df['yield']
cs = CubicSpline(x,y,bc_type='natural')
xnew = np.linspace(0, 30, 100)
ynew = cs(xnew)
plt.figure()
plt.plot(x, y, 'x', xnew, ynew, '--')
plt.legend(['Linear', 'Cubic Spline', 'True'])
# plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.title('Cubic-spline interpolation')
plt.show()