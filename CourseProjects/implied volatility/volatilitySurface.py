import pandas as pd
import datetime
import numpy as np
import ImpliedVolatility
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

excelFile = pd.ExcelFile("OptionPrices.xlsx")
aaplData = excelFile.parse("AAPL Equity")
aaplData = aaplData.ix[:,0:6]
grouped = aaplData.groupby(by=['OptStrike','OptExpDate'])
strTimes = ['3/17/2017','4/21/17','5/19/17','6/16/17','7/21/17','8/18/17','10/20/17',
             '11/17/17','12/15/17','1/19/18','6/15/18','1/18/19']
dateTime = [datetime.date(2017,3,17),datetime.date(2017,4,21),datetime.date(2017,5,19),
            datetime.date(2017,6,16),datetime.date(2017,7,21),datetime.date(2017,8,18),
            datetime.date(2017,10,20),datetime.date(2017,11,17),datetime.date(2017,12,15),
            datetime.date(2018,1,19),datetime.date(2018,6,15),datetime.date(2019,1,18)]
priceDate = datetime.date(2017,3,3)
timeToMature = [(date-priceDate).days for date in dateTime]
strikeStamp = [i for i in range(45,245,5)]
ttt,kkk = np.meshgrid(timeToMature,strikeStamp)
vmatrix = list()
for tt,kk in zip(ttt,kkk):
    vlist = list()
    for t,k in zip(tt,kk):
        c = ImpliedVolatility.optionValueFun(grouped,'p',k.item(),t.item(),priceDate)
        v=ImpliedVolatility.blackScholesImpliedVolatility(c,'p',139.78,k,t/365,-0.1,0,100)
        print('t', t, 'k', k, 'c', c, 'v', v)
        vlist.append(v)
    vmatrix.append(vlist)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(kkk,ttt,vmatrix,cmap=cm.coolwarm)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()