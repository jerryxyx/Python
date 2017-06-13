import matplotlib.finance as finance
import matplotlib.pyplot as plt
import numpy as np
import datetime
stocks=['IBM','GE','WMT','C','AAPL','^GSPC']
begtime = datetime.date(2016,1,1)
endtime = datetime.date(2016,3,1)
def ret_vol_f(stocks,begtime,endtime):
    ret = []
    vol = []
    for i in stocks:
        price = finance.quotes_historical_yahoo(i,begtime,endtime,asobject=True,adjusted=True)
        close=price.aclose
        log_ret = np.log(close[1:]/close[:-1])
        ret.append(np.exp(log_ret.sum())-1)
        vol.append(np.std(log_ret))
    return (ret,vol)
ret,vol = ret_vol_f(stocks,begtime,endtime)
plt.scatter(ret,vol,marker='o',c=np.arange(len(stocks)))
plt.xlabel("ret")
plt.ylabel('vol')
for label,x,y in zip(stocks,ret,vol):
    #plt.annotate(label,xy=(x,y),textcoords='offsetpoints')
    plt.annotate(label,xy=(x,y))
#plt.show()
