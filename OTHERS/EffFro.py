import numpy as np
import scipy.optimize
import pandas as pd
from matplotlib.finance import quotes_historical_yahoo_ohlc

# Step 1: input area
tickers=['^GSPC','IBM','WMT','AAPL','C','MSFT']
begtime=(2015,1,1)
endtime=(2016,7,1)
# Step 2: define function
def obj_fun(w,rets):
    cov=np.cov(rets.T)
    port_var=np.dot(np.dot(w,cov),w)
    port_ret=np.dot(rets.mean(),w)
    return -port_ret/port_var

def monthly_ret(ticker,begtime,endtime):
    p = quotes_historical_yahoo_ohlc(ticker,begtime,endtime,
            asobject=True,adjusted=True)
    date = []
    for i in np.arange(len(p.date)-1):
        date.append(p.date[i+1].strftime("%Y%m"))
    logret = np.log(p.aclose[1:]/p.aclose[:-1])
    ret = pd.DataFrame(logret,date,columns=[ticker])
    return np.exp(ret.groupby(ret.index).sum())-1

# Step 3: estimate optimal portfolio for given return
w0 = np.ones(len(tickers))/len(tickers)
rets = monthly_ret(tickers[0],begtime,endtime)
for i in range(len(tickers)-1):
    rets=pd.merge(rets,monthly_ret(tickers[i+1],begtime,endtime),
        left_index=True,right_index=True)
print(rets)
bnds = [(0,1) for i in range(len(tickers))]
cons = {'type':'eq','fun': lambda w: sum(w)-1}
results = scipy.optimize.minimize(fun=obj_fun,x0=w0,args=rets,bounds=bnds,
        constraints=cons)
print(results)


