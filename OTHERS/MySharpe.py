import numpy as np
import scipy as sp
import pandas as pd
from matplotlib.finance import quotes_historical_yahoo_ohlc
begtime = (1990,1,1)
endtime = (2012,12,31)
tickers = ('IBM','WMT','C')
rf = 0.0003

def ret_annual(ticker,begtime,endtime):
    x = quotes_historical_yahoo_ohlc(ticker,begtime,endtime,asobject=True,adjusted=True)
    logret = np.log(x.aclose[1:]/x.aclose[:-1])
    date = []
    for i in np.arange(len(x)-1):
        date.append(x.date[i+1].strftime("%Y")) #wrong: date=date.append()
    ret = pd.DataFrame(logret,date,columns=[ticker])
    return np.exp(ret.groupby(ret.index).sum())-1

def portfolio_var(rets,w):
    var_mat = sp.cov(rets.T) #rets.T
    return np.matrix(w)*np.matrix(var_mat)*np.matrix(w).T
#Or: return np.dot(np.dot(w,var_mat),w)


equal_w = np.ones(len(tickers))/len(tickers)
rets=pd.DataFrame()
for i,ticker in enumerate(tickers):
    if(i==0):
        rets=ret_annual(ticker,begtime,endtime)
    else:
        rets=pd.merge(rets,ret_annual(ticker,begtime,endtime),left_index=True,right_index=True)
var = portfolio_var(rets,equal_w)
portfolio_ret = np.matrix(equal_w)*np.matrix(rets.mean()).T
sharpe = float((portfolio_ret-rf)/np.sqrt(var))
print(sharpe)

