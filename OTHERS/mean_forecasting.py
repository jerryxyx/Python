import pandas as pd
import scipy as sp
import numpy as np
from matplotlib.finance import quotes_historical_yahoo_ohlc

ticker='IBM'
begtime = (2015,1,1)
endtime = (2015,12,31)

def month_ret(ticker,begtime,endtime):
    prices = quotes_historical_yahoo_ohlc(ticker,begtime,endtime,asobject=True,
            adjusted=True)
    log_ret = np.log(prices.aclose[1:]/prices.aclose[:-1])
    month_date = []
    for i in np.arange(len(log_ret)):
        month_date.append(prices.date[i+1].strftime('%Y%m'))
    log_ret_df = pd.DataFrame(data=log_ret,index=month_date,columns=[ticker])
    return np.exp(log_ret_df.groupby(log_ret_df.index).sum())-1

def month_rets(tickers,begtime,endtime):
    ret0 = month_ret(tickers[0],begtime,endtime)
    if(len(tickers)>1):
        for i,ticker in enumerate(tickers[1:]):
            ret_ = month_ret(ticker,begtime,endtime)
            ret0 = pd.merge(ret0,ret_,left_index=True,right_index=True)
    return ret0

def port_var(tickers,begtime,endtime,w=None):
    rets = month_rets(tickers,begtime,endtime)
    cov = sp.cov(rets.T)
    if(w==None):
        w=np.ones(len(tickers))/len(tickers)
    port_var = np.dot(np.dot(w,cov),w)
    return port_var

def port_ret(tickers,begtime,endtime,w=None):
    rets = month_rets(tickers,begtime,endtime)
    if(w==None):
        w=np.ones(len(tickers))/len(tickers)
    return np.dot(rets.mean(),w)

def obj_f(tickers,begtime,endtime,w=None):
    ret = port_ret(tickers,begtime,endtime,w)
    var = port_var(tickers,begtime,endtime,w)
    return ret/var


