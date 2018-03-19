import pandas as pd
import numpy as np
from matplotlib.finance import quotes_historical_yahoo_ohlc

def LPSD(ticker,begtime,endtime):
    p=quotes_historical_yahoo_ohlc(ticker,begtime,endtime,asobject=True,
            adjusted=True)
    ff=pd.read_pickle('ffDaily.pickle')
    ret=(p.aclose[1:]-p.aclose[:-1])/p.aclose[1:]
    print(ff)
    print(ret)
    print(p.date[1:])
    x=pd.DataFrame(ret,p.date[1:],columns=['ret'])
    print(x)
    final=pd.merge(x,ff,left_index=True,right_index=True)
    print(final)
    k=final.ret-final.Rf
    k1=k[k>0]
    return np.std(k1)*np.sqrt(252)

print(LPSD('IBM',(2010,1,1),(2013,1,1)))
