import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib.finance import quotes_historical_yahoo_ohlc

ff=pd.read_csv('https://raw.githubusercontent.com/alexpetralia/fama_french/master/FF_monthly.CSV',index_col='Date')
begtime=(2008,10,1)
endtime=(2013,11,30)
price=quotes_historical_yahoo_ohlc('IBM',begtime,endtime,asobject=True,adjusted=True)
logret=np.log(price.aclose[1:]/price.aclose[:-1])
month=[]
for i in price.date:
    month.append(int(i.strftime("%Y%m")))

ret=pd.DataFrame(data=logret,index=month[1:],columns=['ret'])
ret=np.exp(ret.groupby(ret.index).sum())-1
final=pd.merge(ff,ret,right_index=True,left_index=True)
y=final.ret
x=final[['Mkt-RF','SMB','HML']]
x=sm.add_constant(x)
results = sm.OLS(y,x).fit()
print(results.params)
