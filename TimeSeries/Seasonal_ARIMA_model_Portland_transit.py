import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('portland-oregon-average-monthly-.csv', index_col=0)
# df.index.name=None
# df.reset_index(inplace=True)
df.drop(df.index[114], inplace=True)
start = datetime.datetime.strptime("1973-01-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,114)]
df.index =date_list
# df.index = df['index']
df.columns = ['riders']
df.riders = df.riders.apply(lambda x: int(x)*100)
print(df.tail())
df.riders.plot()
plt.show()

# decomposition
decomposition = seasonal_decompose(df.riders,freq=12)
decomposition.plot()
plt.show()

# stationarity
from statsmodels.tsa.stattools import adfuller
def test_stationary(time_series):
    # determine rolling statics
    rolmean = pd.rolling_mean(time_series,window=12)
    rolstd = pd.rolling_std(time_series,window=12)
    # rolstd = pd.rolling(window=12).std()
    # plot rolling statistics
    fig = plt.figure(figsize=(12,8))
    orig = plt.plot(time_series,color='blue',label="original")
    mean = plt.plot(rolmean,color='red',label="rolling mean")
    std = plt.plot(rolstd,color='black',label="rolling std")
    plt.legend(loc='best')
    plt.show()

    # perform Dickey-Fuller test
    dftest = adfuller(time_series,regression='c',autolag="BIC")
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationary(df.riders)
df.log_riders = df.riders.apply(lambda x: np.log(x))
test_stationary(df.log_riders)

# first difference
df.first_difference = df.riders - df.riders.shift(1)
print(df.first_difference.head())
test_stationary(df.first_difference.dropna(axis=0,inplace=False))

# seasonal difference
df.seasonal_difference = df.riders-df.riders.shift(12)
test_stationary(df.seasonal_difference.dropna(axis=0,inplace=False))

# first difference and seasonal difference
df.first_seasonal_difference = df.first_difference-df.first_difference.shift(12)
test_stationary(df.first_seasonal_difference.dropna(axis=0,inplace=False))

# autocorrelation
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df.first_seasonal_difference.iloc[13:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.first_seasonal_difference.iloc[13:], lags=40, ax=ax2)
plt.show()

# seasonal ARIMA with exogenous regressor SARIMAX
mod = sm.tsa.statespace.SARIMAX(df.riders,trend='n',order=(0,1,0),seasonal_order=(1,1,1,12))
res = mod.fit()
print(res.summary())

# prediction
df['forecast'] =res.predict(start=100,end=120,dynamic=False)
df['forecast_dynamic']=res.predict(start=100,end=120,dynamic=True)
df[['riders', 'forecast','forecast_dynamic']].plot(figsize=(12, 8))
plt.show() 


