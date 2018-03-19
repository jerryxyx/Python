import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import datetime

df = pd.read_csv('sp1500.csv')
print(df.head())

df['datetime'] = df.DATE.apply(lambda x: datetime.datetime.strptime(str(x),"%Y%m%d"))
print(df.head())
df.index = df.datetime
spdf = df.sprtrn
print(spdf.head())
spdf.plot()
plt.show()


# stationarity test
from statsmodels.tsa.stattools import adfuller
def test_stationary(time_series):
    rolmean = time_series.rolling(window=12).mean()
    rolstd = time_series.rolling(window=12).std()
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

test_stationary(spdf)

# decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
def test_seasonality(time_series,freq):
    decomposition = seasonal_decompose(time_series,freq=freq)
    decomposition.plot()
    plt.show()

test_seasonality(spdf,12)


