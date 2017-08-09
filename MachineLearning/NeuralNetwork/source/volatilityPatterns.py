import numpy as np
from intraday import get_google_data
import pandas as pd
import matplotlib.pyplot as plt
import datetime

n_output = 2
threshold = 0.2
window =10

spy = get_google_data('SPY', 300, 50)
vxx = get_google_data('VXX', 300, 50)
spy['date'] = spy.index.date
spy['time'] = spy.index.time
spy['rolling_std'] = spy.c.rolling(window=window,min_periods=8,center=True).std()
vxx['date'] = vxx.index.date
vxx['time'] = vxx.index.time

spy_close_series = spy.pivot_table(index='time', values='c', columns='date')
spy_close_series = spy_close_series.dropna(thresh=np.size(spy_close_series,0)-3,axis=1)
spy_close_series = spy_close_series.fillna(method='bfill')

spy_rv_series = spy.pivot_table(index='time', values='rolling_std', columns='date')
spy_rv_series = spy_rv_series.ix[:,1:-1] #pattern2
spy_rv_mean = spy_rv_series.mean(axis=1) #pic2
spy_rv_corr = spy_rv_series.T.corr()
eig_vals,eig_vecs = np.linalg.eig(spy_rv_corr)
var_exp = eig_vals.cumsum()/eig_vals.sum()

# plt.plot(eig_vecs[0,:])
# plt.plot(eig_vecs[1,:])
# plt.show()

vxx_close_series = vxx.pivot_table(index='time', values='c', columns='date')
vxx_close_series = vxx_close_series.dropna(thresh=np.size(vxx_close_series,0)-3,axis=1)
vxx_close_series = vxx_close_series.fillna(method='bfill')

spy_volumn_series = spy.pivot_table(index='time', values='v', columns='date')
spy_volumn_series = spy_volumn_series.fillna(method='bfill')#pattern1
spy_volumn_mean = spy_volumn_series.mean(axis=1) #pic1
spy_pct_change = spy_close_series.pct_change()
spy_log_return = np.log(1+spy_pct_change)
spy_log_return = spy_log_return.drop(spy_log_return.index[0],axis=0)
vxx_pct_change = vxx_close_series.pct_change()
vxx_log_return = np.log(1+vxx_pct_change)
vxx_log_return = vxx_log_return.drop(vxx_log_return.index[0],axis=0)
vxx_cum_log_return = vxx_log_return.cumsum(axis=0) #pattern3
vxx_cum_log_return_mean = vxx_cum_log_return.mean(axis=1) #pic3

spy_corr = spy_close_series.T.corr()
eig_vals,eig_vecs = np.linalg.eig(spy_corr)
eig_vals = eig_vals.real
eig_vecs = eig_vecs.real




# spy_daily_returns = 100 * (spy_daily.values[1:] / spy_daily[:-1] - 1)
# spy_daily_returns_bool = spy_daily_returns > threshold
