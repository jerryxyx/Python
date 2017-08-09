import numpy as np
from intraday import get_google_data
import pandas as pd
import matplotlib.pyplot as plt

n_output = 2
threshold = 0.2
spy = get_google_data('SPY', 300, 50)
vxx = get_google_data('VXX', 900, 150)
spy['date'] = [df.date() for df in spy.index]
spy['time'] = [df.time() for df in spy.index]
vxx['date'] = [df.date() for df in vxx.index]
vxx['time'] = [df.time() for df in vxx.index]
spy_close_series = spy.pivot_table(index='time', values='c', columns='date')
spy_close_series = spy_close_series.fillna(method='bfill')
vxx_close_series = vxx.pivot_table(index='time', values='c', columns='date')
vxx_close_series = vxx_close_series.fillna(method='bfill')
spy_volumn_series = spy.pivot_table(index='time', values='v', columns='date')
spy_volumn_series = spy_volumn_series.fillna(method='bfill')
print("Actual number of days: %d" % (np.size(spy_volumn_series, 1)))
# spy_daily = spy_close_series.mean(0)
spy_volumn = spy_volumn_series.mean(1)
spy_pct_change = spy_close_series.pct_change()
spy_log_return = np.log(1+spy_pct_change)
spy_log_return = spy_log_return.drop(spy_log_return.index[0],axis=0)
vxx_pct_change = vxx_close_series.pct_change()
vxx_log_return = np.log(1+vxx_pct_change)
vxx_log_return = vxx_log_return.drop(vxx_log_return.index[0],axis=0)
vxx_log_return_mean = vxx_log_return.mean(axis=1)
vxx_cum_return_mean = [vxx_log_return_mean[:i].sum() for i in range(len(vxx_log_return_mean))]
plt.plot(vxx_log_return_mean)
plt.show()


# spy_daily_returns = 100 * (spy_daily.values[1:] / spy_daily[:-1] - 1)
# spy_daily_returns_bool = spy_daily_returns > threshold
