import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.finance import quotes_historical_yahoo
import datetime
import numpy


begtime = datetime.date(2014,1,1)
endtime = datetime.date.today()
ticker = '^GSPC'
price = quotes_historical_yahoo(ticker,begtime,endtime,asobject=True,adjusted=True)
x=price.date[:]
y = price.close[:]
fig,ax = plt.subplots(2,sharex=True)
ax[0].plot_date(x,y,'-')
ax[0].xaxis.set_major_locator(dates.MonthLocator(interval=6))
ax[0].xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
ax[0].grid(True)
fig.autofmt_xdate()

ax[1].plot_date(x,y,'-')
ax[1].xaxis.set_major_locator(dates.MonthLocator(interval=6))
ax[1].xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
ax[1].autoscale_view()
fig.autofmt_xdate()
