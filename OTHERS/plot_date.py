import matplotlib.pyplot as plt
import matplotlib.finance as finance
import matplotlib.dates as dates
from matplotlib.dates import YEARLY,MONDAY
import datetime

begtime=datetime.date(2016,1,1)
endtime=datetime.date(2016,2,1)
p1=finance.quotes_historical_yahoo('IBM',begtime,endtime)
fig,ax = plt.subplots()
ax.xaxis.set_major_locator(dates.WeekdayLocator(dates.MONDAY))
rule = dates.rrulewrapper(YEARLY,byeaster=1,interval=1)
ax.xaxis.set_minor_locator(dates.RRuleLocator(rule))
ax.xaxis.set_major_formatter(dates.DateFormatter("%d/%m/%y %A"))
ax.xaxis.set_minor_formatter(dates.DateFormatter("%b %d"))
#ax.xaxis_date()
#ax.autoscale_view()
#plt.setp(plt.gca().get_xticklabels(),rotation=80,ha='right')
finance.candlestick(ax,p1,width=0.6)

plt.setp(ax.get_xticklabels(),rotation=80,ha='right')
#ax.xaxis_date()

