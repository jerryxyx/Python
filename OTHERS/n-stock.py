import pandas as pd
import matplotlib.finance as finance

url = 'http://chart.yahoo.com/table.csv?s='
final = pd.read_csv(url+'^GSPC',usecols=[0,6],index_col=0)
final.columns=['^GSPC']
tickers = ['IBM','dell','wmt']
for ticker in tickers:
    print(ticker)
    x = pd.read_csv(url+ticker,usecols=[0,6],index_col=0)
    x.columns=[ticker]
    final = pd.merge(final,x,right_index=True,left_index=True)

