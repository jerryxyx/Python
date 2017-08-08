import quandl
import numpy as np
import pandas as pd
import os.path

# quandl.ApiConfig.api_key = "pgZTGbk8X-D7SdwAS_wN"

# 9.1
nasdaqDF = pd.read_csv("nasdaq100list.csv")
symbols = nasdaqDF.ix[:, 'Symbol'].sort_values().tolist()
tickers = symbols[0:10]
print("tickers:", tickers)
if os.path.isfile("stockDataFrame.csv"):
    df = pd.read_csv("stockDataFrame.csv")
    df.index = df.Date
    df = df.ix[:, 1:]
else:
    closeList = []
    for ticker in tickers:
        stockDF = quandl.Dataset('WIKI/' + ticker).data(
            params={'start_date': '2015-01-01', 'end_date': '2016-12-31'}).to_pandas()
        stockClose = stockDF.ix[:, 'Adj. Close']
        closeList.append(stockClose)

    df = pd.concat(closeList, axis=1)
    df.columns = tickers

# 9.2
closeMatrix = df.as_matrix()
returnMatrix = (closeMatrix[1:, :] - closeMatrix[:-1, :]) / closeMatrix[:-1, :]

# 9.3
meanDailyReturn = returnMatrix.mean(axis=0)
print("mean daily return:")
print(meanDailyReturn)

# 9.4
covariance = np.cov(returnMatrix.T)
print("covariance matrix:")
print(covariance)
