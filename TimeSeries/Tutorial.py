import statsmodels.api as sm
import pandas as pd
from patsy import dmatrices
import matplotlib.pyplot as plt
import quandl

df = sm.datasets.get_rdataset("Guerry", "HistData").data
vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
df = df[vars]
df = df.dropna()

y,X = dmatrices('Lottery~Literacy+Wealth+Region',data=df,return_type='dataframe')

mod = sm.OLS(y,X)
res = mod.fit()
print(res.summary())

data = sm.datasets.scotland.load()



quandl.ApiConfig.api_key = "pgZTGbk8X-D7SdwAS_wN"
start = '2015-01-01'
end = '2018-01-01'
wti_data = quandl.get("CHRIS/CME_CL1",start_date=start,end_date=end)
brent_data = quandl.get("CHRIS/ICE_B1",start_date=start,end_date=end)
print(len(wti_data))
print(len(brent_data))
print(wti_data.head())
