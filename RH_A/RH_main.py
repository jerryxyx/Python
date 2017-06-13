import os
import pandas as pd
import time



def read_f(root_path):
#read data from root_path and return prices for all stocks
    prices = []
    file_pathes = []
    for root,dirs,files in os.walk(root_path):
        for name in files:
            file_path = os.path.join(root,name)
            file_pathes.append(file_path)
            price = pd.read_excel(file_path,sheetname='sheet1',header=0,names=['code','ticker','date','close'],parse_cols=[0,1,2,6])
            prices.append(price)
    return prices



def analysis_f(prices):
# Analysis data from prices and return RH_table, undesirable_table
    RH_table=pd.DataFrame()
    undesirable_table=pd.DataFrame()    #Listed for less than 5 days
    for price in prices:
        if(len(price.query("date=='2000-03-03'or date=='2000-03-06' or date=='2000-03-07' or date=='2000-03-08' or date=='2000-03-09'"))<5):
            undesirable_table = undesirable_table.append({'code':price.code[0],'ticker':price.ticker[0]},ignore_index=True)
        else:
            today_index = price.query("date=='2000-03-09'").index
            today_index = today_index[0]    #covert int64index to int
            today_close = price.at[today_index,'close']
            max_close = price.loc[today_index-4:today_index,'close'].max()
            min_close = price.loc[today_index-4:today_index,'close'].min()
            RH = (max_close-today_close)/(max_close-min_close)
            RH_table = RH_table.append({'code':price.code[0],'ticker':price.ticker[0],'RH':RH},ignore_index=True)
    return RH_table,undesirable_table


# Reading data from directory
print("Step1: Reading data from directory...")
begtime=time.time()
prices=read_f('19991001-20000930')
endtime=time.time()
print("time consumed: {0}".format(endtime-begtime))
# Analysis data and calculate RH for each stock
print("Step2: Analysis data and calculate RH for each stock...")
begtime=time.time()
RH_table,undesirable_table=analysis_f(prices)
endtime=time.time()
print("time consumed: {0}".format(endtime-begtime))
# Rank and save data
print("Step3: Rank and save data...")
RH_table['rank']=RH_table['RH'].rank(method="min",ascending=True)
sort_table = RH_table.sort_values(by='rank')
writer=pd.ExcelWriter('RH_table.xlsx')
sort_table.to_excel(writer,'sheet1',columns=['code','ticker','RH','rank'],index=False)
undesirable_table.to_excel(writer,'sheet2',columns=['code','ticker'],index=False)
writer.save()
print("Done. The result has been saved in RH_table.xlsx in which sheet1 for RH_table and sheet2 for undesirable_table")

