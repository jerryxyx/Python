import pandas as pd
import numpy as np
import urllib.request
import datetime as dt
import matplotlib.pyplot as plt


def get_google_data(symbol, period, window):
    url_root = 'http://www.google.com/finance/getprices?i='
    url_root += str(period) + '&p=' + str(window)
    url_root += 'd&f=d,o,h,l,c,v&df=cpct&q=' + symbol
    response = urllib.request.urlopen(url_root)
    data = response.read().decode().split('\n')
    # actual data starts at index = 7
    # first line contains full timestamp,
    # every other line is offset of period from timestamp
    parsed_data = []
    anchor_stamp = ''
    end = len(data)
    for i in range(7, end):
        cdata = data[i].split(',')
        if 'a' in cdata[0]:
        # first one record anchor timestamp
            anchor_stamp = cdata[0].replace('a', '')
            cts = int(anchor_stamp)
        else:
            try:
                coffset = int(cdata[0])
                cts = int(anchor_stamp) + (coffset * period)
                parsed_data.append((dt.datetime.fromtimestamp(float(cts)), float(cdata[1]), float(cdata[2]), float(cdata[3]),
                        float(cdata[4]), float(cdata[5])))
            except:
                pass  # for time zone offsets thrown into data
    df = pd.DataFrame(parsed_data)
    df.columns = ['ts', 'o', 'h', 'l', 'c', 'v']
    df.index = df.ts
    del df['ts']
    return df








# import urllib.request
# import datetime as dt
# import pandas as pd
#
# def get_google_data(symbol, period, window, exch = 'NYSE'):
#     url_root = ('http://www.google.com/finance/getprices?i='
#                 + str(period) + '&p=' + str(window)
#                 + 'd&f=d,o,h,l,c,v&df=cpct'
#                 + '&x=' + exch.upper()
#                 + '&q=' + symbol.upper())
#     # print(url_root)
#     response = urllib.request.urlopen(url_root)
#     data=response.read().decode().split('\n')       #decode() required for Python 3
#     data = [data[i].split(',') for i in range(len(data)-1)]
#     header = data[0:7]
#     data = data[7:]
#     header[4][0] = header[4][0][8:]                 #get rid of 'Columns:' for label row
#     df=pd.DataFrame(data, columns=header[4])
#     df = df.dropna()                                #to fix the inclusion of more timezone shifts in the .csv returned from the goog api
#     df.index = range(len(df))                       #fix the index from the previous dropna()
#
#     ind=pd.Series(len(df))
#     for i in range(len(df)):
#         if df['DATE'].ix[i][0] == 'a':
#             anchor_time = dt.datetime.fromtimestamp(int(df['DATE'].ix[i][1:]))  #make datetime object out of 'a' prefixed unix timecode
#             ind[i]=anchor_time
#         else:
#             ind[i] = anchor_time +dt.timedelta(seconds = (period * int(df['DATE'].ix[i])))
#     df.index = ind
#
#     df=df.drop('DATE', 1)
#
#     for column in df.columns:                #shitty implementation because to_numeric is pd but does not accept df
#         df[column]=pd.to_numeric(df[column])
#
#     return df

#print(get_google_data('IBM', 60, 2))