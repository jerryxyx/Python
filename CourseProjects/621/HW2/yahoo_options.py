import urllib
import json
from pandas.io.json import json_normalize
import datetime
import pandas as pd

def get_option_yahoo(symbol,t_expired):
    """
    Get options data from yahoo finance
    symbol: underlying ticker
    t_expired: maturity date
    Return: calls_df, puts_df, quote_series
    Example:
        amzn_call, amzn_put, amzn_quote = get_option_yahoo('AMZN',datetime(2018,7,20))
    """
    unix_expiration = int((t_expired- datetime.datetime(1970,1,1)).total_seconds())
    request = urllib.request.Request("https://query2.finance.yahoo.com/v7/finance/options/{}?date{}"\
                                     .format(symbol,unix_expiration))
    response = urllib.request.urlopen(request)
    option_data = response.read()
    data = json.loads(option_data.decode("utf-8"))
    calls_data = data["optionChain"]["result"][0]["options"][0]["calls"][:]
    puts_data = data["optionChain"]["result"][0]["options"][0]["puts"][:]
    calls_df = pd.DataFrame(calls_data)
    puts_df = pd.DataFrame(puts_data)
    quote_series = pd.Series(data["optionChain"]["result"][0]["quote"])
    return calls_df, puts_df, quote_series