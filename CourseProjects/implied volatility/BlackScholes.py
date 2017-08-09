from scipy import stats
import math

def blackScholes (v, optionType, s, k, t, rf, div):
        """ Price an option using the Black-Scholes model.
        s: initial stock price
        k: strike price
        t: expiration time
        v: volatility
        rf: risk-free rate
        div: dividend
        cp: +1/-1 for call/put
        """
        if(optionType in['c',"call",1]):
            cp_flag = 1
        elif(optionType in ['p',"put",-1]):
            cp_flag = -1

        if v == 0:
            return cp_flag*(s-k*math.exp(-rf*t))
        d1 = (math.log(s/k)+(rf-div+0.5*math.pow(v,2))*t)/(v*math.sqrt(t))
        d2 = d1 - v*math.sqrt(t)

        optprice = (cp_flag*s*math.exp(-div*t)*stats.norm.cdf(cp_flag*d1)) - (cp_flag*k*math.exp(-rf*t)*stats.norm.cdf(cp_flag*d2))
        return optprice

def bs_vega(cp_flag,S,K,T,r,v,q=0.0):
    d1 = (math.log(S/K)+(r+v*v/2.)*T)/(v* math.sqrt(T))
    return S * math.sqrt(T)*stats.norm.cdf(d1)