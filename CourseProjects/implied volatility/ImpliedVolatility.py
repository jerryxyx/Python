import scipy.optimize
from BlackScholes import blackScholes
from AmericanPricer import americanOption
import math
import time
import numpy
import datetime

def americanImpliedVolatility(optionValue,optionType,spotPrice,strikePrice,
                              expiration,riskFreeRate,divident,n_steps,maxiter=50):
    f = lambda impliedVolatility: americanOption(optionType, spotPrice,
    strikePrice, expiration, riskFreeRate, impliedVolatility,
    divident, n_steps)-optionValue

    b = 10
    #  Vega is always positive for calls and puts for both European and American options.
    #  This is because an increase in volatility always increases the theoretical value of an option - call or put.
    #  So if f(0)>0 f(x)>0 is surely, there is no root
    if (f(0) > 0):
        print("Warning! the implied volatility do not exist in US option")
        return numpy.nan
    if (f(b)<0):
        print("Warning! the implied volatility is too high in US option")
        b=100



    # newtonSolution = scipy.optimize.newton(f,x0=1,maxiter=maxiter)
    # print("**************"+str(newtonSolution)+"*****************")
    # newtonSolution = math.fabs(newtonSolution)
    impliedVolatility = scipy.optimize.brentq(f,0,b,maxiter=maxiter)

    return impliedVolatility


def blackScholesImpliedVolatility(optionValue, optionType, spotPrice, strikePrice,
                              expiration, riskFreeRate, divident, maxiter=50):
    #time1 = time.time()
    if(numpy.isnan(optionValue)):
        return numpy.nan
    f = lambda impliedVolatility: blackScholes(impliedVolatility, optionType,
        spotPrice, strikePrice, expiration, riskFreeRate, divident)-optionValue

    #  Vega is always positive for calls and puts for both European and American options.
    #  This is because an increase in volatility always increases the theoretical value of an option - call or put.
    #  So if f(0)>0 f(x)>0 is surely, there is no root

    b = 10 #[a,b] is the interval of brent method
    if (f(0) > 0):
        print("Warning! the implied volatility do not exist in BSM option")
        print("strike price is "+str(strikePrice))
        return numpy.nan
    if (f(b)<0):
        print("Warning! the implied volatility is too high in US option")
        b = 100

    # Brenner and Subrahmanyam (1988) provided a closed form estimate of IV
    #x0=math.sqrt(2*math.pi/expiration)*optionValue/spotPrice
    #newton = scipy.optimize.newton(f, x0=x0, maxiter=maxiter)
    #print("**************" + str(newton) + "*****************")
    # As Newton method is not safe than brent, use brend instead

    # if optionType in ['p',"put",-1]:
    #     print("strike price: "+str(strikePrice))
    #     print("f(0): "+str(f(0)))
    #     print("f(10): "+str(f(10)))
    #     print("******************************")

    impliedVolatility = scipy.optimize.brentq(f,0,10,maxiter=maxiter)

    #root = scipy.optimize.root(f,x0=1)
    #time2 = time.time()
    #print("time: "+str(time2-time1))
    #print(newton)
    #time3=time.time()
    #g = lambda impliedVolatility: math.pow(f(impliedVolatility),2)

    return impliedVolatility

def volatilitySmile(optionValues, spot,strikes,optionType,timeToMature,rate,divident,n_steps=100,maxiter=50):
    blackScholes = list()
    americanOption = list()
    for optionValue,strike in zip(optionValues,strikes):
        BSMVol = blackScholesImpliedVolatility(optionValue,optionType,spot,strike,timeToMature,rate,divident,maxiter=maxiter)
        print("BSM: ",BSMVol)
        blackScholes.append(BSMVol)
        USVol = americanImpliedVolatility(optionValue,optionType,spot,strike,timeToMature,rate,divident,n_steps,maxiter=maxiter)
        print("American Trino: ",USVol)
        americanOption.append(USVol)
    #print(blackScholes)
    #print(americanOption)
    return (blackScholes,americanOption)

def optionValueFun(groupedData,optionType,strike,timeToMature,priceDate):
    #optionPutCall = optionType
    if(optionType in [1,'c',"call"]):
        optionPutCall = "Call"
    elif(optionType in [-1,'p',"put"]):
        optionPutCall = "Put"
    else:
        print("wrong option type!")
        return numpy.nan

    expirationDate = priceDate+datetime.timedelta(timeToMature)

    strDate = expirationDate.strftime("%d/%m/%y")
    try:
        groupedData=groupedData.get_group((strike,strDate))
    except:
        print("no such strike")
        print("strike:", strike, "strDate", strDate)
        return numpy.nan

    groupedData = groupedData[groupedData.OptPutCall==optionPutCall]
    #print("*********",groupedData)
    if(len(groupedData)==1):
        optionValue = float(groupedData.OptAsk+groupedData.OptBid)/2
        return optionValue
    if(len(groupedData)==0):
        print("null data")
        print("strike:", strike, "strDate", strDate)
        return numpy.nan

    elif(len(groupedData)>1):
        print("repeated data")
        print("strike:", strike, "strDate", strDate)
        return numpy.nan
