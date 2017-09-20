import pandas as pd
import numpy as np
import math
import scipy.optimize

def calculateAccruedInterest(coupon,daysToNextPayment,daysBetweenPayments):
    return coupon * (180-daysToNextPayment)/daysBetweenPayments

def calculateDirtyPrice(cleanPrice, accruedInterest):
    return cleanPrice - accruedInterest

# We assume we follow the 360-day year convention
def calculateDirtyPriceFromYield(couponRate, parValue, Ntc, YTM, numPayments):
    semiAnnualCoupon = couponRate * parValue/2
    dirtyPrice = 1/(1+YTM/2)**(Ntc/180) \
                 * (semiAnnualCoupon*2/YTM * (1+YTM/2 - (1 + YTM/2)**(1-numPayments))\
                 + parValue/(1+YTM/2)**(numPayments-1))
    return dirtyPrice

def solveYield(dirtyPrice, couponRate, parValue, Ntc, numPayments):
    fun = lambda y: calculateDirtyPriceFromYield(couponRate,parValue,Ntc,y,numPayments)-dirtyPrice
    YTM = scipy.optimize.brenth(fun,-0.5,0.5)
    return YTM

def NielsonSiegelFunction(b0,b1,b2,tau,m):
    y = b0 + b1*(1-np.exp(-m/tau))/(m/tau)+b2*((1-np.exp(-m/tau))/(m/tau)-np.exp(-m/tau))
    return y

def calculateTotalLoss(maturities, YTMs, weights, b0,b1,b2,tau):
    n = len(maturities)
    y = np.asarray(YTMs)
    yHat = np.asarray([NielsonSiegelFunction(b0,b1,b2,tau,m) for m in maturities])
    totalLoss = np.dot(weights, np.square(y - yHat))/n
    return totalLoss


def solveNielsonSiegel(maturities,YTMs,weights,disp=True):
    objectFunction = lambda x: calculateTotalLoss(maturities,YTMs,weights,x[0],x[1],x[2],x[3])
    constraints=({'type':'ineq',
                 'fun': lambda x: x[0]},
                 {'type':'ineq',
                  'fun': lambda x: x[1]+x[0]},
                 {'type':'ineq',
                  'fun': lambda x: x[3]})
    res = scipy.optimize.minimize(objectFunction,[0.1,-0.1,0.1,0.5],
                                  constraints=constraints,tol=1e-12,options={'disp':disp})
    # print(res)
    return res.x

def plotYield(maturites,NSParameters,YTMs=None):
    import matplotlib.pyplot as plt
    m = np.linspace(np.min(maturities),np.max(maturities),50)
    y = NielsonSiegelFunction(NSParameters[0],NSParameters[1],NSParameters[2],NSParameters[3],m)
    if(YTMs==None):
        plt.plot( m, y, '--')
    else:
        plt.plot(maturities, YTMs, 'x',m, y, '--')

    plt.xlabel("time to maturity (year)")
    plt.ylabel("yield")
    plt.show()
    return



df3 = pd.read_excel('BonusInput.xlsx')
# assume par value is 100
parValue = 100
YTMs = []
maturities = []
for i in range(len(df3)):
    cleanPrice = df3.ix[i]['clean price']
    daysToNextPayment = df3.ix[i]['time to next payment'] * 360
    # daysFromLastPayment = 180 - daysToNextPayment
    timeToMaturity = df3.ix[i]['time to maturity']
    numPayments = math.ceil(timeToMaturity/2)
    couponRate = df3.ix[i]['coupon rate']
    accruedInterest = calculateAccruedInterest(couponRate*parValue/2,daysToNextPayment,180)
    dirtyPrice = calculateDirtyPrice(cleanPrice,accruedInterest)
    YTM = solveYield(dirtyPrice,couponRate,parValue,daysToNextPayment,numPayments)
    YTMs.append(YTM)
    maturities.append(timeToMaturity)

# print(YTMs)
df3['yield to maturity'] = YTMs
weights1 = 1/np.array(maturities)
weights2 = [1 for i in range(len(maturities))]
res = solveNielsonSiegel(maturities=maturities,YTMs=YTMs,weights=weights2)
print(res)
plotYield(maturities,res,YTMs)

# another explanation
def calculateBondPriceByExponentialRate(coupon, parValue, paymentTimes, b0,b1,b2,tau):
    maturity = np.max(paymentTimes)
    PV_coupon = 0
    for m in paymentTimes:
        R = NielsonSiegelFunction(b0,b1,b2,tau,m)
        # print(R)
        PV_coupon += coupon * np.exp(-R*m)
    Rm = NielsonSiegelFunction(b0,b1,b2,tau,maturity)
    bondPrice = PV_coupon + parValue*np.exp(-Rm*maturity)
    return bondPrice

def calculateTotalLoss2(df3,b0,b1,b2,tau):
    totalLoss = 0
    for i in range(len(df3)):
        cleanPrice = df3.ix[i]['clean price']
        timeToNextPayment = df3.ix[i]['time to next payment']
        timeToMaturity = df3.ix[i]['time to maturity']
        paymentTimes = np.arange(timeToNextPayment,timeToMaturity+1e-6,0.5)
        weight = 1/timeToMaturity
        couponRate = df3.ix[i]['coupon rate']
        coupon = parValue*couponRate/2
        accruedInterest = calculateAccruedInterest(couponRate * parValue / 2, daysToNextPayment, 180)
        dirtyPrice = calculateDirtyPrice(cleanPrice, accruedInterest)
        # maturities.append(timeToMaturity)
        priceHat = calculateBondPriceByExponentialRate(coupon,parValue,paymentTimes,b0,b1,b2,tau)
        # print(priceHat)
        if(priceHat==np.inf):
            print(b0,b1,b2,tau)
        totalLoss += weight * np.square(priceHat-dirtyPrice)
    return totalLoss

def solveNielsonSiegel2(df3,disp=True):
    objectFunction = lambda x: calculateTotalLoss2(df3,x[0],x[1],x[2],x[3])
    constraints=({'type':'ineq',
                 'fun': lambda x: x[0]},
                 {'type':'ineq',
                  'fun': lambda x: x[1]+x[0]},
                 {'type': 'ineq',
                  'fun': lambda x: x[2] +1},
                 {'type':'ineq',
                  'fun': lambda x: x[3]},
                 {'type': 'ineq',
                  'fun': lambda x: 1 - x[0]},
                 {'type': 'ineq',
                  'fun': lambda x: 1 - x[1]},
                 {'type': 'ineq',
                  'fun': lambda x: 1 - x[2]},
                 )
    res = scipy.optimize.minimize(objectFunction,[0.1,0.1,0.1,2],
                                  constraints=constraints,tol=1e-12,
                                  options={'disp':disp})
    # print(res)
    return res.x
NSParameters = solveNielsonSiegel2(df3)
print(NSParameters)
plotYield(df3['time to maturity'],NSParameters)