import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def calculateDiscountedFactorCorrespondingToFowardRate(forwardRate):
    DF = []
    DF.append(1)
    for i,forward in enumerate(forwardRate):
        DF.append(DF[i]/(1+forward))
    return DF

def calculatePV(cashflow,DF):
    return np.dot(np.asarray(cashflow),np.asarray(DF))

def calculatePV2(cashflow,y):
    DF2 = [1/(1+y)**(i+1) for i in range(len(cashflow))]
    return np.dot(np.asarray(cashflow),np.asarray(DF2))

def calculateCouponAnnualYield(PV,cashflow):
    fun = lambda y: calculatePV2(cashflow,y)-PV
    x0= scipy.optimize.brentq(fun,0,1)
    return x0

def calculateDV01(PV,cashflow,dy=1e-6):
    y = calculateCouponAnnualYield(PV,cashflow)
    dP = calculatePV2(cashflow,y+dy)-PV
    return -dP/dy/10000

def calculateDuration(PV,cashflow,dy=1e-6):
    y = calculateCouponAnnualYield(PV, cashflow)
    dP = calculatePV2(cashflow, y + dy) - PV
    return -dP/PV/dy

# a
print("***********************************************")
print("(a)")
df = pd.read_excel("Input1.xlsx")
print(df)
# b
print("***********************************************")
print("(b)")
cashflow = df['cash flow'][1:10]
PV = calculatePV(df['cash flow'][1:10],df['discount'][1:10])
print("PV:",PV)
print(calculateCouponAnnualYield(PV,cashflow))
# c
print("***********************************************")
print("(d)")
PVs = []
PVs.append(PV)
DV01s = []
Durations = []

for i in range(0,4):
    increment = i*0.01
    newDF = calculateDiscountedFactorCorrespondingToFowardRate(df['forward'][1:10]+increment)
    newPV = calculatePV(cashflow,newDF[1:])
    PVs.append(newPV)
    newDV01 = calculateDV01(newPV, cashflow)
    newDuration = calculateDuration(newPV, cashflow)
    DV01s.append(newDV01)
    Durations.append(newDuration)
increments = [i*0.01 for i in range(0,4)]
f,ax = plt.subplots(3,1,sharex=True)
ax[0].plot(increments,PVs[1:])
# ax[0].set_title('PV-increment')
ax[0].set_ylabel('PV')
ax[1].plot(increments,DV01s)
ax[1].set_ylabel('DV01')
ax[2].plot(increments,Durations)
ax[2].set_ylabel('Duration')
vals = ax[2].get_xticks()
ax[2].set_xticklabels(['{:3.2f}%'.format(x*100) for x in vals])
ax[2].set_xlabel('Forward curve increment')
plt.show()

# save the above data
df2 = pd.DataFrame()
df2['increment'] = increments
df2['PV'] = PVs[1:]
df2['DV01'] = DV01s
df2['Duration'] = Durations
print(df2)
df2.to_csv("Output1.csv",mode='w')

# d
print("***********************************************")
print("(c)")
# print(df2[df2.increment==0.01])
# print(df2[df2.increment==0.02])
# print(df2[df2.increment==0.03])
forward = df.forward[1:10]
PVs_c = []
PVs_c.append(PV)
DV01s_c = []
Durations_c =[]
for i in range(1,10):
    newForward = forward.copy()
    newForward[i]=forward[i]+0.005
    print(newForward)
    newDiscount=calculateDiscountedFactorCorrespondingToFowardRate(newForward)
    newPV = calculatePV(cashflow,newDiscount[1:])
    PVs_c.append(newPV)
    newDV01 = calculateDV01(newPV, cashflow)
    newDuration = calculateDuration(newPV, cashflow)
    DV01s_c.append(newDV01)
    Durations_c.append(newDuration)
# save the above data
df3 = pd.DataFrame()
df3['maturity'] = [i for i in range(1,10)]
df3['PV'] = PVs_c[1:]
df3['DV01'] = DV01s_c
df3['Duration'] = Durations_c
print(df3)
df3.to_csv("Output2.csv",mode='w')


# e
# semianually
print("***********************************************")
print("(e)")
discountedFactor18m = df[df.maturity==3].discount.values[0]
forwardPrice = PV/discountedFactor18m
print("forward price 18 months from today: ",forwardPrice)

# f
print("***********************************************")
print("(f)")
duration = Durations[0]
print("duration of the bond: ", duration)