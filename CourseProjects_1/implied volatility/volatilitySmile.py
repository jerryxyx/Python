import ImpliedVolatility
import BlackScholes
import pandas as pd
import matplotlib.pyplot as plt


filePath = "OptionPrices.xlsx"
excelFile = pd.ExcelFile(filePath)
aapldata = excelFile.parse('AAPL Equity')
#call options
cAAPL = aapldata[aapldata['OptName'].str.contains('AAPL US 03/17/17 C')].ix[:, 0:4]
cStrike = cAAPL.ix[:, 0]
cOptionValue = 0.5 * (cAAPL.ix[:, 2] + cAAPL.ix[:, 3])
#put options
pAAPL = aapldata[aapldata['OptName'].str.contains('AAPL US 03/17/17 P')].ix[:, 0:4]
pStrike = pAAPL.ix[:, 0]
pOptionValue = 0.5 * (pAAPL.ix[:, 2] + pAAPL.ix[:, 3])


# def volatilitySmile(optionValues, spot,strikes,optionType,timeToMature,rate,divident,n_steps=100,maxiter=50):
#     blackScholes = list()
#     americanOption = list()
#     for optionValue,strike in zip(optionValues,strikes):
#         blackScholesImpliedVolatility = ImpliedVolatility.blackScholesImpliedVolatility(optionValue,optionType,spot,strike,timeToMature,rate,divident,maxiter=maxiter)
#         print("BSM: ",blackScholesImpliedVolatility)
#         blackScholes.append(blackScholesImpliedVolatility)
#         americanOptionImpliedVolatility = ImpliedVolatility.americanImpliedVolatility(optionValue,optionType,spot,strike,timeToMature,rate,divident,n_steps,maxiter=maxiter)
#         print("American Trino: ",americanOptionImpliedVolatility)
#         americanOption.append(americanOptionImpliedVolatility)
#     #print(blackScholes)
#     #print(americanOption)
#     return (blackScholes,americanOption)

PBSM,PusOption = ImpliedVolatility.volatilitySmile(pOptionValue, 139.78, pStrike, -1, 14 / 365, -0.02, 0, 100, 1000)
CBSM,CusOption  =ImpliedVolatility.volatilitySmile(cOptionValue, 139.78, cStrike, 1, 14 / 365, -0.02, 0, 100, 1000)
pData={'pStrike':pStrike,'PBSM':PBSM,'PusOption':PusOption}
pVolatilityData = pd.DataFrame(data=pData)
cData={'cStrike':cStrike,'CBSM':CBSM,'CusOption':CusOption}
cVolatilityData = pd.DataFrame(data=cData)
pVolatilityData.to_csv("pImpliedVolatility.csv")
cVolatilityData.to_csv("cImpliedVolatility.csv")

cResidue = [i - j for (i, j) in zip(CBSM, CusOption)]
pResidue = [i - j for (i, j) in zip(PBSM, PusOption)]
figure,axarr = plt.subplots(2,2,sharex='col',sharey='col')
axarr[0,0].plot(cStrike, CBSM,'ro-')
axarr[0,0].plot(cStrike, CusOption,'bx--')
# axarr[0,0].plot(cStrike,CBSM,cStrike,CusOption,color='black')
# axarr[0,0].fill_between(cStrike,CBSM,CusOption,where=CusOption>CBSM,facecolor='blue',interpolate=True)
# axarr[0,0].fill_between(cStrike,CBSM,CusOption,where=CusOption<CBSM,facecolor='red',interpolate=True)
axarr[0,0].set_title("AAPL Call Smile")
axarr[0,1].plot(cStrike, cResidue)
axarr[0,1].set_title("AAPL Call Residue Between US and EU")
axarr[1,0].plot(pStrike,PBSM,'ro-')
axarr[1,0].plot(pStrike,PusOption,'bx--')
# axarr[1,0].plot(pStrike,PBSM,pStrike,PusOption,color='black')
# axarr[1,0].fill_between(pStrike,PBSM,PusOption,where=PusOption>PBSM,facecolor='blue',interpolate=True)
# axarr[1,0].fill_between(pStrike,PBSM,PusOption,where=PusOption<PBSM,facecolor='red',interpolate=True)
axarr[1,0].set_title("AAPL Put Smile")
axarr[1,1].plot(pStrike,pResidue)
axarr[1,1].set_title("AAPL Put Residue Between US and EU")

#f = lambda v,k: BlackScholes.blackScholes(v,-1,139.78,k,14/365,0.02,0)

figure.show()