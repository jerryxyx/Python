import math
import logging

log = logging.getLogger("American Pricer")

def americanOption (optionType, spotPrice,
    strikPrice, expiration, riskFreeRate, volatility,
    divident, n_steps):

    dt=expiration/n_steps
    discountFactor = math.exp(-riskFreeRate * dt)
    rate = riskFreeRate - divident
    # if(optionType==1 or optionType=='c' or optionType=="call"):
    #     cp_flag = 1
    # elif(optionType==2 or optionType=='p' or optionType=="put"):
    #     cp_flag = -1
    if (optionType in ['c', "call", 1]):
        cp_flag = 1
    elif (optionType in ['p', "put", -1]):
        cp_flag = -1
    else:
        logging.error("wrong input for optionType!")
        return 0

    #if volatility equals to 0, trinomial tree model degenerated to single strand
    if(volatility==0):
        # expiration
        value = max(0,cp_flag*(spotPrice*math.exp(riskFreeRate*expiration)-strikPrice))
        # time to mature
        for step in range(n_steps-1,-1,-1):
            riskFreeOptionValue = value
            timeSpend = expiration-dt
            payoff = max(0,cp_flag*(spotPrice*math.exp(riskFreeRate*timeSpend)-strikPrice))
            value = max(riskFreeOptionValue,payoff)*discountFactor
        return value

    u = math.exp(volatility*math.sqrt(2*dt))
    #log.info("u:"+str(u))

    d = 1/u
    Pu = ((math.exp(rate * dt / 2) - math.exp(-volatility * math.sqrt(dt / 2)))\
          / (math.exp(volatility * math.sqrt(dt / 2)) - math.exp(-volatility * math.sqrt(dt / 2))))
    Pu = Pu*Pu

    #print("Pu:"+str(Pu))

    Pd = ((math.exp(volatility * math.sqrt(dt / 2)) - math.exp(rate * dt / 2))\
        /(math.exp(volatility * math.sqrt(dt / 2)) - math.exp(-volatility * math.sqrt(dt / 2))))
    Pd = Pd*Pd
    #print("Pd:" + str(Pd))
    Pm = 1-Pu-Pd

    value = []

    for i in range(0,2*n_steps+1,1):
        riskFreeOptionValue = max(0,cp_flag*(spotPrice * math.pow(u,max(i-n_steps,0))\
                                  * math.pow(d,max(n_steps-i,0))-strikPrice))
        #print("option value N:"+str(riskFreeOptionValue)
        value.append(riskFreeOptionValue)

    for step in range(n_steps-1,-1,-1):
        for i in range(0,step*2+1,1):
            payoff = cp_flag*(spotPrice*math.pow(u,max(i-step,0))\
                *math.pow(d,max(step-i,0))-strikPrice)
            riskFreeOptionValue = Pu*value[i+2] + Pm*value[i+1] + Pd*value[i]

            #print("payoff:"+str(payoff)+"option value:"+str(riskFreeOptionValue))

            value[i]=max(payoff,riskFreeOptionValue)*discountFactor

            #print("step"+str(step)+" value"+str(i)+" "+str(value[i]))

    return value[0]



