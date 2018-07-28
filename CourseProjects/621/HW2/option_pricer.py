import numpy as np
import scipy as sp
import scipy.stats
import math
from cvxopt import matrix, solvers

class OptionPricer:
    def __init__(self,S0,K,T,r,sigma,underlying_process="geometric brownian motion"):
        self.underlying_process = underlying_process
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
    def simulate(self, n_trails, n_steps):
        dt = self.T/n_steps
        self.n_trails = n_trails
        self.n_steps = n_steps
        if(self.underlying_process=="geometric brownian motion"):
#             first_step_prices = np.ones((n_trails,1))*np.log(self.S0)
            log_price_matrix = np.zeros((n_trails,n_steps))
            normal_matrix = np.random.normal(size=(n_trails,n_steps))
            cumsum_normal_matrix = normal_matrix.cumsum(axis=1)
#             log_price_matrix = np.concatenate((first_step_prices,log_price_matrix),axis=1)
            deviation_matrix = cumsum_normal_matrix*self.sigma*np.sqrt(dt) + \
    (self.r-self.sigma**2/2)*dt*np.arange(1,n_steps+1)
            log_price_matrix = deviation_matrix+np.log(self.S0)
            price_matrix = np.exp(log_price_matrix)
            price_zero = (np.ones(n_trails)*self.S0)[:,np.newaxis]
            price_matrix = np.concatenate((price_zero,price_matrix),axis=1)
            self.price_matrix = price_matrix
        return price_matrix
    
    def BlackScholesPricer(self,option_type='c'):
        S = self.S0
        K = self.K
        T = self.T
        r = self.r
        sigma = self.sigma
        d1 = (np.log(S/K)+r*T +0.5*sigma**2*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        N = lambda x: sp.stats.norm.cdf(x)
        call = S * N(d1) - np.exp(-r*T) * K * N(d2)
        put = call - S + K*np.exp(-r*T)
        if(option_type=="c"):
            return call
        elif(option_type=="p"):
            return put
        else:
            print("please enter the option type: (c/p)")
        pass
    
    def BinomialTreePricer(self, n, am=False, option_type='c'):
        """
        Price an option using the Binomial Options Pricing model
        S: initial spot price of stock
        K: strick price of option
        sigma: volatility
        r: risk-free interest rate
        t: time to maturity (in years)
        div: dividend yield (continuous compounding)
        call: 1 if call option, -1 if put
        n: binomial steps
        am: True for American option, False for European option
        """
        S = self.S0
        K = self.K
        t = self.T
        r = self.r
        sigma = self.sigma
        div = 0
        if(option_type=='c'):
            call = 1
        elif(option_type=='p'):
            call = -1
        else:
            print("wrong option type, should only be (c/p)")
            return
        
        delta = float(t) / n
        u = math.exp(sigma * math.sqrt(delta))
        d = float(1) / u
        q = float((math.exp((r - div) * delta) - d)) / (u - d)  # Prob. of up step
        stock_val = np.zeros((n + 1, n + 1))
        opt_val = np.zeros((n + 1, n + 1))

        # Calculate stock value at maturity
        stock_val[0, 0] = S
        for i in range(1, n + 1):
            stock_val[i, 0] = stock_val[i - 1, 0] * u
            for j in range(1, i + 1):
                stock_val[i, j] = stock_val[i - 1, j - 1] * d

        # Recursion for option price
        for j in range(n + 1):
            opt_val[n, j] = max(0, call*(stock_val[n, j] - K))
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                opt_val[i, j] = \
                    (q * opt_val[i + 1, j] + (1 - q) * opt_val[i + 1, j + 1]) \
                    / math.exp(r * delta)
                if am:
                    opt_val[i, j] = max(opt_val[i, j], call*(stock_val[i, j] - K))
        return opt_val[0, 0]

    
    def MCPricer(self,n_steps,n_trails,option_type='c'):
        price_matrix = self.price_matrix
        # k = n_steps
        self.n_steps = n_steps
        self.n_trails = n_trails
        dt = self.T/self.n_steps
        df = np.exp(- self.r*dt)
        strike = self.K
        risk_free_rate = self.r
        time_to_maturity = self.T
        
        n_trails = self.n_trails
        n_steps = self.n_steps
        
        if(option_type=="c"):
            payoff = (price_matrix[:,n_steps] - strike)
        elif(option_type=="p"):
            payoff = (strike - price_matrix[:,n_steps])
        else:
            print("please enter the option type: (c/p)")
            return
        
        payoff = matrix(np.where(payoff<0,0,payoff))
        vk = payoff*df
        regular_mc_price = np.average(payoff*np.exp(-risk_free_rate*time_to_maturity))
        self.mc_price = regular_mc_price
        return regular_mc_price
        
    def OHMCPricer(self,n_steps,n_trails,option_type='c', func_list=[lambda x: x**0, lambda x: x]):
        def _calculate_Q_matrix(S_k,S_kp1,df,func_list):
            dS = df*S_kp1 - S_k
            A = np.array([func(S_k) for func in func_list]).T
            B = (np.array([func(S_k) for func in func_list])*dS).T
            return np.concatenate((-A,B),axis=1)
        self.n_steps = n_steps
        self.n_trails = n_trails
        self.simulate(n_trails=self.n_trails,n_steps=self.n_steps)
        price_matrix = self.price_matrix
        # k = n_steps
        dt = self.T/self.n_steps
        df = np.exp(- self.r*dt)
        n_basis = len(func_list)
        n_trails = self.n_trails
        n_steps = self.n_steps
        strike = self.K
        
        if(option_type=="c"):
            payoff = (price_matrix[:,n_steps] - strike)
        elif(option_type=="p"):
            payoff = (strike - price_matrix[:,n_steps])
        else:
            print("please enter the option type: (c/p)")
            return
        
        payoff = matrix(np.where(payoff<0,0,payoff))
        vk = payoff*df
#         print("regular MC price",regular_mc_price)
    
        # k = 1,...,n_steps-1
        for k in range(n_steps-1,0,-1):
            Sk = price_matrix[:,k]
            Skp1 = price_matrix[:,k+1]
            Qk = matrix(_calculate_Q_matrix(Sk,Skp1,df,func_list))
            P = Qk.T * Qk
            q = Qk.T * vk
            A = matrix(np.ones(n_trails,dtype=np.float64)).T * Qk
            b = - matrix(np.ones(n_trails,dtype=np.float64)).T * vk
            sol = solvers.coneqp(P=P,q=q,A=A,b=b)
            ak = sol["x"][:n_basis]
            bk = sol["x"][n_basis:]
            vk = matrix(np.array([func(price_matrix[:,k]) for func in func_list])).T*ak*df
        
        # k = 0
        v0 = vk
        S0 = price_matrix[:,0]
        S1 = price_matrix[:,1]
        dS0 = df*S1 - S0
        Q0 = np.concatenate((-np.ones(n_trails)[:,np.newaxis],dS0[:,np.newaxis]),axis=1)
        Q0 = matrix(Q0)
        P = Q0.T*Q0
        q = Q0.T*v0
        A = matrix(np.ones(n_trails,dtype=np.float64)).T * Q0
        b = - matrix(np.ones(n_trails,dtype=np.float64)).T * v0
        C1 = matrix(ak).T * np.array([func(S1) for func in func_list]).T
        sol = solvers.coneqp(P=P,q=q,A=A,b=b)
        self.sol = sol
#         residual_risk = (v0.T*v0 + 2*sol["primal objective"])/n_trails
#         self.residual_risk = residual_risk[0]    # the value of unit matrix
        
        return sol["x"][0]
    
    def pricing(self, option_type='c', func_list=[lambda x: x**0, lambda x: x]):
        
        def _calculate_Q_matrix(S_k,S_kp1,df,func_list):
            dS = df*S_kp1 - S_k
            A = np.array([func(S_k) for func in func_list]).T
            B = (np.array([func(S_k) for func in func_list])*dS).T
            return np.concatenate((-A,B),axis=1)
        
        price_matrix = self.price_matrix
        # k = n_steps
        dt = self.T/self.n_steps
        df = np.exp(- self.r*dt)
        n_basis = len(func_list)
        n_trails = self.n_trails
        n_steps = self.n_steps
        
        if(option_type=="c"):
            payoff = (price_matrix[:,n_steps] - strike)
        elif(option_type=="p"):
            payoff = (strike - price_matrix[:,n_steps])
        else:
            print("please enter the option type: (c/p)")
            return
        
        payoff = matrix(np.where(payoff<0,0,payoff))
        vk = payoff*df
        regular_mc_price = np.average(payoff*np.exp(-risk_free_rate*time_to_maturity))
        black_sholes_price = self.BlackScholesPricer(option_type)
#         print("regular MC price",regular_mc_price)
    
        # k = 1,...,n_steps-1
        for k in range(n_steps-1,0,-1):
            Sk = price_matrix[:,k]
            Skp1 = price_matrix[:,k+1]
            Qk = matrix(_calculate_Q_matrix(Sk,Skp1,df,func_list))
            P = Qk.T * Qk
            q = Qk.T * vk
            A = matrix(np.ones(n_trails,dtype=np.float64)).T * Qk
            b = - matrix(np.ones(n_trails,dtype=np.float64)).T * vk
            sol = solvers.coneqp(P=P,q=q,A=A,b=b)
            ak = sol["x"][:n_basis]
            bk = sol["x"][n_basis:]
            vk = matrix(np.array([func(price_matrix[:,k]) for func in func_list])).T*ak*df
        
        # k = 0
        v0 = vk
        S0 = price_matrix[:,0]
        S1 = price_matrix[:,1]
        dS0 = df*S1 - S0
        Q0 = np.concatenate((-np.ones(n_trails)[:,np.newaxis],dS0[:,np.newaxis]),axis=1)
        Q0 = matrix(Q0)
        P = Q0.T*Q0
        q = Q0.T*v0
        A = matrix(np.ones(n_trails,dtype=np.float64)).T * Q0
        b = - matrix(np.ones(n_trails,dtype=np.float64)).T * v0
        C1 = matrix(ak).T * np.array([func(S1) for func in func_list]).T
        sol = solvers.coneqp(P=P,q=q,A=A,b=b)
        self.sol = sol
        residual_risk = (v0.T*v0 + 2*sol["primal objective"])/n_trails
        self.residual_risk = residual_risk[0]    # the value of unit matrix
        
        return({"OHMC": sol["x"][0],"regular MC": regular_mc_price,"Black-Scholes":black_sholes_price})
    
    def hedging(self):
        S = self.S0
        K = self.K
        T = self.T
        r = self.r
        sigma = self.sigma
        d1 = (np.log(S/K)+r*T +0.5*sigma**2*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        N = lambda x: sp.stats.norm.cdf(x)
        return({"OHMC optimal hedge": self.sol["x"][1],"Black-Scholes delta hedge":-N(d1),"OHMC residual risk":self.residual_risk})
   