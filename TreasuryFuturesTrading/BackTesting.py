"""
    class: BackTestingSystem
    author: Jerry Xia
    email: xyxjerry@gmail.com
    date: 16/Apr/2017
    modules:
     - data input
     - preprocessing
     - strategy
     - fitting
"""



import numpy as np
from datetime import datetime
import pandas as pd

class BackTestingSystem:


    def __init__(self, numEquities, pointPrices, tickSizePrices, margins):
        self.numEquities = numEquities
        if (len(pointPrices) == numEquities):
            self.pointPrices = np.array(pointPrices)
        else:
            print("number of equities unmatch: point prices")
        if (len(tickSizePrices) == numEquities):
            self.tickSizePrices = np.array(tickSizePrices)
        else:
            print("number of equities unmatch: tickSizes")
        if (len(margins) == numEquities):
            self.margins = np.array(margins)
        else:
            print("number of equities unmatch: margins")

    def set_rollDate(self, rollDate):
        self.rollDate = rollDate

    def get_rollDate(self):
        return self.rollDate

    def set_exitUpLevel(self, exitUpLevel):
        self.exitUpLevel = exitUpLevel

    def set_exitDownLevel(self, exitDownLevel):
        self.exitDownLevel

    def set_triggerS(self, triggerS):
        self.triggerS = triggerS

    def set_triggerT(self, triggerT):
        self.triggerT = triggerT

    def get_marginPrices(self):
        return self.margins / self.pointPrices

    def get_tickSizes(self):
        return self.pointPrices * tickSizePrices

    def set_AUM(self, AUM):
        self.AUM = AUM

    def set_rollingStats(self, dfRollingStats):
        self.dfRollingStats = dfRollingStats
        self.df = pd.concat([self.df, self.rollingStats], axis=1)

    def set_maxPoistions(self, maxPositions):
        self.maxPositions = 30

    def set_percentageInvested(self, pctInvest):
        self.percentageInvested = pctInvest

    def set_maxPositions(self, maxPositions):
        self.maxPositions = maxPositions

    def input_data(self, dfPrices, dfDurations, dfOptWeights, dfRollingStats):
        self.dfPrices = dfPrices
        self.dfDurations = dfDurations
        self.dfOptWeights = dfOptWeights
        self.df = pd.concat([self.dfPrices, self.dfDurations, self.dfOptWeights, dfRollingStats], axis=1)

    # todo: delete
    #     def input_whole_data(self,df):
    #         self.df = df

    def get_df(self):
        return self.df

    def time_delta_365(self, timeDelta):
        if (timeDelta.days > 0):
            return timeDelta.days / 365
        else:
            return 0

    def preprocessing(self):

        # todo: preprocessing
        print("****************************************************************")
        print("Start preprocessing...")
        # basic setting
        self.marginPrices = self.margins / self.pointPrices
        self.maxInitMargin = self.AUM * self.percentageInvested
        self.positionInitMargin = self.maxInitMargin / self.maxPositions
        self.tickSizes = self.pointPrices * self.tickSizePrices
        self.marginPrices = self.margins / self.pointPrices

        # time to maturity
        timeDeltas = self.rollDate - self.df.index
        self.df['TimeToMaturity'] = timeDeltas
        self.df.TimeToMaturity = self.df.TimeToMaturity.apply(self.time_delta_365)
        self.timeToMaturity = self.df.TimeToMaturity
        print(self.df.head())
        # future duration
        futureDurationsColumns = ["dfFutureDuration" + dur_str[8:] for dur_str in self.dfDurations.columns]
        self.dfFutureDurations = pd.DataFrame(index=self.df.index, columns=futureDurationsColumns)
        for index, row in self.dfDurations.iterrows():
            self.dfFutureDurations.loc[index, :] = (row - self.df.TimeToMaturity[index]).values

        # margin unit
        #         self.marginUnit = pd.Series(index = self.df.index, name="MarginUnit")
        #         for index, row in self.dfOptWeights.iterrows():
        #             self.marginUnit[index] = np.inner(np.abs(row.values), self.marginPrices)
        self.marginUnit = self.dfOptWeights.apply(lambda x: np.inner(np.abs(x), self.marginPrices), axis=1)
        self.marginUnit.rename("MarginUnit")

        # national
        self.portNotional = self.positionInitMargin / self.marginUnit
        self.portNotional.rename("PortNotional", inplace=True)

        # positions
        # todo: uncouple columns name
        positionsColumns = ["dfPosition" + dur_str[8:] for dur_str in self.dfDurations.columns]
        self.dfPositions = pd.DataFrame(index=self.df.index, columns=positionsColumns)
        for index, row in self.dfOptWeights.iterrows():
            self.dfPositions.loc[index, :] = row.values * self.portNotional[index] / self.pointPrices
        # tick size
        self.portTickSize = self.dfPositions.apply(lambda x: np.inner(np.abs(x), self.tickSizes), axis=1)
        self.portTickSize.rename("PortTickSize", inplace=True)

        # current price
        self.portPrice = pd.Series(index=self.df.index, name="PortPrice")
        for idx in self.df.index:
            self.portPrice[idx] = np.inner(self.dfPrices.loc[idx, :], self.dfOptWeights.loc[idx, :])

        # tick size price
        self.portTickSizePrice = pd.Series(index=self.df.index, name="PortTickSizePrice")
        for idx in self.df.index:
            self.portTickSizePrice[idx] = self.portTickSize[idx] / self.portNotional[idx]

        # z-score
        self.ZScore = pd.Series(index=self.df.index, name="ZScore")
        for idx in self.df.index:
            self.ZScore[idx] = (self.portPrice[idx] - self.df.RollingAvg[idx]) / self.df.RollingStd[idx]

        # t-score
        self.TScore = pd.Series(index=self.df.index, name="TScore")
        for idx in self.df.index:
            self.TScore[idx] = (self.portPrice[idx] - self.df.RollingAvg[idx]) / self.portTickSizePrice[idx]

        # concat all results
        self.df = pd.concat([self.df, self.dfFutureDurations, self.marginUnit, self.portNotional,
                             self.dfPositions, self.portTickSize, self.portPrice, self.portTickSizePrice,
                             self.ZScore, self.TScore], axis=1)

        print("Preprocessing finished!")
        print("****************************************************************")
        return self.df