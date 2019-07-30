# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 21:44:25 2019

@author: houwenxin
"""

def CalCostOfCapital(StockVal, DebtVal, beta, StockRiskPrem, RiskFreeRate, DebtRiskPrem, TaxRate):
    return StockVal/(StockVal+DebtVal)*(RiskFreeRate+beta*StockRiskPrem)+ \
            DebtVal/(StockVal+DebtVal)*(1-TaxRate)*(RiskFreeRate+DebtRiskPrem)
            
#cost = CalCostOfCapital(6, 4, 1.2, 0.09, 0.08, 0, 0)
cost = 0.6 * 8 * (1-0.35) + 0.4*15
print(cost)