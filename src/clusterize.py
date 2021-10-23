import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

def roundDownTo5(row, col):
    return int(row[col]) - int(row[col])%5

standsFile = "../data/Aircraft_Stands_Private.csv"

stands = pd.read_csv(standsFile)

stands['Taxiing_Time'] = stands.apply(roundDownTo5, col='Taxiing_Time', axis=1)
stands['1'] = stands.apply(roundDownTo5,col='1', axis=1)
stands['2'] = stands.apply(roundDownTo5,col='2', axis=1)
stands['3'] = stands.apply(roundDownTo5,col='3', axis=1)
stands['4'] = stands.apply(roundDownTo5,col='4', axis=1)
stands['5'] = stands.apply(roundDownTo5,col='5', axis=1)


df = stands.groupby(by=['JetBridge_on_Arrival','JetBridge_on_Departure','1','2','3','4','5','Terminal','Taxiing_Time']).mean()

stands.to_csv("../data/Aircraft_Stands_Private_Round5.csv")