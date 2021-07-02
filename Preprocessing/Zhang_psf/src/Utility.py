from __future__ import division
import pandas as pd
import math


def Utility(observed_data, exptected_data):
    '''
    Calculate Chi-square utility
    :param observed_data:
    :param exptected_data:
    :return: chi square
    '''
    o = observed_data.copy()
    e = exptected_data.copy()
    df = pd.concat((o, e), axis=0, ignore_index=True)
    df = df.drop_duplicates()
    df.index = range(df.__len__())

    L1 = 0
    L2 = 0
    Chisqr = 0

    for i in df.index:
        o = observed_data.copy()
        e = exptected_data.copy()
        q = df.iloc[i]
        for attr in q.index:
            o = o[(o[attr] == q[attr])]
            e = e[(e[attr] == q[attr])]
        L1 = L1 + abs(o.__len__() - e.__len__())
        L2 = L2 + math.pow((o.__len__() - e.__len__()), 2)
        Chisqr = (Chisqr + pow(o.__len__() - e.__len__(), 2) * 1.0 / e.__len__()) if e.__len__() > 0 else (
            Chisqr + pow(o.__len__() - e.__len__(), 2) * 1.0 / o.__len__())
    # print 'L1:', L1
    # print 'L2:', L2
    # print 'chisqrt:', Chisqr
    return [Chisqr]
