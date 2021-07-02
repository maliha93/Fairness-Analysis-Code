from __future__ import division
import pandas as pd
import numpy as np
from math import pow, sqrt
from Basic import adult, dutch, get_group
import os


def Judge(df, C, E, Qs, tau):
    result = pd.DataFrame(data=np.zeros((Qs.__len__(), 5)),
                          columns=['p1', 'p2', 'diff', 'qualified', 'count'])
    len = df.__len__()
    groupbyQ = df.groupby(list(Qs.columns))

    for i in Qs.index:
        group_q = get_group(groupbyQ, tuple(Qs.iloc[i]))
        if group_q.__len__() > 0:
            groupbyC = group_q.groupby(C['name'])
            groupbyCE = group_q.groupby([C['name'], E['name']])

            c_pos = get_group(groupbyC, C['pos']).__len__()
            c_neg = get_group(groupbyC, C['neg']).__len__()
            p1 = get_group(groupbyCE, (C['pos'], E['pos'])).__len__() / c_pos if c_pos > 0 else 0.0
            p2 = get_group(groupbyCE, (C['neg'], E['pos'])).__len__() / c_neg if c_neg > 0 else 0.0
            quanlified = False if c_pos < 10 or c_neg < 10 else True
            result.iloc[i, :] = [p1, p2, p1 - p2, quanlified, group_q.__len__(), ]

    avg = np.average(result['diff'], weights=result['count'] / len)
    sigmasqr = pd.Series.sum(result.apply(lambda x: x['count'] / len * pow((x['diff'] - avg), 2), axis=1))
    std = sqrt(sigmasqr)
    nonemptyset = result[result['count'] > 0].__len__()

    # our target is qualified groups
    result = result[result['qualified']==True]
    largerset = result[(result['diff'] > tau)].__len__()
    smallerset = result[(result['diff'] < -tau)].__len__()
    minvalue = result['diff'].min()
    maxvalue = result['diff'].max()
    quanlified = result.__len__()

    return avg, std, quanlified, nonemptyset, largerset, smallerset, minvalue, maxvalue


if __name__ == '__main__':
    os.chdir('..')
    for Data_Func in [adult, dutch]:
        data, C, E, Qs, Xs, Ys = Data_Func()
        avg, std, quanlified, nonemptyset, largerset, smallerset, minvalue, maxvalue = Judge(data, C, E, Qs, 0.05)

        print('aveage:    %0.3f' % avg)
        print('sigma:     %0.3f' % std)
        print('quanlified/non-empty: %d/%d' % (quanlified, nonemptyset))
        print('large set: %d' % largerset)
        print('small set: %d' % smallerset)
        print('max:       %0.3f' % maxvalue)
        print('min:       %0.3f' % minvalue)
