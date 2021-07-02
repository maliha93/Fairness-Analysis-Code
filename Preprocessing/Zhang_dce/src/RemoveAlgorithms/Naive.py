import random, time
import pandas as pd
import numpy as np
import os
from Basic import testdata, adult, dutch
from Detection import Judge
from Utility import Utility

'''
in previous version, seed = time.time(). There may be a little bit difference.
'''


def Naive(tau, data, C, E, Qs, Xs, Ys):
    df = data.copy()
    df.index = range(df.__len__())
    n_pos = df[df[C['name']] == C['pos']].__len__()
    df.loc[:, C['name']] = C['neg']
    random.seed(684564198)
    rindex = random.sample(list(df.index), n_pos)
    df.iloc[rindex, df.columns.get_loc(C['name'])] = C['pos']

    return df


if __name__ == '__main__':
    os.chdir('..')
    # data, C, E, Qs, Xs, Ys = testdata()
    #
    # avg, std, nonemptyset, largerset, smallerset, minvalue, maxvalue = Judge(df=data, C = C, E = E, Qs = Qs)
    #
    # print('aveage:    %0.3f' % avg)
    # print('sigma:     %0.3f' % std)
    # print('non-empty: %0.3f' % nonemptyset)
    # print('large set: %0.3f' % largerset)
    # print('small set: %0.3f' % smallerset)
    # print('max:       %0.3f' % maxvalue)
    # print('min:       %0.3f' % minvalue)
    #
    # random.seed(96312)
    # df = Naive(0.05, data, C, E, Qs, Xs, Ys)
    # avg, std, nonemptyset, largerset, smallerset, minvalue, maxvalue = Judge(df=df, C = C, E = E, Qs = Qs)
    #
    # print('aveage:    %0.3f' % avg)
    # print('sigma:     %0.3f' % std)
    # print('non-empty: %0.3f' % nonemptyset)
    # print('large set: %0.3f' % largerset)
    # print('small set: %0.3f' % smallerset)
    # print('max:       %0.3f' % maxvalue)
    # print('min:       %0.3f' % minvalue)

    # os.chdir('..')
    # data, C, E, Qs, Xs, Ys = adult()
    #
    # avg, std, nonemptyset, largerset, smallerset, minvalue, maxvalue = Judge(df=data, C = C, E = E, Qs = Qs)
    #
    # print('aveage:    %0.3f' % avg)
    # print('sigma:     %0.3f' % std)
    # print('non-empty: %0.3f' % nonemptyset)
    # print('large set: %0.3f' % largerset)
    # print('small set: %0.3f' % smallerset)
    # print('max:       %0.3f' % maxvalue)
    # print('min:       %0.3f' % minvalue)
    #
    # for i in range(1000000):
    #     random.seed(i)
    #     df = Naive(0.05, data, C, E, Qs, Xs, Ys)
    #     avg, std, nonemptyset, largerset, smallerset, minvalue, maxvalue = Judge(df=df, C = C, E = E, Qs = Qs)
    #
    #     if largerset + smallerset < 20:
    #         print(i, largerset + smallerset)

    for tau in [0.05]:
        for Data_Func in [testdata]:
            print('Dataset: %s' % Data_Func.__name__)
            result = pd.DataFrame(data=np.zeros((6, 5)),
                                  index=['MGraph', 'MData', 'Naive', 'LM', 'LPS', 'DI'],
                                  columns=['Distance', 'n_T', 'chisqr', 'detect', 'non-empty'])

            for Remove_func in [Naive]:
                data, C, E, Qs, Xs, Ys = Data_Func()
                data_new = Remove_func(tau, data, C, E, Qs, Xs, Ys)
                result.loc[Remove_func.__name__, :3] = Utility(data, data_new)

                avg, std, nonemptyset, largerset, smallerset, minvalue, maxvalue = Judge(data_new, C, E, Qs)
                result.loc[Remove_func.__name__, 3:] = [largerset + smallerset, nonemptyset]
            print(result)
