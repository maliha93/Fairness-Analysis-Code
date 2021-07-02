from __future__ import division
from math import ceil
from Basic import get_group, adult, dutch
from Detection import Judge
from Utility import Utility
import pandas as pd
import numpy as np
'''
in previous version, seed = time.time(). There may be a little bit difference.
'''


def MData(tau, data, C, E, Qs, Xs, Ys):
    np.random.seed(6435417)
    df = data.copy()
    groupbyQ = df.groupby(list(Qs.columns))

    for i in Qs.index:
        group_q = get_group(groupbyQ, tuple(Qs.iloc[i]))

        if group_q.__len__() > 0:
            groupbyC = group_q.groupby(C['name'])
            groupbyCE = group_q.groupby([C['name'], E['name']])

            n_cpos = get_group(groupbyC, (C['pos'])).__len__()
            n_cneg = get_group(groupbyC, (C['neg'])).__len__()
            p1 = get_group(groupbyCE, (C['pos'], E['pos'])).__len__() / n_cpos if n_cpos > 0 else 0.0
            p2 = get_group(groupbyCE, (C['neg'], E['pos'])).__len__() / n_cneg if n_cneg > 0 else 0.0

            # switch if there is an inversion
            C['pos'], C['neg'], n_cpos, n_cneg = [C['pos'], C['neg'], n_cpos, n_cneg] if p1 >= p2 else [C['neg'],
                                                                                                        C['pos'],
                                                                                                        n_cneg, n_cpos]
            p_delta = abs(p1 - p2)

            if p_delta > tau:
                if (n_cneg <= n_cpos and n_cneg >= int(0.5 / tau)) or (n_cneg > n_cpos and n_cpos < int(0.5 / tau)):
                    group_changed = get_group(groupbyCE, (C['neg'], E['neg']))
                    num_changed = min(int(ceil(n_cneg * (p_delta - tau))), group_changed.__len__())
                    index_sampled = group_changed.sample(num_changed).index
                    df.loc[index_sampled, E['name']] = E['pos']
                else:
                    group_changed = get_group(groupbyCE, (C['pos'], E['pos']))
                    num_changed = min(int(ceil(n_cpos * (p_delta - tau))), group_changed.__len__())
                    index_sampled = group_changed.sample(num_changed).index
                    df.loc[index_sampled, E['name']] = E['neg']

    return df


if __name__ == '__main__':
    import os

    tau = 0.05
    os.chdir('..')
    for Data_Func in [adult, dutch]:
        result = pd.DataFrame(data=np.zeros((6, 5)),
                              index=['MGraph', 'MData', 'Naive', 'LM', 'LPS', 'DI'],
                              columns=['Distance', 'n_T', 'chisqr', 'detect', 'non-empty'])

        data, C, E, Qs, Xs, Ys = Data_Func()
        data_new = MData(0.05, data, C, E, Qs, Xs, Ys)
        result.loc['MData', :3] = Utility(data, data_new)

        avg, std, nonemptyset, largerset, smallerset, minvalue, maxvalue = Judge(data_new, C, E, Qs)
        result.loc['MData', 3:] = [largerset + smallerset, nonemptyset]

        print(result)
