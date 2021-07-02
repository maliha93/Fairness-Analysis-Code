from __future__ import division
import pandas as pd
import numpy as np
from Basic import adult, dutch, get_group
from Utility import Utility
from Detection import Judge
from math import pow, floor, ceil
from cvxopt import matrix, solvers


def MGraph(tau, data0, C, E, Qs, Xs, Ys):
    data = data0.copy()
    Q_len = Qs.__len__()
    len = data.__len__()
    num_cq = np.zeros(2 * Q_len)
    num_cq_ep = np.zeros(2 * Q_len)
    Q_mat = np.zeros((2 * Q_len, 2 * Q_len))
    p_vec = np.zeros((2 * Q_len))
    G_mat = np.zeros((6 * Q_len, 2 * Q_len))
    h_vec = np.zeros((6 * Q_len))

    groupbyQ = data.groupby(list(Qs.columns))
    for i in Qs.index:
        q = Qs.iloc[i]
        group_q = get_group(groupbyQ, tuple(q))

        if group_q.__len__() > 0:
            groupbyC = group_q.groupby(C['name'])
            groupbyCE = group_q.groupby([C['name'], E['name']])

            for j, c in enumerate([C['pos'], C['neg']]):
                group_c = get_group(groupbyC, c)
                num_cq[i + j * Q_len] = group_c.__len__()
                num_cq_ep[i + j * Q_len] = get_group(groupbyCE, (c, E['pos'])).__len__()

                Peta = num_cq_ep[i + j * Q_len] / num_cq[i + j * Q_len] \
                    if num_cq[i + j * Q_len] > 0 else 0.0

                if Ys.__len__() == 0:
                    Beta = 2 * pow(num_cq[i + j * Q_len] / len, 2)
                else:
                    Beta = 0.0
                    for k, e in enumerate([E['pos'], E['neg']]):
                        group_ce = get_group(groupbyCE, (c, e))
                        num_cq_e = group_ce.__len__()
                        P_ecq = num_cq_e / group_c.__len__()

                        groupbyY = group_ce.groupby(list(Ys.columns))
                        for l in Ys.index:
                            y = Ys.iloc[l]
                            group_y = get_group(groupbyY, tuple(y))
                            P_y = group_y.__len__() / num_cq_e
                            Beta = Beta + pow(P_ecq * P_y, 2)

                Q_mat[i + Q_len * j, i + Q_len * j] = Beta
                p_vec[i + Q_len * j] = -2 * Beta * Peta

        G_mat[i, i] = 1.0
        G_mat[i, i + Q_len] = -1.0
        G_mat[Q_len + i, i] = -1.0
        G_mat[Q_len + i, i + Q_len] = 1.0

        G_mat[2 * Q_len + i, i] = 1.0
        G_mat[3 * Q_len + i, i + Q_len] = 1.0
        G_mat[4 * Q_len + i, i] = -1.0
        G_mat[5 * Q_len + i, i + Q_len] = -1.0

    h_vec[0: 2 * Q_len] = tau
    h_vec[2 * Q_len: 4 * Q_len] = 1.0
    h_vec[4 * Q_len: 6 * Q_len] = 0.0

    Q = 2 * matrix(Q_mat)
    p = matrix(p_vec)
    G = matrix(G_mat)
    h = matrix(h_vec)

    solvers.options['feastol'] = 1e-9
    solvers.options['show_progress'] = False
    sol = solvers.qp(Q, p, G, h)

    x = np.array(sol['x']).reshape(2 * Q_len)
    # since the estimated numbers from the causal bayesian network are real numbers,
    # we have to convert them into integer numbers. To reduce the round bias, we adopt
    # floor and ceil, and choose the one who can minimize the round bias.

    for i in range(Q_len):
        n_cpep = num_cq[i] * x[i]
        n_cnep = num_cq[i + Q_len] * x[i + Q_len]
        joint = pd.DataFrame(data=[
            [floor(n_cpep), floor(n_cnep)],
            [floor(n_cpep), ceil(n_cnep)],
            [ceil(n_cpep), floor(n_cnep)],
            [ceil(n_cpep), ceil(n_cnep)]
        ], columns=['pos', 'neg'])
        joint['pr'] = abs(joint['pos'] / num_cq[i] - joint['neg'] / num_cq[i + Q_len])
        try:
            num_cq_ep[i], num_cq_ep[Q_len + i] = joint.iloc[joint['pr'].idxmin(), :2]
        except:
            num_cq_ep[i], num_cq_ep[Q_len + i] = joint.iloc[0, :2]
    # convert into integer numbers
    num_cq_ep = np.int_(num_cq_ep)
    num_cq_en = np.int_(num_cq - num_cq_ep)

    df_new = []
    for i in Qs.index:
        q = Qs.iloc[i]
        group_q = get_group(groupbyQ, tuple(q))
        groupbyCE = group_q.groupby([C['name'], E['name']])

        num_q_est = {
            (C['pos'], E['pos']): num_cq_ep[i],
            (C['pos'], E['neg']): num_cq_en[i],
            (C['neg'], E['pos']): num_cq_ep[i + Q_len],
            (C['neg'], E['neg']): num_cq_en[i + Q_len]
        }
        num_q_true = {
            (C['pos'], E['pos']): get_group(groupbyCE, (C['pos'], E['pos'])).__len__(),
            (C['pos'], E['neg']): get_group(groupbyCE, (C['pos'], E['neg'])).__len__(),
            (C['neg'], E['pos']): get_group(groupbyCE, (C['neg'], E['pos'])).__len__(),
            (C['neg'], E['neg']): get_group(groupbyCE, (C['neg'], E['neg'])).__len__()
        }

        for ce in num_q_est.keys():
            if Ys.__len__() == 0:
                df_new.extend([list(q) + list(ce)] * num_q_est[ce])

            else:
                count = 0
                group_ce = get_group(groupbyCE, ce)

                if group_ce.__len__() > 0:
                    g = group_ce.groupby(list(Ys.columns))
                    for y, group_y in sorted(g, key=lambda x: x[1].__len__(), reverse=True):
                        num_tuple = int(round(group_y.__len__() / num_q_true[ce] * num_q_est[ce]))
                        if count + num_tuple < num_q_est[ce]:
                            if num_tuple >= 1:
                                df_new.extend([list(q) + list(y) + list(ce)] * num_tuple)
                                count = count + num_tuple
                    else:
                        df_new.extend([list(q) + list(y) + list(ce)] * (num_q_est[ce] - count))

                else:  # create new tuples according the whole population of y
                    g = group_q.groupby(list(Ys.columns))
                    for y, group_y in sorted(g, key=lambda x: x[1].__len__(), reverse=True):
                        num_tuple = int(round(group_y.__len__() / len * num_q_est[ce]))
                        if count + num_tuple < num_q_est[ce]:
                            if num_tuple >= 1:
                                df_new.extend([list(q) + list(y) + list(ce)] * num_tuple)
                                count = count + num_tuple
                    else:
                        df_new.extend([list(q) + list(y) + list(ce)] * (num_q_est[ce] - count))

    return pd.DataFrame(data=df_new, columns=list(list(Qs.columns) + list(Ys.columns) + [C['name'], E['name']]))


if __name__ == '__main__':
    import os

    os.chdir('..')
    tau = 0.05
    for Data_Func in [dutch]:
        print('---------------------------------')
        print('Dataset: %s' % Data_Func.__name__)

        result = pd.DataFrame(data=np.zeros((6, 5)),
                              index=['MGraph', 'MData', 'Naive', 'LM', 'LPS', 'DI'],
                              columns=['Distance', 'n_T', 'chisqr', 'detect', 'non-empty'])

        data, C, E, Qs, Xs, Ys = Data_Func()
        data_new = MGraph(tau, data, C, E, Qs, Xs, Ys)

        # result.loc[Remove_func.__name__, :3] = Utility(data, data_new)
        avg, std, nonemptyset, largerset, smallerset, minvalue, maxvalue = Judge(data_new, C, E, Qs, tau)
        result.loc[MGraph.__name__, :] = Utility(data, data_new) + [largerset + smallerset, nonemptyset]

        print(result)
