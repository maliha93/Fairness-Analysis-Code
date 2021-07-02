from __future__ import print_function, division
import pandas as pd
import numpy as np
from PSEDD import PSEDD
from cvxopt import matrix, solvers
from Model import AssignValue, FindConditionalProb, ListSubstract, Combine, Model
from Utility import Utility


def cases(v_Star_set, S_set_pi, S_set_npi, QAE_set, qae_set, model):
    X_dim = qae_set.__len__()
    Coef = np.zeros((4, X_dim))
    if model.E.name in model.O_set:  ##### case 1
        ##### print('case 1')
        for i in v_Star_set.index:
            v_Star = v_Star_set.iloc[i].copy()
            pnp, pnn, pmp, pmn, po = 1.0, 1.0, 1.0, 1.0, 1.0
            for N in ListSubstract(S_set_npi, model.E.name):
                Parents = model.graph.pred[N].keys()
                pnp = pnp * FindConditionalProb(Target=N, target=v_Star[N], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.act),
                                                Dist=model.Dist)
                pnn = pnn * FindConditionalProb(Target=N, target=v_Star[N], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.bas),
                                                Dist=model.Dist)

            for M in ListSubstract(S_set_pi, model.E.name):
                Parents = model.graph.pred[M].keys()
                pmp = pmp * FindConditionalProb(Target=M, target=v_Star[M], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.act),
                                                Dist=model.Dist)
                pmn = pmn * FindConditionalProb(Target=M, target=v_Star[M], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.bas),
                                                Dist=model.Dist)

            v_Star = v_Star_set.iloc[i]
            for O in ListSubstract(model.O_set, model.E.name):
                Parents = model.graph.pred[O].keys()
                po = po * FindConditionalProb(Target=O, target=v_Star[O], Given=Parents, given=v_Star[Parents],
                                              Dist=model.Dist)

            v_Star = v_Star_set.iloc[i]
            qae = v_Star[QAE_set]
            loc = qae_set.loc['-'.join(qae)]['loc']
            Coef[0, loc] = Coef[0, loc] + pnn * (pmp - pmn) * po
            Coef[1, loc] = Coef[1, loc] + (pnp - pnn) * pmp * po
            Coef[2, loc] = Coef[0, loc] + pnp * (pmn - pmp) * po
            Coef[3, loc] = Coef[1, loc] + (pnn - pnp) * pmn * po

    if model.E.name in S_set_npi:  ##### case 2
        ##### print('case 2')
        for i in v_Star_set.index:
            v_Star = v_Star_set.iloc[i].copy()
            pnp, pnn, pmp, pmn, po = 1.0, 1.0, 1.0, 1.0, 1.0
            for N in ListSubstract(S_set_npi, model.E.name):
                Parents = model.graph.pred[N].keys()
                pnp = pnp * FindConditionalProb(Target=N, target=v_Star[N], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.act),
                                                Dist=model.Dist)
                pnn = pnn * FindConditionalProb(Target=N, target=v_Star[N], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.bas),
                                                Dist=model.Dist)

            for M in ListSubstract(S_set_pi, model.E.name):
                Parents = model.graph.pred[M].keys()
                pmp = pmp * FindConditionalProb(Target=M, target=v_Star[M], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.act),
                                                Dist=model.Dist)
                pmn = pmn * FindConditionalProb(Target=M, target=v_Star[M], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.bas),
                                                Dist=model.Dist)

            v_Star = v_Star_set.iloc[i]
            for O in ListSubstract(model.O_set, model.E.name):
                Parents = model.graph.pred[O].keys()
                po = po * FindConditionalProb(Target=O, target=v_Star[O], Given=Parents, given=v_Star[Parents],
                                              Dist=model.Dist)

            qae = AssignValue(v_Star[QAE_set], model.C.name, model.C.act)
            loc = qae_set.loc['-'.join(qae)]['loc']
            Coef[1, loc] = Coef[1, loc] + po * pmp * pnp
            Coef[2, loc] = Coef[2, loc] + po * (pmn - pmp) * pnp
            Coef[3, loc] = Coef[3, loc] - po * pmn * pnp

            qae = AssignValue(v_Star[QAE_set], model.C.name, model.C.bas)
            loc = qae_set.loc['-'.join(qae)]['loc']
            Coef[0, loc] = Coef[0, loc] + po * (pmp - pmn) * pnn
            Coef[1, loc] = Coef[1, loc] - po * pmp * pnn
            Coef[3, loc] = Coef[3, loc] + po * pmn * pnn

    if model.E.name in S_set_pi:  ##### case 3
        ##### print('case 3')
        for i in v_Star_set.index:
            v_Star = v_Star_set.iloc[i].copy()
            pnp, pnn, pmp, pmn, po = 1.0, 1.0, 1.0, 1.0, 1.0
            for N in ListSubstract(S_set_npi, model.E.name):
                Parents = model.graph.pred[N].keys()
                pnp = pnp * FindConditionalProb(Target=N, target=v_Star[N], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.act),
                                                Dist=model.Dist)
                pnn = pnn * FindConditionalProb(Target=N, target=v_Star[N], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.bas),
                                                Dist=model.Dist)
            for O in ListSubstract(model.O_set, model.E.name):
                Parents = model.graph.pred[O].keys()
                po = po * FindConditionalProb(Target=O, target=v_Star[O], Given=Parents, given=v_Star[Parents],
                                              Dist=model.Dist)

            ##### pm = lambda
            for M in ListSubstract(S_set_pi, model.E.name):
                Parents = model.graph.pred[M].keys()
                pmp = pmp * FindConditionalProb(Target=M, target=v_Star[M], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.act),
                                                Dist=model.Dist)
                pmn = pmn * FindConditionalProb(Target=M, target=v_Star[M], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.bas),
                                                Dist=model.Dist)

            qae = AssignValue(v_Star[QAE_set], model.C.name, model.C.act)
            loc = qae_set.loc['-'.join(qae)]['loc']
            Coef[0, loc] = Coef[0, loc] + pnn * pmp * po
            Coef[2, loc] = Coef[2, loc] - pnp * pmp * po
            Coef[1, loc] = Coef[1, loc] + (pnp - pnn) * pmp * po

            qae = AssignValue(v_Star[QAE_set], model.C.name, model.C.bas)
            loc = qae_set.loc['-'.join(qae)]['loc']
            Coef[0, loc] = Coef[0, loc] - pnn * pmn * po
            Coef[2, loc] = Coef[2, loc] + pnp * pmn * po
            Coef[3, loc] = Coef[3, loc] + (pnn - pnp) * pmn * po

    return Coef[[0, 2], :]


def PSEDR(model, tau):
    QAE_set = sorted(list(model.graph.pred[model.E.name].keys()) + [model.E.name])
    qae_set = model.Dist[model.E.name]
    qae_set = qae_set[qae_set[model.E.name] == model.E.pos]
    qae_set.loc[:, 'loc'] = range(qae_set.__len__())

    V_except_E = list(set(model.graph.nodes()) - {model.E.name})
    # v_set = Comb(data=model.df, V_max_list=[model.E.name], V_min_list=V_except_E, way='mixed')
    v_Star_set = Combine(model.df, V_max_list=QAE_set, V_min_list=list(set(model.graph.nodes()) - set(QAE_set)),
                         way='mixed')
    v_Star_set = v_Star_set[v_Star_set[model.E.name] == model.E.pos]
    v_Star_set.index = range(v_Star_set.__len__())

    X_dim = qae_set.__len__()
    P_mat = np.zeros((X_dim, X_dim))
    q_vec = np.zeros(X_dim)
    X_vec = np.zeros(X_dim)
    ##### Gamma_vec = np.zeros(X_dim)
    df_from_CBN = []
    df_repaired = []
    num = model.df.__len__()

    ##### objective: mimimize the distance of two distribution
    for i in range(v_Star_set.__len__()):
        v_Star = v_Star_set.iloc[i].copy()

        qae = v_Star[QAE_set]
        loc = qae_set.loc['-'.join(qae)]['loc']

        v_Star = AssignValue(v_Star, model.E.name, model.E.pos)
        gamma = 1.0
        for V in V_except_E:
            Attrs = sorted(list(model.graph.pred[V].keys()) + [V])
            gamma = gamma * FindConditionalProb(Target=V, target=v_Star[V], Given=Attrs, given=v_Star[Attrs],
                                                Dist=model.Dist)

        theta = FindConditionalProb(Target=model.E.name, target=v_Star[model.E.name], Given=QAE_set,
                                    given=v_Star[QAE_set],
                                    Dist=model.Dist)

        P_mat[loc, loc] = P_mat[loc, loc] + 2 * gamma ** 2
        q_vec[loc] = q_vec[loc] - 2 * theta * gamma ** 2
        X_vec[loc] = theta
        ##### Gamma_vec[loc] = Gamma_vec[loc] + gamma

        v_Star = AssignValue(v_Star, model.E.name, model.E.neg)
        gamma = 1.0
        for V in V_except_E:
            Attrs = sorted(list(model.graph.pred[V].keys()) + [V])
            gamma = gamma * FindConditionalProb(Target=V, target=v_Star[V], Given=Attrs, given=v_Star[Attrs],
                                                Dist=model.Dist)

        P_mat[loc, loc] = P_mat[loc, loc] + 2 * gamma ** 2
        q_vec[loc] = q_vec[loc] - 2 * theta * gamma ** 2
        X_vec[loc] = theta
        ##### Gamma_vec[loc] = Gamma_vec[loc] + gamma

    ##### subject to PSE <= tau and 0 <= Pr <= 1
    v_Star_set = model.df.drop([model.C.name, model.E.name], axis=1).drop_duplicates()
    v_Star_set[model.C.name] = model.C.bas
    v_Star_set[model.E.name] = model.E.pos
    v_Star_set.index = range(v_Star_set.__len__())

    G_size = 2 * X_dim + len(model.Succ_sets) * 2
    G_mat = np.zeros((G_size, X_dim))
    h_vec = np.zeros(G_size)
    G_mat[:X_dim, :] = np.identity(X_dim)  ##### Pr <= 1
    G_mat[X_dim: 2 * X_dim, :] = -1 * np.identity(X_dim)  ##### Pr <= 1
    h_vec[: X_dim] = 1  ##### Pr <= 1
    h_vec[X_dim: 2 * X_dim] = 0  ##### Pr >= 0
    h_vec[2 * X_dim:] = tau  ##### PSE ,= tau
    ##### PSE
    for j, S_set_pi in enumerate(model.Succ_sets):
        S_set_npi = list(set(model.graph.succ[model.C.name]) - set(S_set_pi))
        G_mat[2 * X_dim + 2 * j: 2 * X_dim + 2 * j + 2, :] = cases(v_Star_set, S_set_pi, S_set_npi, QAE_set, qae_set,
                                                                   model)

    ##### solver
    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(P_mat), matrix(q_vec), matrix(G_mat), matrix(h_vec))
    pred = np.array(sol['x']).reshape(X_dim)

    ##### print solution
    # for j in range(len(model.Succ_sets)):
    #     Coef = G_mat[2 * X_dim + 2 * j: 2 * X_dim + 2 * j + 2, :]
    #     print(np.dot(Coef, X_vec))
    #     print(np.dot(Coef, pred))

    ##### regenerate data from modified graph
    QAE_set = sorted(list(model.graph.pred[model.E.name].keys()) + [model.E.name])
    qae_set = model.Dist[model.E.name]
    qae_set = qae_set[qae_set[model.E.name] == model.E.pos]
    qae_set.loc[:, 'loc'] = range(qae_set.__len__())

    V_except_E = list(set(model.graph.nodes()) - {model.E.name})
    # v_set = Comb(data=model.df, V_max_list=[model.E.name], V_min_list=V_except_E, way='mixed')
    v_Star_set = Combine(model.df, V_max_list=QAE_set, V_min_list=list(set(model.graph.nodes()) - set(QAE_set)),
                         way='mixed')
    v_Star_set = v_Star_set[v_Star_set[model.E.name] == model.E.pos]
    v_Star_set.index = range(v_Star_set.__len__())

    for i in range(v_Star_set.__len__()):
        v_Star = v_Star_set.iloc[i].copy()

        qae = v_Star[QAE_set]
        loc = qae_set.loc['-'.join(qae)]['loc']

        gamma = 1.0
        v_Star = AssignValue(v_Star, model.E.name, model.E.pos)
        for V in V_except_E:
            Attrs = sorted(list(model.graph.pred[V].keys()) + [V])
            gamma = gamma * FindConditionalProb(Target=V, target=v_Star[V], Given=Attrs, given=v_Star[Attrs],
                                                Dist=model.Dist)

        Pr = gamma * pred[loc]
        amt = int(round(Pr * num))
        df_repaired.extend([list(v_Star)] * amt)

        Pr = gamma * X_vec[loc]
        amt = int(round(Pr * num))
        df_from_CBN.extend([list(v_Star)] * amt)

        gamma = 1.0
        v_Star = AssignValue(v_Star, model.E.name, model.E.neg)
        for V in V_except_E:
            Attrs = sorted(list(model.graph.pred[V].keys()) + [V])
            gamma = gamma * FindConditionalProb(Target=V, target=v_Star[V], Given=Attrs, given=v_Star[Attrs],
                                                Dist=model.Dist)

        Pr = gamma * (1.0 - pred[loc])
        amt = int(round(Pr * num))
        df_repaired.extend([list(v_Star)] * amt)

        Pr = gamma * (1.0 - X_vec[loc])
        amt = int(round(Pr * num))
        df_from_CBN.extend([list(v_Star)] * amt)

    # return pd.DataFrame(data=df_from_CBN, columns=v_Star_set.columns), pd.DataFrame(data=df_repaired, columns=v_Star_set.columns)
    return pd.DataFrame(data=df_repaired, columns=v_Star_set.columns)


if __name__ == '__main__':
    tau = 0.05
    epsilon = 0.001

    dir = '../data/'
    subdir = 'm0.01/'

    dataset = 'adult'
    print('Dataset:\t', dataset)
    RA = 'maritial'
    print("Redline Attribute: ", RA)
    epsilon = 0.001
    model = Model(dataset=dataset, df=None, subdir=subdir)

    R_set_list = [[['-'], [RA]]]  ##### [[['-']], [[RA]], [['-'], [RA]]]
    for R_set in R_set_list:
        model.setRedlineAttrSet(R_set=R_set)

        for tau in [0.025, 0.05, 0.075, 0.1]:
            df_from_CBN, df_repaired = Repair(model, tau - epsilon)

            repaired_model = Model(dataset=dataset, df=df_repaired, subdir=subdir)
            repaired_model.setRedlineAttrSet(R_set=R_set)

            print(PSEDD(repaired_model))
            print(Utility(model.df, repaired_model.df))
            #######################
            # dataset = 'dutch'
            # print('Dataset:\t', dataset)
            # RA = 'maritial'
            # print("Redline Attribute: ", RA)
            # epsilon = 0.03
            #
            # model = Model(dataset=dataset, df=None, subdir=subdir)
            #
            # R_set_list = [[['-'], [RA]]]  ##### [[['-']], [[RA]], [['-'], [RA]]]
            # for R_set in R_set_list:
            #     model.setRedlineAttrSet(R_set=R_set)
            #     df_from_CBN, df_repaired = Repair(model, tau-epsilon)
            #
            #     repaired_model = Model(dataset=dataset, df=df_repaired, subdir=subdir)
            #     repaired_model.setRedlineAttrSet(R_set=R_set)
            #
            #     PSE(repaired_model)
            #     Utility(model.df, repaired_model.df)
