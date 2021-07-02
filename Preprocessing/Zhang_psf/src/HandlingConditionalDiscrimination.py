import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from Model import Model
from Utility import Utility
from PSEDD import PSEDD


def Ranker(X, Y):
    # One-hot-encoding features
    for f in X.columns:
        X_dummy = pd.get_dummies(X[f], prefix=f)
        X = X.drop([f], axis=1)
        X = pd.concat((X, X_dummy), axis=1)

    # label encoder
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, Y)
    prob = clf.predict_proba(X)

    x = range(len(Y))
    return prob[x, Y]


def HCDetection(data, Q, C, E):
    try:
        P1 = (data[(data[C.name] == C.act) & (data[E.name] == E.pos)]).__len__() * 1.0 / (
            data[(data[C.name] == C.act)]).__len__()
    except:
        P1 = 0
    try:
        P2 = (data[(data[C.name] == C.bas) & (data[E.name] == E.pos)]).__len__() * 1.0 / (
            data[(data[C.name] == C.bas)]).__len__()
    except:
        P2 = 0
    Dall = P1 - P2

    for exp in data.drop([E.name, C.name], axis=1).columns:

        df = data[[C.name, exp, E.name]]
        Dexp = 0
        vals = pd.unique(df[exp])
        for v in vals:

            # P*(+|ei)
            try:  # P(+|ei,m)
                Peim = (data[(data[E.name] == E.pos) & (data[C.name] == C.act) & (
                    data[exp] == v)]).__len__() * 1.0 / (
                           data[(data[C.name] == C.act) & (data[exp] == v)]).__len__()
            except:
                Peim = 0

            try:  # P(+|ei,f)
                Peif = (data[(data[E.name] == E.pos) & (data[C.name] == C.bas) & (
                    data[exp] == v)]).__len__() * 1.0 / (
                           data[(data[C.name] == C.bas) & (data[exp] == v)]).__len__()
            except:
                Peif = 0

            Ps = 0.5 * (Peif + Peim)

            # P(ei|m)
            try:
                Peim = (data[(data[exp] == v) & (data[C.name] == C.act)]).__len__() * 1.0 / (
                    data[(data[C.name] == C.act)]).__len__()
            except:
                Peim = 0

            # P(ei|f)
            try:
                Peif = (data[(data[exp] == v) & (data[C.name] == C.bas)]).__len__() * 1.0 / (
                    data[(data[C.name] == C.bas)]).__len__()
            except:
                Peif = 0

            Dexp = Dexp + (Peim - Peif) * Ps

        print(exp, Dall - Dexp, Dall)


def Partition(df, con, q, C, E):
    Part = df.copy()
    Part = Part[(Part[con] == q)]
    # P*(+|ei)
    try:  # P(+|ei,m)
        Peim = (Part[(Part[E.name] == E.pos) & (Part[C.name] == C.act)]).__len__() * 1.0 / (
            Part[(Part[C.name] == C.act)]).__len__()
    except:
        Peim = 0

    try:  # P(+|ei,f)
        Peif = (Part[(Part[E.name] == E.pos) & (Part[C.name] == C.bas)]).__len__() * 1.0 / (
            Part[(Part[C.name] == C.bas)]).__len__()
    except:
        Peif = 0
    Ps = 0.5 * (Peif + Peim)
    return Part, Ps, Peim, Peif


def LocalMSG(data, Condition, C, E):
    df = data.copy()
    X = data.drop(E.name, axis=1)
    Y = data[E.name]
    df['prob'] = Ranker(X, Y)
    df2 = None
    qs = data[Condition].drop_duplicates()
    for q in qs:
        Part, Ps, Peim, Peif = Partition(df, Condition, q, C, E)
        if Peim > Ps:
            PartM = Part[Part[C.name] == C.act]
            delta = int(round(PartM.__len__() * abs(Peim - Ps)))
            PartM = PartM.sort_values(by='prob', ascending=1)
            j = 0
            for i in PartM.index:
                if PartM.loc[i, E.name] == E.pos:
                    PartM.loc[i, E.name] = E.neg
                    j = j + 1
                if j > delta:
                    break
            PartF = Part[Part[C.name] == C.bas]
            delta = int(round(PartF.__len__() * abs(Peif - Ps)))
            PartF = PartF.sort_values(by='prob', ascending=1)
            j = 0
            for i in PartF.index:
                if PartF.loc[i, E.name] == E.neg:
                    PartF.loc[i, E.name] = E.pos
                    j = j + 1
                if j > delta:
                    break
        else:
            PartM = Part[Part[C.name] == C.act]
            delta = int(round(PartM.__len__() * abs(Peim - Ps)))
            PartM = PartM.sort_values(by='prob', ascending=1)
            j = 0
            for i in PartM.index:
                if PartM.loc[i, E.name] == E.neg:
                    PartM.loc[i, E.name] = E.pos
                    j = j + 1
                if j > delta:
                    break

            PartF = Part[Part[C.name] == C.bas]
            delta = int(round(PartF.__len__() * abs(Peif - Ps)))
            PartF = PartF.sort_values(by='prob', ascending=1)
            j = 0
            for i in PartF.index:
                if PartF.loc[i, E.name] == E.pos:
                    PartF.loc[i, E.name] = E.neg
                    j = j + 1
                if j > delta:
                    break

        if df2 is None:
            df2 = pd.concat((PartM, PartF), axis=0, ignore_index=False)
        else:
            df2 = pd.concat((df2, PartM), axis=0, ignore_index=False)
            df2 = pd.concat((df2, PartF), axis=0, ignore_index=False)

    df2 = df2.drop('prob', axis=1)
    df2.index = range(len(df2))
    return df2


def LocalPS(data, Condition, C, E):
    df = data.copy()
    X = df.drop(E.name, axis=1)
    Y = df[E.name]
    df['prob'] = Ranker(X, Y)
    df2 = None

    qs = data[Condition].drop_duplicates()
    for q in qs:
        Part, Ps, Peim, Peif = Partition(df, Condition, q, C, E)
        if Peim > Ps:
            PartM = Part[Part[C.name] == C.act]
            delta = int(round(PartM.__len__() * abs(Peim - Ps) / 2))
            PartM = PartM.sort_values(by='prob', ascending=1)
            j = 0
            for i in PartM.index:
                if PartM.loc[i, E.name] == E.pos:
                    PartM = PartM.drop(i)
                    j = j + 1
                if j >= delta:
                    break
            j = 0
            for i in PartM.index:
                if PartM.loc[i, E.name] == E.neg:
                    PartM = PartM.append(PartM.loc[i])
                    j = j + 1
                if j >= delta:
                    break

            PartF = Part[Part[C.name] == C.bas]
            delta = int(round(PartF.__len__() * abs(Peif - Ps) / 2))
            PartF = PartF.sort_values(by='prob', ascending=1)
            j = 0
            for i in PartF.index:
                if PartF.loc[i, E.name] == E.neg:
                    PartF = PartF.drop(i)
                j = j + 1
                if j >= delta:
                    break
            j = 0
            for i in PartF.index:
                if PartF.loc[i, E.name] == E.pos:
                    PartF = PartF.append(PartF.loc[i])
                j = j + 1
                if j >= delta:
                    break
        else:
            PartM = Part[Part[C.name] == C.act]
            delta = int(round(PartM.__len__() * abs(Peim - Ps) / 2))
            PartM = PartM.sort_values(by='prob', ascending=1)
            j = 0
            for i in PartM.index:
                if PartM.loc[i, E.name] == E.neg:
                    PartM = PartM.drop(i)
                    j = j + 1
                if j >= delta:
                    break

            j = 0
            for i in PartM.index:
                if PartM.loc[i, E.name] == E.pos:
                    PartM = PartM.append(PartM.loc[i])
                    j = j + 1
                if j >= delta:
                    break

            PartF = Part[Part[C.name] == C.bas]
            delta = int(round(PartF.__len__() * abs(Peif - Ps) / 2))
            PartF = PartF.sort_values(by='prob', ascending=1)
            j = 0
            for i in PartF.index:
                if PartF.loc[i, E.name] == E.pos:
                    PartF = PartF.drop(i)
                j = j + 1
                if j >= delta:
                    break

            j = 0
            for i in PartF.index:
                if PartF.loc[i, E.name] == E.neg:
                    PartF = PartF.append(PartF.loc[i])
                j = j + 1
                if j >= delta:
                    break
        if df2 is None:
            df2 = pd.concat((PartM, PartF), axis=0, ignore_index=True)
        else:
            df2 = pd.concat((df2, PartM), axis=0, ignore_index=True)
            df2 = pd.concat((df2, PartF), axis=0, ignore_index=True)

    df2 = df2.drop('prob', axis=1)
    df2.index = range(len(df2))
    return df2


def Removeall(model):
    df_list = []
    for Remove_func in [LocalMSG, LocalPS]:
        condition = list(set(model.df.columns) - set([model.C.name, model.E.name, 'maritial']))
        df_new = model.df.copy()
        for attr in condition:
            df_new = Remove_func(df_new, attr, model.C, model.E)

        df_list.append(df_new)
    return df_list


def LMSG(model, tau):
    condition = list(set(model.df.columns) - set([model.C.name, model.E.name, 'maritial']))
    df_new = model.df.copy()
    for attr in condition:
        df_new = LocalMSG(df_new, attr, model.C, model.E)
    return df_new


def LPS(model, tau):
    condition = list(set(model.df.columns) - set([model.C.name, model.E.name, 'maritial']))
    df_new = model.df.copy()
    for attr in condition:
        df_new = LocalPS(df_new, attr, model.C, model.E)
    return df_new


if __name__ == '__main__':
    tau = 0.05

    dir = '../data/'
    subdir = 'm0.01/'

    for dataset in ['adult', 'dutch']:
        print('Dataset:\t', dataset)
        RA = 'maritial'
        print("Redline Attribute: ", RA)

        model = Model(dataset=dataset, df=None, subdir=subdir)

        R_set_list = [[['-'], [RA]]]  ##### [[['-']], [[RA]], [['-'], [RA]]]
        for R_set in R_set_list:
            model.setRedlineAttrSet(R_set=R_set)
            df_LocalMSG, df_LocalPS = Removeall(model)

            MSG_repaired_model = Model(dataset=dataset, df=df_LocalMSG, subdir=subdir)
            MSG_repaired_model.setRedlineAttrSet(R_set=R_set)

            PSE(MSG_repaired_model)
            Utility(model.df, MSG_repaired_model.df)

            PS_repaired_model = Model(dataset=dataset, df=df_LocalPS, subdir=subdir)
            PS_repaired_model.setRedlineAttrSet(R_set=R_set)

            PSE(PS_repaired_model)
            Utility(model.df, PS_repaired_model.df)
