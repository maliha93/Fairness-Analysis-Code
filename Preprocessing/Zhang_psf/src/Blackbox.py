from __future__ import division
import csv
from blackboxrepairers.GeneralRepairer import Repairer
from Utility import Utility
import pandas as pd
from PSEDD import PSEDD
from Model import Shuffle, Model


def remove(fin):
    with open(fin) as f:
        data = [line for line in csv.reader(f)]
        headers = data.pop(0)
        cols = [[row[i] for row in data] for i, col in enumerate(headers)]

        data = [[col[j] for col in cols] for j in xrange(len(data))]

    # Calculte the indices to repair by and to ignore.
    try:
        index_to_repair = headers.index('sex')
    except ValueError as e:
        raise Exception("Response header '{}' was not found in the following headers: {}".format('sex', headers))

    repairer = Repairer(data, index_to_repair, repair_level=1.0, features_to_ignore=[])

    repaired = repairer.repair(data)
    df = pd.DataFrame(data=repaired, columns=headers)

    return df


def blackboxrepair(df0, Cname, Ename, repair_level):
    df = df0.copy()
    index_to_repair = df.columns.get_loc(Cname)
    data = df.values.tolist()
    repairer = Repairer(data, index_to_repair, repair_level=repair_level, features_to_ignore=[])
    repaired = repairer.repair(data)
    df_new = pd.DataFrame(data=repaired, columns=df.columns)

    df_new[Cname] = (list(df[Cname]))
    df_new[Ename] = list(df[Ename])

    return df_new


from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


def BER(X, Y):
    # One-hot-encoding features
    for f in X.columns:
        X_dummy = pd.get_dummies(X[f], prefix=f)
        X = X.drop([f], axis=1)
        X = pd.concat((X, X_dummy), axis=1)

    # label encoder
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    clf = SVC()
    clf.fit(X, Y)
    pred = clf.predict(X)

    vals = pd.unique(Y)

    P = pd.DataFrame(data=Y, columns=['true'])
    P['pred'] = pred

    prob = 0.0

    for v in vals:
        prob = prob + len(P[(P['pred'] == v) & (P['true'] != v)]) / len(P[P['true'] != v])

    return prob / 2


if __name__ == '__main__':
    tau = 0.0

    dir = '../data/'
    subdir = 'm0.01/'

    dataset = 'adult'
    print('Dataset:\t', dataset)
    RA = 'maritial'
    print("Redline Attribute: ", RA)

    model = Model(dataset=dataset, df=None, subdir=subdir)
    df_BB_repaired = blackboxrepair(df0=model.df, Cname=model.C.name, Ename=model.E.name,
                                    repair_level=0.725)
    repaired_model = Model(dataset=dataset, df=df_BB_repaired, subdir=subdir)

    R_set_list = [[['-'], [RA]]]
    for R_set in R_set_list:
        model.setRedlineAttrSet(R_set=R_set)
        repaired_model.setRedlineAttrSet(R_set=R_set)
        PSEDD(repaired_model)

    repaired_model.df[model.C.name] = Shuffle(repaired_model.df[model.C.name])
    Utility(model.df, repaired_model.df)

    dataset = 'dutch'
    print('Dataset:\t', dataset)
    RA = 'maritial'
    print("Redline Attribute: ", RA)

    model = Model(dataset=dataset, df=None, subdir=subdir)
    df_BB_repaired = blackboxrepair(df0=model.df, Cname=model.C.name, Ename=model.E.name,
                                    repair_level=0.725)
    repaired_model = Model(dataset=dataset, df=df_BB_repaired, subdir=subdir)

    R_set_list = [[['-'], [RA]]]
    for R_set in R_set_list:
        model.setRedlineAttrSet(R_set=R_set)
        repaired_model.setRedlineAttrSet(R_set=R_set)
        PSEDD(repaired_model)

    repaired_model.df[model.C.name] = Shuffle(repaired_model.df[model.C.name])
    Utility(model.df, repaired_model.df)
