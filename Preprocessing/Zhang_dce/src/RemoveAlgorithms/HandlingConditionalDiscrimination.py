from __future__ import division, print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from Basic import adult, dutch, testdata
from Utility import Utility
from Detection import Judge
from Basic import get_group


def Ranker(X, Y, Epos):
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

    # Data preprocessing techniques for classification without discrimination Page 15
    # ranking the objects according to their positive class probability
    Epos = le.fit_transform([Epos])[0]
    return prob[:, Epos]


def LM(df, Q, C, E):
    # Learn a ranker
    X = df.drop(E['name'], axis=1)
    Y = df[E['name']]
    df['prob'] = Ranker(X, Y, E['pos'])

    for name, group_q in df.groupby(Q):
        # Partion(X, e), q in our code is alternative of e in their paper
        groupbyC = group_q.groupby(C['name'])
        groupbyCE = group_q.groupby([C['name'], E['name']])

        p1 = get_group(groupbyCE, (C['pos'], E['pos'])).__len__() / get_group(groupbyC, C['pos']).__len__()
        p2 = get_group(groupbyCE, (C['neg'], E['pos'])).__len__() / get_group(groupbyC, C['neg']).__len__()
        p_star = (p1 + p2) / 2
        # switch if there is an inversion
        C['pos'], C['neg'] = [C['pos'], C['neg']] if p1 >= p2 else [C['neg'], C['pos']]

        for c, e_selected, e_target, ascending in [(C['pos'], E['pos'], E['neg'], True),
                                                   (C['neg'], E['neg'], E['pos'], False)]:
            # the promotion candidates (such as male) are sorted according to descending score
            # the demotion candidates (such as female) are sorted according to ascending score

            delta = int(round(get_group(groupbyC, c).__len__() * abs(p_star - p1)))

            group_changed = get_group(groupbyCE, (c, e_selected)).sort_values(by='prob', ascending=ascending)
            len_changed = min([group_changed.__len__(), delta])
            index_selected = group_changed.iloc[:len_changed, :].index
            df.loc[index_selected, E['name']] = e_target

    return df


def LPS(df, Q, C, E):
    # Learn a ranker
    X = df.drop(E['name'], axis=1)
    Y = df[E['name']]
    df['prob'] = Ranker(X, Y, E['pos'])

    for name, group_q in df.groupby(Q):
        # Partion(X, e), q in our code is alternative of e in their paper
        groupbyC = group_q.groupby(C['name'])
        groupbyCE = group_q.groupby([C['name'], E['name']])

        p1 = get_group(groupbyCE, (C['pos'], E['pos'])).__len__() / get_group(groupbyC, C['pos']).__len__()
        p2 = get_group(groupbyCE, (C['neg'], E['pos'])).__len__() / get_group(groupbyC, C['neg']).__len__()
        p_star = (p1 + p2) / 2
        # switch if there is an inversion
        C['pos'], C['neg'] = [C['pos'], C['neg']] if p1 >= p2 else [C['neg'], C['pos']]

        for c, e_deleted, e_duplicated, ascending in [(C['pos'], E['pos'], E['neg'], True),
                                                      (C['neg'], E['neg'], E['pos'], False)]:
            # the promotion candidates (such as male) are sorted according to descending score
            # the demotion candidates (such as female) are sorted according to ascending score

            delta = int(round(get_group(groupbyC, c).__len__() * abs(p_star - p1) / 2))

            group_deleted = get_group(groupbyCE, (c, e_deleted)).sort_values(by='prob', ascending=ascending)
            group_duplicated = get_group(groupbyCE, (c, e_duplicated)).sort_values(by='prob', ascending=ascending)
            len_changed = min([group_deleted.__len__(), group_duplicated.__len__(), delta])
            index_deleted = group_deleted.iloc[:len_changed, :].index
            index_duplicated = group_duplicated.iloc[:len_changed, :].index

            df = df.drop(index_deleted, axis=0)
            df = df.append(df.loc[index_duplicated, :])

    return df.reset_index(drop=True)


def Removeall(Remove_func, tau, data, C, E, Qs, Xs, Ys):
    condition = list(set(data.columns) - set([C['name'], E['name']]))
    df = data.copy()

    for attr in condition:
        df = Remove_func(df, attr, C, E)

    return df.drop(['prob'], axis=1)


if __name__ == '__main__':
    import os

    tau = 0.05
    os.chdir('..')
    for Data_Func in [adult, dutch]:
        result = pd.DataFrame(data=np.zeros((6, 5)),
                              index=['MGraph', 'MData', 'Naive', 'LM', 'LPS', 'DI'],
                              columns=['Distance', 'n_T', 'chisqr', 'detect', 'non-empty'])

        for Remove_func in [LPS]:
            print(Remove_func.__name__)
            data, C, E, Qs, Xs, Ys = Data_Func()
            data_new = Removeall(Remove_func, 0.05, data, C, E, Qs, Xs, Ys)
            result.loc[Remove_func.__name__, :3] = Utility(data, data_new)

            avg, std, nonemptyset, largerset, smallerset, minvalue, maxvalue = Judge(data_new, C, E, Qs)
            result.loc[Remove_func.__name__, 3:] = [largerset + smallerset, nonemptyset]

        print(result)
