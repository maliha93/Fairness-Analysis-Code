from __future__ import print_function, division
import pandas as pd
from PSEDD import PSEDD
from Model import AssignValue, FindConditionalProb, Combine, Model, ListAdd, ConditionalProb
from Utility import Utility


def NaiveRepair(model):
    Parent_E = model.graph.pred[model.E.name].keys()
    Parent_E_bar = Parent_E

    for pathset in model.Pi_Sets:
        cut_nodes = [path[-2] for path in pathset]
        Parent_E_bar = list(set(Parent_E_bar) - set(cut_nodes))

    def Cal_E_Dist():

        Dist = {}
        vals = Combine(data=model.df, V_max_list=ListAdd(Parent_E_bar, model.E.name), V_min_list=[], way='max')
        Given = vals.columns
        V = model.E.name
        vals['prob'] = vals.apply(
            lambda l: ConditionalProb(Target=V, target=l[V], Given=Given, given=l, df=model.df.copy()), axis=1)
        vals['hash'] = vals.apply(lambda l: '-'.join(l[Given]), axis=1)
        Dist[V] = vals.set_index('hash')
        return Dist

    E_Dist = Cal_E_Dist()

    df_from_CBN = []
    df_new = []
    num = model.df.__len__()

    QAE_set = sorted(model.graph.pred[model.E.name].keys() + [model.E.name])
    qae_set = model.Dist[model.E.name]
    qae_set = qae_set[qae_set[model.E.name] == model.E.pos]
    qae_set.loc[:, 'loc'] = range(qae_set.__len__())
    V_except_E = list(set(model.graph.nodes()) - {model.E.name})
    v_set = Combine(data=model.df, V_max_list=[model.E.name], V_min_list=V_except_E, way='mixed')
    v_Star_set = Combine(model.df, V_max_list=QAE_set, V_min_list=list(set(model.graph.nodes()) - set(QAE_set)),
                         way='mixed')
    v_Star_set = v_Star_set[v_Star_set[model.E.name] == model.E.pos]

    for i in range(v_Star_set.__len__()):
        v_Star = v_Star_set.iloc[i]

        gamma = 1.0
        v_Star = AssignValue(v_Star, model.E.name, model.E.pos)
        for V in V_except_E:
            Attrs = sorted(model.graph.pred[V].keys() + [V])
            gamma = gamma * FindConditionalProb(Target=V, target=v_Star[V], Given=Attrs, given=v_Star[Attrs],
                                                Dist=model.Dist)

        theta_new = FindConditionalProb(Target=model.E.name, target=v_Star[model.E.name],
                                        Given=ListAdd(Parent_E_bar, model.E.name),
                                        given=v_Star[ListAdd(Parent_E_bar, model.E.name)],
                                        Dist=E_Dist)
        theta = FindConditionalProb(Target=model.E.name, target=v_Star[model.E.name], Given=QAE_set,
                                    given=v_Star[QAE_set],
                                    Dist=model.Dist)

        Pr = gamma * theta_new
        amt = int(round(Pr * num))
        df_new.extend([list(v_Star)] * amt)

        Pr = gamma * theta
        amt = int(round(Pr * num))
        df_from_CBN.extend([list(v_Star)] * amt)

        gamma = 1.0
        v_Star = AssignValue(v_Star, model.E.name, model.E.neg)
        for V in V_except_E:
            Attrs = sorted(model.graph.pred[V].keys() + [V])
            gamma = gamma * FindConditionalProb(Target=V, target=v_Star[V], Given=Attrs, given=v_Star[Attrs],
                                                Dist=model.Dist)

        Pr = gamma * (1.0 - theta_new)
        amt = int(round(Pr * num))
        df_new.extend([list(v_Star)] * amt)

        Pr = gamma * (1.0 - theta)
        amt = int(round(Pr * num))
        df_from_CBN.extend([list(v_Star)] * amt)

    return pd.DataFrame(data=df_from_CBN, columns=v_Star_set.columns), pd.DataFrame(data=df_new,
                                                                                    columns=v_Star_set.columns)


if __name__ == '__main__':
    dir = '../data/'
    subdir = 'm0.01/'
    for dataset in ['adult', 'dutch']:
        print('Dataset:\t', dataset)
        RA = 'maritial'
        print("Redline Attribute: ", RA)
        R_set = [['-'], [RA]]
        model = Model(dataset=dataset, df=None, subdir=subdir)
        model.setRedlineAttrSet(R_set=R_set)
        df_from_CBN, df_repaired = NaiveRepair(model)
        repaired_model = Model(dataset=dataset, df=df_repaired, subdir=subdir)
        repaired_model.setRedlineAttrSet(R_set=R_set)
        PSEDD(repaired_model)
        Utility(model.df, repaired_model.df)
