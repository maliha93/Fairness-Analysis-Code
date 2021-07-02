from __future__ import print_function, division
from Model import AssignValue, FindConditionalProb, Model


def PSEDD(model):
    v_Star_set = model.df.drop([model.C.name, model.E.name], axis=1).drop_duplicates()
    v_Star_set[model.C.name] = model.C.bas
    v_Star_set[model.E.name] = model.E.pos
    v_Star_set.index = range(v_Star_set.__len__())

    PSE_result = []

    for j, S_set_pi in enumerate(model.Succ_sets):
        S_set_npi = list(set(model.graph.succ[model.C.name]) - set(S_set_pi))
        pse_pi, pse_npi, pse_pi_, pse_npi_ = 0.0, 0.0, 0.0, 0.0

        for i in v_Star_set.index:
            v_Star = v_Star_set.iloc[i].copy()
            pnp, pnn, pmp, pmn, po = 1.0, 1.0, 1.0, 1.0, 1.0
            for N in S_set_npi:
                Parents = model.graph.pred[N].keys()
                pnp = pnp * FindConditionalProb(Target=N, target=v_Star[N], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.act),
                                                Dist=model.Dist)
                pnn = pnn * FindConditionalProb(Target=N, target=v_Star[N], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.bas),
                                                Dist=model.Dist)

            for M in S_set_pi:
                Parents = model.graph.pred[M].keys()
                pmp = pmp * FindConditionalProb(Target=M, target=v_Star[M], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.act),
                                                Dist=model.Dist)
                pmn = pmn * FindConditionalProb(Target=M, target=v_Star[M], Given=Parents,
                                                given=AssignValue(v_Star[Parents], model.C.name, model.C.bas),
                                                Dist=model.Dist)

            for O in model.O_set:
                Parents = model.graph.pred[O].keys()
                po = po * FindConditionalProb(Target=O, target=v_Star[O], Given=Parents, given=v_Star[Parents],
                                              Dist=model.Dist)

            pse_pi = pse_pi + pnn * (pmp - pmn) * po
            pse_npi = pse_npi + (pnp - pnn) * pmp * po

            pse_pi_ = pse_pi_ + pnp * (pmn - pmp) * po
            pse_npi_ = pse_npi_ + (pnn - pnp) * pmn * po

        PSE_result.append(round(pse_pi, 4))
    return PSE_result


if __name__ == '__main__':
    dir = '../data/'
    subdir = 'm0.01/'

    dataset = 'german'
    print('Dataset:\t', dataset)
    RA = 'Age'
    print("Redline Attribute: ", RA)

    model_adult = Model(dataset=dataset, df=None, subdir=subdir)

    R_set = [['-']]
    model_adult.setRedlineAttrSet(R_set=R_set)
    print('Direct:', PSEDD(model_adult)[0])

    R_set = [[RA]]
    model_adult.setRedlineAttrSet(R_set=R_set)
    print('Indirect:', PSEDD(model_adult)[0])
