from __future__ import division, print_function
from blackboxrepairers.GeneralRepairer import Repairer
from Utility import Utility
import pandas as pd
from Basic import adult, dutch
from Detection import Judge


def blackboxrepair(df, Cname, Ename, repair_level):
    df_backup = df.copy()
    index_to_repair = df.columns.get_loc(Cname)
    data = df.values.tolist()
    repairer = Repairer(data, index_to_repair, repair_level=repair_level, features_to_ignore=[])
    repaired = repairer.repair(data)

    df_new = pd.DataFrame(data=repaired, columns=df.columns)
    df_new[Cname] = list(df_backup[Cname])
    df_new[Ename] = list(df_backup[Ename])
    return df_new


def DI(tau, data, C, E, Qs, Xs, Ys):
    return blackboxrepair(df=data, Cname=C['name'], Ename=E['name'], repair_level=1)


if __name__ == '__main__':

    for Data_Func in [adult, dutch]:
        data, C, E, Qs, Xs, Ys = Data_Func()
        data_new = blackboxrepair(df=data, Cname=C['name'], Ename=E['name'], repair_level=1)
        Utility(data, data_new)
        Judge(data_new, C, E, Qs)
        # EvaluateKF(Remove_func, tau, data, C, E, Qs, Xs, Ys)
        print('==========')
