from __future__ import division, print_function
import pandas as pd
import numpy as np
import math
from Basic import get_group


def Utility(data, data_new):
    df = pd.concat((data, data_new), axis=0)
    df = df.drop_duplicates()
    df.index = range(df.__len__())

    L1 = 0.0
    L2 = 0.0
    Chisqr = 0.0

    groupby_old = data.groupby(list(df.columns))
    groupby_new = data_new.groupby(list(df.columns))

    for i in df.index:
        len_new = get_group(groupby_new, tuple(df.loc[i])).__len__()
        len_old = get_group(groupby_old, tuple(df.loc[i])).__len__()
        diff = len_new - len_old
        L1 = L1 + abs(diff)
        L2 = L2 + math.pow(diff, 2)
        # Chisqr = Chisqr + pow(diff, 2) / (len_new if len_new > 0 else len_old)
        Chisqr = Chisqr + ((pow(diff, 2) / len_new) if len_new >= 5 else 0.0)

    return [math.sqrt(L2) / data.__len__(), L1, Chisqr]


if __name__ == '__main__':
    df = pd.DataFrame(np.array([
        [1, 1, 1],
        [1, 2, 1],
        [1, 2, 1]]),
        columns=['A', 'B', 'C']
    )

    df_new = pd.DataFrame(np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 1],
        [3, 3, 3]]),
        columns=['A', 'B', 'C']
    )
    print(Utility(df, df_new))
