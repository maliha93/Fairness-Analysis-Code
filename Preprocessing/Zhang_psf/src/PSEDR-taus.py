from Utility import Utility
import pandas as pd
import numpy as np
from PSEDR import PSEDR
from PSEDD import PSEDD
from Model import Model
from sklearn.cross_validation import KFold
import time, warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    epsilon = 0.001
    dir = '../data/'
    subdir = 'm0.01/'
    RA = 'maritial'
    stack = []

    for dataset in ['adult']:
        print('Dataset:', dataset)
        print("Redline Attribute: ", RA)
        R_set = [['-'], [RA]]
        model = Model(dataset=dataset, df=None, subdir=subdir)
        df = model.df.sample(frac=1, random_state=123).reset_index(drop=True)
        model.setRedlineAttrSet(R_set=R_set)

        for tau in [0.025, 0.05, 0.075, 0.1]:
            df_repaired = PSEDR(model, tau - epsilon)
            repaired_model = Model(dataset=dataset, df=df_repaired, subdir=subdir)
            repaired_model.setRedlineAttrSet(R_set=R_set)
            stack.extend(PSEDD(repaired_model))
            stack.extend(Utility(df, df_repaired))

        vals = np.array(stack).reshape([4, 3]).transpose()
        print(pd.DataFrame(data=vals, columns=['0.025', '0.05', '0.075', '0.1']))
