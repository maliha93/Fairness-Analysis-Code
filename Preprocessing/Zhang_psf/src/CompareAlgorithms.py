'''
compare the performance of 4 types of algorithms over all the data
result:
PSE-DD	        DI	            MSG	            PS
0.012496943	    0.049948263	    -0.142129995	-0.14151063
0.049332336	    0.002399076	    0.28840677	    0.173907243
10384.62059	    49644.31534	    19241.89234	    12917.50538
'''

from Utility import Utility
import pandas as pd
from PSEDR import PSEDR
from PSEDD import PSEDD
from Model import Model, Shuffle
from Blackbox import blackboxrepair
from HandlingConditionalDiscrimination import LMSG, LPS
import numpy as np
import time, warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dir = '../data/'
    subdir = 'm0.01/'
    RA = 'maritial'
    tau = 0.05
    epsilon = 0.001
    stack = []

    for dataset in ['adult']:
        print('Dataset:', dataset)
        print("Redline Attribute: ", RA)
        R_set = [['-'], [RA]]
        model = Model(dataset=dataset, df=None, subdir=subdir)
        model.setRedlineAttrSet(R_set=R_set)
        df = model.df.copy()

        for RemoveAlgo in [PSEDR, LMSG, LPS]:
            df_repaired = RemoveAlgo(model, tau - epsilon)
            model_repaired = Model(dataset=dataset, df=df_repaired, subdir=subdir)
            model_repaired.setRedlineAttrSet(R_set=R_set)
            stack.extend(PSEDD(model_repaired))
            stack.extend(Utility(df, df_repaired))

        df_repaired = blackboxrepair(model.df, model.C.name, model.E.name, repair_level=0.725)
        # detect indirect discrimination
        BB_repaired_model = Model(dataset=dataset, df=df_repaired, subdir=subdir)
        BB_repaired_model.setRedlineAttrSet(R_set=[R_set[1]])
        stack.extend(PSEDD(BB_repaired_model))
        # detect direct discrimination
        df_repaired[model.C.name] = Shuffle(df_repaired[model.C.name])
        BB_repaired_model = Model(dataset=dataset, df=df_repaired, subdir=subdir)
        BB_repaired_model.setRedlineAttrSet(R_set=[R_set[0]])
        stack.extend(PSEDD(BB_repaired_model))
        stack.extend(Utility(df, df_repaired))

    stack[9:11] = stack[9:11]
    vals = np.array(stack).reshape([4, 3]).transpose()
    metric_df = pd.DataFrame(data=vals, columns=['PSE-DR', 'LMSG', 'LPS', 'DI'])
    print(metric_df[['PSE-DR', 'DI', 'LMSG', 'LPS']])
