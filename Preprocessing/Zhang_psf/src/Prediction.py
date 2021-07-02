import pandas as pd
from PSEDR import PSEDR
from PSEDD import PSEDD
from Model import Model
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn import tree, linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time, warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

def repair_data(dtset, ra):
    tau = 0.05
    epsilon = 0.001

    nrow = 0
    ncol = 0
    metric_df = pd.DataFrame(data=np.zeros((24, 5)))
    def Stack(l):
        global nrow
        global ncol
        le = l.__len__()
        metric_df.ix[nrow: nrow + le - 1, ncol] = l
        nrow = nrow + le
    dir = '../data/'
    subdir = 'm0.01/'
    RA = ra

    for dataset in [dtset]:
        print('Dataset:', dataset)
        print("Redline Attribute: ", RA)
        R_set = [['-'], RA]
        model = Model(dataset=dataset, df=None, subdir=subdir)
        df = model.df.reset_index(drop=True)
        #kf = KFold(n_splits=5, shuffle=True, random_state=2016)
        idx = np.arange(df.shape[0])
        train_index, test_index = train_test_split(idx, test_size=0.3, shuffle=False)
        model_from_training = Model(dataset=dataset, df=df.iloc[train_index], subdir=subdir)
        model_from_training.setRedlineAttrSet(R_set=R_set)
        df_repaired = PSEDR(model_from_training, tau - epsilon)
    return df_repaired

def Zhang(dataset, ra):
    
    if dataset == 'adult':
        df = repair_data('adult', ra)
        df.to_csv("../results_Zhang/adult_train_repaired.csv", index=False)
    elif dataset == 'compas':
        df = repair_data('compas', ra)
        df.to_csv("../results_Zhang/compas_train_repaired.csv", index=False)
    else:
        df = repair_data('german', ra)
        df.to_csv("../results_Zhang/german_train_repaired.csv", index=False)
        
    return df
        