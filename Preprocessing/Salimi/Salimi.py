import os
import sys
import warnings
import time
warnings.filterwarnings('ignore')
capuchin_path = ""
sys.path.append(capuchin_path)

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
print(__doc__)
from Core.indep_repair import Repair
from Core.Log_Reg_Classifier import *
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
from Modules.MatrixOprations.contin_table import *
from utils.read_data import read_from_csv

def Adult(rep):   
    data_dir = capuchin_path
    data = read_from_csv(os.path.join(data_dir, "dataset/adult_bin_train.csv"))
    output_dir = os.path.join(data_dir+"results_adult/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    D_features = ['sex', 'race', 'native_country']
    Y_features = ['income']
    X_features = ['age', 'marital_status', 'hours_per_week','occupation','workclass','edu_level',\
                  'relationship']

    features=D_features+Y_features+X_features
    D = [D_features, Y_features, X_features]
    data=data[features]
    #print(data)
    k=1
    data_split(data, 'income', output_dir, k=k, test_size=0.01)
    indeps = [D]
    for method in [rep]: #'sat','naive','IC','MF',
        inf = Info(data)
        for indep in indeps:

            #print(indep)
            X = indep[0]
            Y = indep[1]
            Z = indep[2]
            mi = inf.CMI(X, Y, Z)
            
        rep = Repair()
        path1 = os.path.join(output_dir,'train_')
        if method == 'sat':
            rep.from_file_indep_repair(path1, X, Y, Z, method=method, n_parti=100,
                                       k=k, sample_frac=1, insert_ratio='hard', conf_weight=1)
            rep.from_file_indep_repair(path1, X, Y, Z, method=method, n_parti=100,
                                       k=k, sample_frac=1, insert_ratio='soft', conf_weight=1)
        else:
            rep.from_file_indep_repair(path1, X, Y, Z, method=method, n_parti=100,
                                       k=k, sample_frac=1, insert_ratio=1, conf_weight=2000)


def COMPAS(rep, f="dataset/compas_bin_train.csv", t="results_compas/"):
    data_dir = capuchin_path
    data = read_from_csv(os.path.join(data_dir, f))
    output_dir = os.path.join(data_dir+t)
    #data = data[data['race'].isin(['African-American', 'Caucasian'])]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    D_features = ['Race', 'Sex']
    Y_features = ['two_year_recid']
    X_features = ['Age', 'Prior']

    features = D_features+Y_features+X_features
    data = data[features]
    D = [D_features, Y_features, X_features]
    indeps = [D]
    k = 1

    data_split(data, 'two_year_recid', output_dir, k=k, test_size=0.05)

    #labels = [1, 2, 3, 4, 5, 6]
    #data['priors_count'] = pd.cut(data['priors_count'], [-100, 0.5, 2, 3, 4, 5, 100],labels=labels)
    #data['priors_count'] = pd.to_numeric(data['priors_count'], errors='ignore')
   
    indeps = [D]
    for method in [rep]: #'sat','naive','IC','MF',
        inf = Info(data)
        for indep in indeps:
            #print(indep)
            X = indep[0]
            Y = indep[1]
            Z = indep[2]
            mi = inf.CMI(X, Y, Z)
            
        rep = Repair()
        path1 = os.path.join(output_dir,'train_')
        if method == 'sat':
            rep.from_file_indep_repair(path1, X, Y, Z, method=method, n_parti=100,
                                       k=k, sample_frac=1, insert_ratio='hard', conf_weight=1)
            rep.from_file_indep_repair(path1, X, Y, Z, method=method, n_parti=100,
                                       k=k, sample_frac=1, insert_ratio='soft', conf_weight=1)
        else:
            rep.from_file_indep_repair(path1, X, Y, Z, method=method, n_parti=100,
                                       k=k, sample_frac=1, insert_ratio=1, conf_weight=2000)


def German(rep):    
    data_dir = capuchin_path
    data = read_from_csv(os.path.join(data_dir, "dataset/german_bin_train.csv"))
    output_dir = os.path.join(data_dir+"results_german/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    D_features = ['Sex']
    Y_features = ['credit']
    X_features = ['Age','Month','Investment', 'Credit_amount', 'Status', 'Housing', 'Savings',\
                  'Property', 'Credit_history']

    features=D_features+Y_features+X_features
    D = [D_features, Y_features, X_features]
    data=data[features]
    k=1
    data_split(data, 'credit', output_dir, k=k, test_size=0.002)
    indeps = [D]
    for method in [rep]: #'sat','naive','IC','MF',
        inf = Info(data)
        for indep in indeps:

            #print(indep)
            X = indep[0]
            Y = indep[1]
            Z = indep[2]
            mi = inf.CMI(X, Y, Z)
            
        rep = Repair()
        path1 = os.path.join(output_dir,'train_')
        if method == 'sat':
            rep.from_file_indep_repair(path1, X, Y, Z, method=method, n_parti=100,
                                       k=k, sample_frac=1, insert_ratio='hard', conf_weight=1)
            rep.from_file_indep_repair(path1, X, Y, Z, method=method, n_parti=100,
                                       k=k, sample_frac=1, insert_ratio='soft', conf_weight=1)
        else:
            rep.from_file_indep_repair(path1, X, Y, Z, method=method, n_parti=100,
                                       k=k, sample_frac=1, insert_ratio=1, conf_weight=2000)




def vectorize(df,org_df,col):
    freq1=dict()
    freq2=dict()
    shared=dict()
    for name, group in org_df.groupby(col):
        freq1[name]=len(group.index)
    freq=dict()
    for name, group in df.groupby(col):
        freq2[name]=len(group.index)

    for key, value in freq1.items():
        if key in  freq2.keys():
           shared[key]=[value,freq2[key]]
        else:
           shared[key]=[value,0]

    for key, value in freq2.items():
        if key not in  shared.keys():
            shared[key]=[0,value]

    l1=list()
    l2=list()
    #print(shared.values())
    for key, value in shared.items():
        l1.insert(0,value[0])
        l2.insert(0,value[1])
    return l1,l2


def add_remov(df1,df2,col):
    l1,l2=vectorize(df1,df2,col)
    diff=[]
    for i in range(0,len(l1)):
        diff.insert(0,l2[i]-l1[i])
    insert=0
    delete=0
    for item in diff:
        if item<0:
           delete=delete+ item *-1
        else:
           insert=insert+ item
    print('insert: ',insert)
    print('delete: ',delete)
    return insert,delete


def Salimi(dataset, rep):
    print("Dataset:", dataset)
    if dataset == 'adult':
        Adult(rep)
    elif dataset == 'compas':
        COMPAS(rep)
    elif dataset == 'german':
        German(rep)
    elif dataset == 'credit':
        Credit(rep)
        