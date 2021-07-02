from __future__ import print_function, division
import sys, time
import pandas as pd
import numpy as np
from RemoveAlgorithms.Basic import adult, dutch, compas, german
from RemoveAlgorithms.Utility import Utility
from RemoveAlgorithms.Detection import Judge
from RemoveAlgorithms.Naive import Naive
from RemoveAlgorithms.MGraph import MGraph
from RemoveAlgorithms.MData import MData
from RemoveAlgorithms.HandlingConditionalDiscrimination import LM, LPS, Removeall
from RemoveAlgorithms.Blackbox import DI
import pandas as pd
from sklearn.model_selection import train_test_split


def get_group(group, name):
    try:
        return group.get_group(name)
    except:
        return pd.DataFrame(data=None)


def Domains(Atts, df):
    if Atts.__len__() > 0:
        domain = df[Atts].drop_duplicates()
        domain.index = range(domain.__len__())
    else:
        domain = pd.DataFrame()
    return domain


def load(fin, X, Y, Q):
    df = pd.read_csv(fin, dtype=str)
    data, test = train_test_split(df, test_size=0.3, shuffle=False)
    data = data.reset_index(drop=True)
    Xs = Domains(X, data)
    Ys = Domains(Y, data)
    Qs = Domains(Q, data)
    return data, Qs, Xs, Ys


def adult():
    fin = '../data/Adult_binarized.csv'
    C = {'name': 'sex', 'pos': 'Male', 'neg': 'Female'}
    E = {'name': 'income', 'pos': 1, 'neg': 0}
    Q = ['age', 'workclass', 'edu_level', 'marital_status', 'occupation', 'relationship', 'race', 'hours_per_week',
         'native_country']
    X = []
    Y = []

    data, Qs, Xs, Ys = load(fin, X, Y, Q)
    return data, C, E, Qs, Xs, Ys

def compas():
    fin = '../corrupt_data/Compas_binarized_miss_outcome.csv'
    C = {'name': 'Race', 'pos': 'Other', 'neg': 'African-American'}
    E = {'name': 'two_year_recid', 'pos': 1, 'neg': 0}
    Q = ['Sex', 'Prior', 'Age']
    X = []
    Y = []

    data, Qs, Xs, Ys = load(fin, X, Y, Q)
    return data, C, E, Qs, Xs, Ys
    
def german():
    fin = '../data/German_binarized.csv'
    C = {'name': 'Sex', 'pos': 'Male', 'neg': 'Female'}
    E = {'name': 'credit', 'pos': 1, 'neg': 0}
    Q = ['Credit_amount', 'Investment', 'Status', 'Savings', 'Housing', 'Property', 'Month', 'Status', 'Credit_history']
    X = []
    Y = ['Age']

    data, Qs, Xs, Ys = load(fin, X, Y, Q)
    return data, C, E, Qs, Xs, Ys


def testdata():
    fin = '../data/test/test.csv'
    C = {'name': 'X1', 'pos': '1', 'neg': '0'}
    E = {'name': 'X4', 'pos': '1', 'neg': '0'}
    Q = ['X2', 'X3']
    X = []
    Y = []

    data, Qs, Xs, Ys = load(fin, X, Y, Q)
    return data, C, E, Qs, Xs, Ys


def dutch():
    fin = '../data/Dutch.csv'
    C = {'name': 'sex', 'pos': '1', 'neg': '2'}
    E = {'name': 'occupation', 'pos': '2_1', 'neg': '5_4_9'}
    Q = ['age', 'edulevel']
    X = []
    Y = ['householdposition', 'householdsize', 'prevresidence', 'citizenship', 'countrybirth', 'economicstate',
         'curecoactivity', 'maritial']

    data, Qs, Xs, Ys = load(fin, X, Y, Q)
    return data, C, E, Qs, Xs, Ys




def get_dataset(dataset):
    if dataset == 'adult':
        return adult
    elif dataset == 'compas':
        return compas
    elif dataset == 'german':
        return german
    
def Zhang(dataset):

    print('=================================================================')
    print(' MData @ tau = 0.050')
    tau = 0.05
    for Data_Func in [get_dataset(dataset)]:

        print('---------------------------------')
        print('Dataset: %s' % Data_Func.__name__)


        for Remove_func in [MData]:
            data, C, E, Qs, Xs, Ys = Data_Func()

            t = time.clock()
            data_new = Remove_func(tau, data, C, E, Qs, Xs, Ys)
            print('Elapsed time for %s: %f' % (Remove_func.__name__, time.clock() - t))
            data_new.to_csv('../corrupt_results_Zhang_nondiscrimination/'+Data_Func.__name__+"_train_repared_miss_outcome_.csv", index=False)


   
Zhang('compas')