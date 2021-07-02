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
            data_new.to_csv('../results_Zhang_nondiscrimination/'+Data_Func.__name__+"_train_repaired.csv", index=False)


   