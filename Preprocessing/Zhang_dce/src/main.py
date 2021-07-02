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

if __name__ == '__main__':
    if sys.argv.__len__() < 2:
        print('''
        To detect discrimination, run python main.py detect
        To compare the effectiveness of algorithms, run python main.py compare
        To compare the utility of with varied tau, run python main.py tau
        ''')

    if 'compare' in sys.argv:
        print('=================================================================')
        print(' MData @ tau = 0.050')
        tau = 0.05
        for Data_Func in [adult, compas, german]:

            print('---------------------------------')
            print('Dataset: %s' % Data_Func.__name__)


            for Remove_func in [MData]:
                data, C, E, Qs, Xs, Ys = Data_Func()

                t = time.clock()
                data_new = Remove_func(tau, data, C, E, Qs, Xs, Ys)
                print('Elapsed time for %s: %f' % (Remove_func.__name__, time.clock() - t))
                data_new.to_csv('../results_Zhang/'+Data_Func.__name__+"_train_repared.csv", index=False)


    if 'tau' in sys.argv:
        print('=======================================================')
        print('Comparison utility with varied tau for MGraph and MData')
        for Data_Func in [adult, dutch]:

            print('---------------------------------')
            print('Dataset: %s' % Data_Func.__name__),

            for Remove_func in [MGraph, MData]:
                print('Apporach: %s' % Remove_func.__name__)
                result = pd.DataFrame(data=np.zeros((4, 5)), index=['0.025', '0.050', '0.075', '0.100'],
                                      columns=['Distance', 'n_T', 'chi', 'Discriminated', 'Non-empty'])
                for tau in [0.025, 0.050, 0.075, 0.100]:
                    data, C, E, Qs, Xs, Ys = Data_Func()
                    data_new = Remove_func(tau, data, C, E, Qs, Xs, Ys)
                    _, _, _, non_empty, largerset, smallerset, _, _ = Judge(data_new, C, E, Qs, tau)
                    result.loc['%0.3f' % tau, :] = Utility(data, data_new) + [largerset + smallerset, non_empty]
                print(result)
