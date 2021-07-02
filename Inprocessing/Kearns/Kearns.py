import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("../")
from aif360.algorithms.inprocessing import GerryFairClassifier
from aif360.algorithms.inprocessing.gerryfair.clean import array_to_tuple
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult,\
load_preproc_data_compas, load_preproc_data_german, load_preproc_data_credit 
from sklearn import svm
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
from aif360.metrics import BinaryLabelDatasetMetric
from IPython.display import Image
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import time 
from metric import metric, cd

def Adult():
    protected = 'sex'
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    data_set = load_preproc_data_adult(['sex'])
    dataset_orig_train, dataset_orig_test = data_set.split([0.7], shuffle=False)
    index = data_set.feature_names.index(protected)

    s = time.time()
    max_iterations = 100
    C = 100
    print_flag = True
    gamma = .005
    predictor = linear_model.LinearRegression()
    fair_model = GerryFairClassifier(C=C, printflag=print_flag, predictor=predictor, gamma=gamma, fairness_def='FN',
                 max_iters=max_iterations, heatmapflag=False)
    # fit method
    fair_model.fit(dataset_orig_train, early_termination=True)
    dataset_yhat = fair_model.predict(dataset_orig_test, threshold=0.5)
    e = time.time()
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_yhat.labels)
    y_cd = cd(index, dataset_orig_test, dataset_yhat.labels, fair_model)
    
    train, test = load_preproc_data_adult(['sex']).split([0.7], shuffle=False) 
    test, _ = test.convert_to_dataframe(de_dummy_code=True)
    test['pred'] = dataset_yhat.labels
    test.to_csv("results_Kearns/adult_test_repaired.csv", index=False)
    np.savetxt("results_Kearns/adult_test_repaired_cd.csv", y_cd, delimiter=",")
    
    
def Compas(f1='', f2='', fname=None):
    protected = 'Race'
    privileged_groups = [{'Race': 1}]
    unprivileged_groups = [{'Race': 0}]
    dataset_orig = load_preproc_data_compas(['Race'], fname=fname)
    index = dataset_orig.feature_names.index(protected)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)

    s = time.time()
    max_iterations = 100
    C = 100
    print_flag = True
    gamma = .005
    #predictor = linear_model.LinearRegression()
    fair_model = GerryFairClassifier(C=C, printflag=print_flag, gamma=gamma, fairness_def='FP',
                 max_iters=max_iterations, heatmapflag=False)
    # fit method
    fair_model.fit(dataset_orig_train, early_termination=True)
    dataset_yhat = fair_model.predict(dataset_orig_test, threshold=0.9898)
    e = time.time()
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_yhat.labels)
    y_cd = cd(index, dataset_orig_test, dataset_yhat.labels, fair_model)
    
    train, test = load_preproc_data_compas(['Race'], fname=fname).split([0.7], shuffle=False) 
    test, _ = test.convert_to_dataframe(de_dummy_code=True)
    test['pred'] = dataset_yhat.labels
    test.to_csv(f1+"results_Kearns/compas_test"+f2+".csv", index=False)
    np.savetxt(f1+"results_Kearns/compas_test_repaired"+f2+"_cd.csv", y_cd, delimiter=",")
    
    
    
def German():
    protected = 'Sex'
    privileged_groups = [{'Sex': 1}]
    unprivileged_groups = [{'Sex': 0}]
    dataset_orig = load_preproc_data_german(['Sex'])
    index = dataset_orig.feature_names.index(protected)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)

    s = time.time()
    max_iterations = 100
    C = 100
    print_flag = True
    gamma = .005
    #predictor = linear_model.LinearRegression()
    fair_model = GerryFairClassifier(C=C, printflag=print_flag, gamma=gamma, fairness_def='FP',
                 max_iters=max_iterations, heatmapflag=False)
    # fit method
    fair_model.fit(dataset_orig_train, early_termination=True)
    dataset_yhat = fair_model.predict(dataset_orig_test, threshold=0.98)
    e = time.time()
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_yhat.labels)
    y_cd = cd(index, dataset_orig_test, dataset_yhat.labels, fair_model)
    
    train, test = load_preproc_data_german(['Sex']).split([0.7], shuffle=False) 
    test, _ = test.convert_to_dataframe(de_dummy_code=True)
    test['pred'] = dataset_yhat.labels
    test.to_csv("results_Kearns/german_test_repaired.csv")
    np.savetxt("results_Kearns/german_test_repaired_cd.csv", y_cd, delimiter=",")
    
    
    
def Kearns(dataset):
    
    if dataset == 'adult':
        Adult()
    elif dataset == 'compas':
        Compas()
    elif dataset == 'german':
        German()
    


