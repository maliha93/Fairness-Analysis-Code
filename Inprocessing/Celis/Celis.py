import sys
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, CreditDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german, load_preproc_data_credit
from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier
from aif360.algorithms.inprocessing.celisMeta.utils import getStats
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from metric import metric, cd

def Adult(fname=None):
    protected = 'sex'
    dataset_orig = AdultDataset(protected_attribute_names=[protected], features_to_keep=['age', 'edu_level'],\
                                categorical_features=[], fname=fname)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    index = dataset_orig_train.feature_names.index(protected)
    debiased_model = MetaFairClassifier(tau=0.8, sensitive_attr="sex")
    debiased_model.fit(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_debiasing_test.labels)
    y_cd = cd(index, dataset_orig_test, dataset_debiasing_test.labels, debiased_model)
    
    train, test =  AdultDataset().split([0.7], shuffle=False)
    test, _ = test.convert_to_dataframe(de_dummy_code=True)
    test['pred'] = dataset_debiasing_test.labels
    test.to_csv("results_Celis/adult_test_repaired.csv", index=False)
    np.savetxt("results_Celis/adult_test_repaired_cd.csv", y_cd, delimiter=",")
    
    
    
def Compas(f1='', f2='', fname=None):    
    protected = 'Race'
    dataset_orig = CompasDataset(fname=fname)
    index = dataset_orig.feature_names.index(protected)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    start = time.time()
    debiased_model = MetaFairClassifier(tau=0.8, sensitive_attr="Race")
    debiased_model.fit(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_debiasing_test.labels)
    y_cd = cd(index, dataset_orig_test, dataset_debiasing_test.labels, debiased_model)
    
    end = time.time()
    train, test =  CompasDataset(fname=fname).split([0.7], shuffle=False)
    test, _ = test.convert_to_dataframe(de_dummy_code=True)
    test['pred'] = dataset_debiasing_test.labels
    test.to_csv(f1+"results_Celis/compas_test"+f2+".csv", index=False)
    np.savetxt(f1+"results_Celis/compas_test_repaired"+f2+"_cd.csv", y_cd, delimiter=",")
    
    

def German(fname=None):    
    protected = 'Sex'
    dataset_orig = GermanDataset(features_to_drop=['Housing', 'Property', ],\
                                categorical_features=['Status', 'Credit_history', 'Savings'], fname=fname)
    index = dataset_orig.feature_names.index(protected)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    start = time.time()
    debiased_model = MetaFairClassifier(tau=0.8, sensitive_attr="Sex")
    debiased_model.fit(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    end = time.time()
    index = dataset_orig.feature_names.index(protected)
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_debiasing_test.labels)
    y_cd = cd(index, dataset_orig_test, dataset_debiasing_test.labels, debiased_model)
    
    train, test =  GermanDataset().split([0.7], shuffle=False)
    test, _ = test.convert_to_dataframe(de_dummy_code=True)
    test['pred'] = dataset_debiasing_test.labels
    test.to_csv("results_Celis/german_test_repaired.csv", index=False)
    np.savetxt("results_Celis/german_test_repaired_cd.csv", y_cd, delimiter=",")
    
    
def Celis(dataset, fname=None):
    
    if dataset == 'adult':
        Adult(fname=fname)
    elif dataset == 'compas':
        Compas(fname=fname)
    elif dataset == 'german':
        German(fname=fname)