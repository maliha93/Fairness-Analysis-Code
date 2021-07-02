from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from matplotlib import pyplot as plt

import sys
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC as SVM
from sklearn.preprocessing import MinMaxScaler
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import AdultDataset, CompasDataset, GermanDataset, CreditDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from metric import metric


def Adult(rep, fname=None):    
    protected = 'sex'
    ad = AdultDataset(protected_attribute_names=[protected],
        privileged_classes=[['Male']], fname=fname)

    scaler = MinMaxScaler(copy=False)
    train, test = ad.split([0.7], shuffle=False)
    train.features = scaler.fit_transform(train.features)
    test.features = scaler.fit_transform(test.features)
    index = train.feature_names.index(protected)
    
    for level in rep:
        dis = DisparateImpactRemover(repair_level=level)
        train_repd = dis.fit_transform(train)
        test_repd = dis.fit_transform(test)
        train_repd, _ = train_repd.convert_to_dataframe(de_dummy_code=True)
        test_repd, _ = test_repd.convert_to_dataframe(de_dummy_code=True)
        train_repd.to_csv("results_Feldman/adult_train_repaired.csv", index=False)
        test_repd.to_csv("results_Feldman/adult_test_repaired.csv", index=False)
        test, _ = test.convert_to_dataframe(de_dummy_code=True)
        test.to_csv("results_Feldman/adult_test_notrepaired.csv", index=False)
        
def Compas(rep, f1='', f2='', fname=None):    
    protected = 'Race'
    ad = CompasDataset(protected_attribute_names=[protected], fname=fname)

    scaler = MinMaxScaler(copy=False)
    train, test = ad.split([0.7], shuffle=False)
    train.features = scaler.fit_transform(train.features)
    test.features = scaler.fit_transform(test.features)
    index = train.feature_names.index(protected)
    for level in rep:
        dis = DisparateImpactRemover(repair_level=level)
        train_repd = dis.fit_transform(train)
        test_repd = dis.fit_transform(test)
        train_repd, _ = train_repd.convert_to_dataframe(de_dummy_code=True)
        test_repd, _ = test_repd.convert_to_dataframe(de_dummy_code=True)
        train_repd.to_csv(f1+"results_Feldman/compas_train_repaired"+f2+".csv", index=False)
        test_repd.to_csv(f1+"results_Feldman/compas_test_repaired"+f2+".csv", index=False)
        test, _ = test.convert_to_dataframe(de_dummy_code=True)
        test.to_csv(f1+"results_Feldman/compas_test_notrepaired"+f2+".csv", index=False)
        
        
        
def German(rep, fname=None):    
    protected = 'Sex'
    ad = GermanDataset(protected_attribute_names=[protected], fname=fname)

    scaler = MinMaxScaler(copy=False)
    train, test = ad.split([0.7], shuffle=False)
    train.features = scaler.fit_transform(train.features)
    test.features = scaler.fit_transform(test.features)
    index = train.feature_names.index(protected)

    for level in rep:
        dis = DisparateImpactRemover(repair_level=level)
        train_repd = dis.fit_transform(train)
        test_repd = dis.fit_transform(test)
        train_repd, _ = train_repd.convert_to_dataframe(de_dummy_code=True)
        test_repd, _ = test_repd.convert_to_dataframe(de_dummy_code=True)
        train_repd.to_csv("results_Feldman/german_train_repaired.csv", index=False)
        test_repd.to_csv("results_Feldman/german_test_repaired.csv", index=False)
        test, _ = test.convert_to_dataframe(de_dummy_code=True)
        test.to_csv("results_Feldman/german_test_notrepaired.csv", index=False)
               
        
def Feldman(dataset, rep, fname):
    if dataset == 'adult':
        Adult([rep], fname=fname)
    elif dataset == 'compas':
        Compas([rep], fname=fname)
    elif dataset == 'german':
        German([rep], fname=fname)







