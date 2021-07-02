# Load all necessary packages
import sys
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, CreditDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german, load_preproc_data_credit
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import tensorflow as tf
from metric import metric, cd
from time import time
import numpy as np


def Adult():
    # Get the dataset and split into train and test
    protected = 'sex'
    dataset_orig = AdultDataset(protected_attribute_names=['sex'], privileged_classes=[['Male']], \
                categorical_features=[ 'race', 'workclass', 'marital_status', 'occupation', 'relationship',\
                'native_country'], features_to_keep=['age', 'edu_level', 'race', 'sex', 'workclass', 'occupation'],\
                                features_to_drop=['marital_status', 'relationship', 'native_country']  )
    #print(dataset_orig)
    index = dataset_orig.feature_names.index(protected)
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)

    min_max_scaler = MaxAbsScaler()
    dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
    metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                 unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups)
    metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                                 unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups)
    
    s = time()     
    sess = tf.Session()
    # Learn parameters with debias set to True
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                              unprivileged_groups = unprivileged_groups,
                              scope_name='debiased_classifier',
                              debias=True,
                              sess=sess)
    debiased_model.fit(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    e = time()
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_debiasing_test.labels)
    y_cd = cd(index, dataset_orig_test, dataset_debiasing_test.labels, debiased_model)
    sess.close()
    tf.reset_default_graph()
    
    train, test = AdultDataset().split([0.7], shuffle=False) 
    test, _ = test.convert_to_dataframe(de_dummy_code=True)
    test['pred'] = dataset_debiasing_test.labels
    test.to_csv("results_Zhang/adult_test_repaired.csv", index=False)
    np.savetxt("results_Zhang/adult_test_repaired_cd.csv", y_cd, delimiter=",")
    
    
def Compas(f1='', f2='', fname=None):
    # Get the dataset and split into train and test
    protected = 'Race'
    privileged_groups = [{'Race': 1}]
    unprivileged_groups = [{'Race': 0}]
    dataset_orig = load_preproc_data_compas(['Race'], fname=fname)
    index = dataset_orig.feature_names.index(protected)

    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    min_max_scaler = MaxAbsScaler()
    dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
    metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                 unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups)
    metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                                 unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups)
     
    s = time()
    sess = tf.Session()
    # Learn parameters with debias set to True
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                              unprivileged_groups = unprivileged_groups,
                              scope_name='debiased_classifier',
                              debias=True,
                              sess=sess)
    debiased_model.fit(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    e = time()
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_debiasing_test.labels)
    y_cd = cd(index, dataset_orig_test, dataset_debiasing_test.labels, debiased_model)
    sess.close()
    tf.reset_default_graph()
    
    train, test = load_preproc_data_compas(['Race'], fname=fname).split([0.7], shuffle=False) 
    test, _ = test.convert_to_dataframe(de_dummy_code=True)
    test['pred'] = dataset_debiasing_test.labels
    test.to_csv(f1+"results_Zhang/compas_test"+f2+".csv", index=False)
    np.savetxt(f1+"results_Zhang/compas_test_repaired"+f2+"_cd.csv", y_cd, delimiter=",")
    
    
def German():
    # Get the dataset and split into train and test
    protected = 'Sex'
    privileged_groups = [{'Sex': 1}]
    unprivileged_groups = [{'Sex': 0}]
    dataset_orig = GermanDataset()
    index = dataset_orig.feature_names.index(protected)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    min_max_scaler = MaxAbsScaler()
    dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
    metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                 unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups)
    metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                                 unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups)
          
    s = time()
    sess = tf.Session()
    # Learn parameters with debias set to True
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                              unprivileged_groups = unprivileged_groups,
                              scope_name='debiased_classifier',
                              debias=True,
                              sess=sess)
    debiased_model.fit(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)
    e = time()
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_debiasing_test.labels)
    y_cd = cd(index, dataset_orig_test, dataset_debiasing_test.labels, debiased_model)
    sess.close()
    tf.reset_default_graph()
    
    train, test = GermanDataset().split([0.7], shuffle=False) 
    test, _ = test.convert_to_dataframe(de_dummy_code=True)
    test['pred'] = dataset_debiasing_test.labels
    test.to_csv("results_Zhang/german_test_repaired.csv", index=False)
    np.savetxt("results_Zhang/german_test_repaired_cd.csv", y_cd, delimiter=",")
    
    
def Zhang(dataset):
    
    if dataset == 'adult':
        Adult()
    elif dataset == 'compas':
        Compas()
    elif dataset == 'german':
        German()
   


