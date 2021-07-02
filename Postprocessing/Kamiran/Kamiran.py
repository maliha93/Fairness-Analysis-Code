# Load all necessary packages
import sys
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
from warnings import warn

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, CreditDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas, load_preproc_data_credit
from aif360.algorithms.postprocessing.reject_option_classification\
        import RejectOptionClassification
from common_utils import compute_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider
from metric import metric, cd


def Adult():
    ## import dataset
    import time
    protected = 'sex'
    dataset_used = "adult" 
    protected_attribute_used = 1

    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    dataset_orig = load_preproc_data_adult(['sex'])
    index = dataset_orig.feature_names.index(protected)
    metric_name = "Statistical parity difference"
    # Upper and lower bound on the fairness metric used
    metric_ub = 0.05
    metric_lb = -0.05
    np.random.seed(1)

    # Get the dataset and split into train and test
    dataset_orig_tr, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    dataset_orig_train, dataset_orig_valid = dataset_orig_tr.split([0.7], shuffle=False)
    test, _ = dataset_orig_test.convert_to_dataframe(de_dummy_code=True)
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    y_train_pred = lmod.predict(X_train)
    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
        
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                 dataset_orig_valid_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                           +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    s = time.time()

    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                     privileged_groups=privileged_groups, 
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                      num_class_thresh=100, num_ROC_margin=50,
                                      metric_name=metric_name,
                                      metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
    e = time.time()

    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label
    # Metrics for the transformed test set
    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_transf_test_pred.labels)
    y_cd = cd(index, dataset_orig_test_pred, dataset_transf_test_pred.labels, ROC)
    
    test['pred'] = dataset_transf_test_pred.labels
    test.to_csv("results_Kamiran/adult_test_repaired.csv", index=False)
    np.savetxt("results_Kamiran/adult_test_repaired_cd.csv", y_cd, delimiter=",")
    
    
def Compas(f1='', f2='', fname=None):
    ## import dataset
    import time
    protected = 'Race'
    privileged_groups = [{'Race': 1}]
    unprivileged_groups = [{'Race': 0}]
    dataset_orig = load_preproc_data_compas(['Race'], fname=fname)
    index = dataset_orig.feature_names.index(protected)
    metric_name = "Statistical parity difference"

    # Upper and lower bound on the fairness metric used
    metric_ub = 0.05
    metric_lb = -0.05
    np.random.seed(1)

    # Get the dataset and split into train and test
    dataset_orig_tr, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    dataset_orig_train, dataset_orig_valid = dataset_orig_tr.split([0.7], shuffle=False)
    test, _ = dataset_orig_test.convert_to_dataframe(de_dummy_code=True)
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    y_train_pred = lmod.predict(X_train)
    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
        
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                 dataset_orig_valid_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                           +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    s = time.time()
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                     privileged_groups=privileged_groups, 
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                      num_class_thresh=100, num_ROC_margin=50,
                                      metric_name=metric_name,
                                      metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
    e = time.time()
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label
    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_transf_test_pred.labels)
    y_cd = cd(index, dataset_orig_test_pred, dataset_transf_test_pred.labels, ROC)
    
    test['pred'] = dataset_transf_test_pred.labels
    test.to_csv(f1+"results_Kamiran/compas_test_repaired"+f2+".csv", index=False)
    np.savetxt(f1+"results_Kamiran/compas_test_repaired"+f2+"_cd.csv", y_cd, delimiter=",")
    
    
def German():
    ## import dataset
    import time
    protected = 'Sex'
    privileged_groups = [{'Sex': 1}]
    unprivileged_groups = [{'Sex': 0}]
    dataset_orig = load_preproc_data_german(['Sex'])
    index = dataset_orig.feature_names.index(protected)
    metric_name = "Statistical parity difference"    

    # Upper and lower bound on the fairness metric used
    metric_ub = 0.01
    metric_lb = -0.01
    np.random.seed(1)

    # Get the dataset and split into train and test
    dataset_orig_tr, dataset_orig_test = dataset_orig.split([0.7], shuffle=False)
    dataset_orig_train, dataset_orig_valid = dataset_orig_tr.split([0.7], shuffle=False)
    test, _ = dataset_orig_test.convert_to_dataframe(de_dummy_code=True)
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    y_train_pred = lmod.predict(X_train)
    # positive class index
    pos_ind = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    dataset_orig_train_pred.labels = y_train_pred

    dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
    X_valid = scale_orig.transform(dataset_orig_valid_pred.features)
    y_valid = dataset_orig_valid_pred.labels
    dataset_orig_valid_pred.scores = lmod.predict_proba(X_valid)[:,pos_ind].reshape(-1,1)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    X_test = scale_orig.transform(dataset_orig_test_pred.features)
    y_test = dataset_orig_test_pred.labels
    dataset_orig_test_pred.scores = lmod.predict_proba(X_test)[:,pos_ind].reshape(-1,1)

    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        
        fav_inds = dataset_orig_valid_pred.scores > class_thresh
        dataset_orig_valid_pred.labels[fav_inds] = dataset_orig_valid_pred.favorable_label
        dataset_orig_valid_pred.labels[~fav_inds] = dataset_orig_valid_pred.unfavorable_label
        
        classified_metric_orig_valid = ClassificationMetric(dataset_orig_valid,
                                                 dataset_orig_valid_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        
        ba_arr[idx] = 0.5*(classified_metric_orig_valid.true_positive_rate()\
                           +classified_metric_orig_valid.true_negative_rate())

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]

    s = time.time()
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                     privileged_groups=privileged_groups, 
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                      num_class_thresh=100, num_ROC_margin=50,
                                      metric_name=metric_name,
                                      metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
    e = time.time()
    fav_inds = dataset_orig_test_pred.scores > best_class_thresh
    dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
    dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label

    # Metrics for the transformed test set
    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
    metric(index, dataset_orig_test.features, dataset_orig_test.labels, dataset_transf_test_pred.labels)
    y_cd = cd(index, dataset_orig_test_pred, dataset_transf_test_pred.labels, ROC)
    
    test['pred'] = dataset_transf_test_pred.labels
    test.to_csv("results_Kamiran/german_test_repaired.csv", index=False)
    np.savetxt("results_Kamiran/german_test_repaired_cd.csv", y_cd, delimiter=",")
    
    
    
def Kamiran(dataset):
    
    if dataset == 'adult':
        Adult()
    elif dataset == 'compas':
        Compas()
    elif dataset == 'german':
        German()
    


